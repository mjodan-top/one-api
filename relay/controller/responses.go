package controller

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/songquanpeng/one-api/common"
	"github.com/songquanpeng/one-api/common/config"
	"github.com/songquanpeng/one-api/common/logger"
	"github.com/songquanpeng/one-api/common/render"
	"github.com/songquanpeng/one-api/relay"
	"github.com/songquanpeng/one-api/relay/adaptor"
	"github.com/songquanpeng/one-api/relay/adaptor/openai"
	"github.com/songquanpeng/one-api/relay/apitype"
	"github.com/songquanpeng/one-api/relay/billing"
	billingratio "github.com/songquanpeng/one-api/relay/billing/ratio"
	"github.com/songquanpeng/one-api/relay/channeltype"
	"github.com/songquanpeng/one-api/relay/meta"
	"github.com/songquanpeng/one-api/relay/model"
	"github.com/songquanpeng/one-api/relay/relaymode"
)

// ResponsesHelper handles /v1/responses requests
func ResponsesHelper(c *gin.Context, relayMode int) *model.ErrorWithStatusCode {
	ctx := c.Request.Context()
	metaObj := meta.GetByContext(c)

	// Parse Responses API request
	var responsesReq model.ResponsesRequest
	if err := common.UnmarshalBodyReusable(c, &responsesReq); err != nil {
		logger.Errorf(ctx, "Failed to parse responses request: %s", err.Error())
		return openai.ErrorWrapper(err, "invalid_request", http.StatusBadRequest)
	}

	// Convert Responses request to Chat Completions request
	chatReq := convertResponsesToChatCompletions(&responsesReq)
	metaObj.IsStream = chatReq.Stream

	// Map model name
	metaObj.OriginModelName = chatReq.Model
	chatReq.Model, _ = getMappedModelName(chatReq.Model, metaObj.ModelMapping)
	metaObj.ActualModelName = chatReq.Model

	// Get ratios
	modelRatio := billingratio.GetModelRatio(chatReq.Model, metaObj.ChannelType)
	groupRatio := billingratio.GetGroupRatio(metaObj.Group)
	ratio := modelRatio * groupRatio

	// Calculate prompt tokens
	promptTokens := openai.CountTokenMessages(chatReq.Messages, chatReq.Model)
	metaObj.PromptTokens = promptTokens

	// Pre-consume quota
	preConsumedQuota, bizErr := preConsumeQuota(ctx, chatReq, promptTokens, ratio, metaObj)
	if bizErr != nil {
		logger.Warnf(ctx, "preConsumeQuota failed: %+v", *bizErr)
		return bizErr
	}

	// Override request URL path to chat/completions since we converted the request
	metaObj.RequestURLPath = "/v1/chat/completions"

	// Get adaptor
	adaptorObj := relay.GetAdaptor(metaObj.APIType)
	if adaptorObj == nil {
		return openai.ErrorWrapper(fmt.Errorf("invalid api type: %d", metaObj.APIType), "invalid_api_type", http.StatusBadRequest)
	}
	adaptorObj.Init(metaObj)

	// Get request body
	requestBody, err := getResponsesRequestBody(c, metaObj, chatReq, adaptorObj)
	if err != nil {
		return openai.ErrorWrapper(err, "convert_request_failed", http.StatusInternalServerError)
	}

	// Do request
	resp, err := adaptorObj.DoRequest(c, metaObj, requestBody)
	if err != nil {
		logger.Errorf(ctx, "DoRequest failed: %s", err.Error())
		return openai.ErrorWrapper(err, "do_request_failed", http.StatusInternalServerError)
	}

	if isErrorHappened(metaObj, resp) {
		billing.ReturnPreConsumedQuota(ctx, preConsumedQuota, metaObj.TokenId)
		return RelayErrorHandler(resp)
	}

	// Do response with conversion
	usage, respErr := doResponsesResponse(c, resp, metaObj, adaptorObj, chatReq.Stream)
	if respErr != nil {
		logger.Errorf(ctx, "respErr is not nil: %+v", respErr)
		billing.ReturnPreConsumedQuota(ctx, preConsumedQuota, metaObj.TokenId)
		return respErr
	}

	// Post-consume quota
	go postConsumeQuota(ctx, usage, metaObj, chatReq, ratio, preConsumedQuota, modelRatio, groupRatio, false)
	return nil
}

func getResponsesRequestBody(c *gin.Context, metaObj *meta.Meta, chatReq *model.GeneralOpenAIRequest, adaptorObj adaptor.Adaptor) (io.Reader, error) {
	if !config.EnforceIncludeUsage &&
		metaObj.APIType == apitype.OpenAI &&
		metaObj.OriginModelName == metaObj.ActualModelName &&
		metaObj.ChannelType != channeltype.Baichuan &&
		metaObj.ForcedSystemPrompt == "" {

		// Need to ensure stream_options for usage
		if chatReq.Stream {
			if chatReq.StreamOptions == nil {
				chatReq.StreamOptions = &model.StreamOptions{}
			}
			chatReq.StreamOptions.IncludeUsage = true
		}
		jsonData, err := json.Marshal(chatReq)
		if err != nil {
			return nil, err
		}
		return bytes.NewBuffer(jsonData), nil
	}

	// Use adaptor's ConvertRequest
	convertedRequest, err := adaptorObj.ConvertRequest(c, relaymode.ChatCompletions, chatReq)
	if err != nil {
		return nil, err
	}
	jsonData, err := json.Marshal(convertedRequest)
	if err != nil {
		return nil, err
	}
	return bytes.NewBuffer(jsonData), nil
}

// captureWriter intercepts what adaptor.DoResponse writes to gin.Context
type captureWriter struct {
	gin.ResponseWriter
	body       *bytes.Buffer
	statusCode int
}

func newCaptureWriter(w gin.ResponseWriter) *captureWriter {
	return &captureWriter{
		ResponseWriter: w,
		body:           &bytes.Buffer{},
	}
}

func (cw *captureWriter) Write(data []byte) (int, error) {
	return cw.body.Write(data)
}

func (cw *captureWriter) WriteString(s string) (int, error) {
	return cw.body.WriteString(s)
}

func (cw *captureWriter) WriteHeader(code int) {
	cw.statusCode = code
}

func doResponsesResponse(c *gin.Context, resp *http.Response, metaObj *meta.Meta, adaptorObj adaptor.Adaptor, isStream bool) (*model.Usage, *model.ErrorWithStatusCode) {
	if metaObj.APIType == apitype.OpenAI {
		// OpenAI format: parse directly
		if isStream {
			return responsesStreamHandler(c, resp)
		}
		return responsesNonStreamHandler(c, resp)
	}

	// Non-OpenAI (Anthropic, etc.): use adaptor to convert response to OpenAI format first
	// Capture adaptor's output via a buffer writer
	realWriter := c.Writer
	cw := newCaptureWriter(realWriter)
	c.Writer = cw

	usage, respErr := adaptorObj.DoResponse(c, resp, metaObj)

	// Restore real writer
	c.Writer = realWriter

	if respErr != nil {
		return usage, respErr
	}

	captured := cw.body.Bytes()

	if isStream {
		return responsesConvertStreamFromOpenAI(c, captured, usage)
	}
	return responsesConvertNonStreamFromOpenAI(c, captured, usage)
}

// responsesConvertNonStreamFromOpenAI converts captured OpenAI chat completion JSON to Responses format
func responsesConvertNonStreamFromOpenAI(c *gin.Context, data []byte, adaptorUsage *model.Usage) (*model.Usage, *model.ErrorWithStatusCode) {
	var chatResp SlimTextResponse
	if err := json.Unmarshal(data, &chatResp); err != nil {
		return nil, openai.ErrorWrapper(err, "unmarshal_response_failed", http.StatusInternalServerError)
	}

	if chatResp.Error.Type != "" {
		return nil, &model.ErrorWithStatusCode{
			Error:      chatResp.Error,
			StatusCode: http.StatusBadRequest,
		}
	}

	responseId := generateResponseId()
	outputItems := make([]interface{}, 0)

	for i, choice := range chatResp.Choices {
		messageItem := map[string]interface{}{
			"type":    "message",
			"id":      fmt.Sprintf("msg_%d", i),
			"role":    "assistant",
			"status":  "completed",
			"content": []interface{}{},
		}

		if choice.Message.Content != nil {
			textContent := choice.Message.StringContent()
			if textContent != "" {
				messageItem["content"] = []interface{}{
					map[string]interface{}{
						"type": "output_text",
						"text": textContent,
					},
				}
			}
		}

		for _, tc := range choice.Message.ToolCalls {
			var argsStr string
			switch v := tc.Function.Arguments.(type) {
			case string:
				argsStr = v
			default:
				argsBytes, _ := json.Marshal(v)
				argsStr = string(argsBytes)
			}
			toolItem := map[string]interface{}{
				"type":      "function_call",
				"id":        fmt.Sprintf("fc_%s", tc.Id),
				"call_id":   tc.Id,
				"name":      tc.Function.Name,
				"arguments": argsStr,
				"status":    "completed",
			}
			outputItems = append(outputItems, toolItem)
		}

		outputItems = append(outputItems, messageItem)
	}

	finalUsage := &chatResp.Usage
	if adaptorUsage != nil {
		finalUsage = adaptorUsage
	}

	responsesResp := map[string]interface{}{
		"id":      responseId,
		"object":  "response",
		"created": chatResp.Created,
		"model":   chatResp.Model,
		"status":  "completed",
		"output":  outputItems,
		"usage":   convertUsageToResponsesFormat(finalUsage),
	}

	c.JSON(http.StatusOK, responsesResp)
	return finalUsage, nil
}

// responsesConvertStreamFromOpenAI converts captured OpenAI SSE stream data to Responses SSE format
func responsesConvertStreamFromOpenAI(c *gin.Context, data []byte, adaptorUsage *model.Usage) (*model.Usage, *model.ErrorWithStatusCode) {
	responseId := generateResponseId()
	var usage *model.Usage
	var outputText strings.Builder
	assistantItemId := fmt.Sprintf("msg_%s", uuid.New().String()[:24])
	toolCallIds := make(map[int]string)

	common.SetEventStreamHeaders(c)

	// Send response.created event
	sendResponsesEvent(c, "response.created", map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":     responseId,
			"object": "response",
			"status": "in_progress",
		},
	})

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		dataContent := strings.TrimPrefix(line, "data: ")
		if dataContent == "[DONE]" {
			break
		}

		var streamResp ChatCompletionsStreamResponse
		if err := json.Unmarshal([]byte(dataContent), &streamResp); err != nil {
			continue
		}

		if streamResp.Usage != nil {
			usage = streamResp.Usage
		}

		for _, choice := range streamResp.Choices {
			if contentStr, ok := choice.Delta.Content.(string); ok && contentStr != "" {
				outputText.WriteString(contentStr)
				sendResponsesEvent(c, "response.output_text.delta", map[string]interface{}{
					"type":         "response.output_text.delta",
					"delta":        map[string]string{"text": contentStr},
					"item_id":      assistantItemId,
					"output_index": 0,
				})
			}

			for i, tc := range choice.Delta.ToolCalls {
				if tc.Id != "" {
					toolCallIds[i] = tc.Id
					sendResponsesEvent(c, "response.output_item.added", map[string]interface{}{
						"type": "response.output_item.added",
						"item": map[string]interface{}{
							"type":    "function_call",
							"id":      fmt.Sprintf("fc_%s", tc.Id),
							"call_id": tc.Id,
							"name":    tc.Function.Name,
							"status":  "in_progress",
						},
					})
				}
				if tc.Function.Arguments != nil {
					var argsStr string
					switch v := tc.Function.Arguments.(type) {
					case string:
						argsStr = v
					default:
						argsBytes, _ := json.Marshal(v)
						argsStr = string(argsBytes)
					}
					if argsStr != "" {
						sendResponsesEvent(c, "response.function_call_arguments.delta", map[string]interface{}{
							"type":    "response.function_call_arguments.delta",
							"item_id": fmt.Sprintf("fc_%s", toolCallIds[i]),
							"delta": map[string]string{
								"arguments": argsStr,
							},
						})
					}
				}
			}
		}
	}

	if adaptorUsage != nil {
		usage = adaptorUsage
	}

	// Send output_item.done
	sendResponsesEvent(c, "response.output_item.done", map[string]interface{}{
		"type": "response.output_item.done",
		"item": map[string]interface{}{
			"type":   "message",
			"id":     assistantItemId,
			"role":   "assistant",
			"status": "completed",
			"content": []interface{}{
				map[string]interface{}{
					"type": "output_text",
					"text": outputText.String(),
				},
			},
		},
	})

	// Send response.completed
	sendResponsesEvent(c, "response.completed", map[string]interface{}{
		"type": "response.completed",
		"response": map[string]interface{}{
			"id":     responseId,
			"object": "response",
			"status": "completed",
			"output": []interface{}{},
			"usage":  convertUsageToResponsesFormat(usage),
		},
	})
	render.Done(c)

	return usage, nil
}

// responsesStreamHandler handles stream responses from OpenAI-format upstream directly
func responsesStreamHandler(c *gin.Context, resp *http.Response) (*model.Usage, *model.ErrorWithStatusCode) {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Split(bufio.ScanLines)

	responseId := generateResponseId()
	var usage *model.Usage
	var outputText strings.Builder
	assistantItemId := fmt.Sprintf("msg_%s", uuid.New().String()[:24])
	toolCallIds := make(map[int]string)

	common.SetEventStreamHeaders(c)

	// Send response.created event
	sendResponsesEvent(c, "response.created", map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":     responseId,
			"object": "response",
			"status": "in_progress",
		},
	})

	for scanner.Scan() {
		data := scanner.Text()
		if !strings.HasPrefix(data, "data: ") {
			continue
		}

		dataContent := strings.TrimPrefix(data, "data: ")
		if dataContent == "[DONE]" {
			sendResponsesEvent(c, "response.output_item.done", map[string]interface{}{
				"type": "response.output_item.done",
				"item": map[string]interface{}{
					"type":   "message",
					"id":     assistantItemId,
					"role":   "assistant",
					"status": "completed",
					"content": []interface{}{
						map[string]interface{}{
							"type": "output_text",
							"text": outputText.String(),
						},
					},
				},
			})

			sendResponsesEvent(c, "response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id":     responseId,
					"object": "response",
					"status": "completed",
					"output": []interface{}{},
					"usage":  convertUsageToResponsesFormat(usage),
				},
			})
			render.Done(c)
			break
		}

		var streamResp ChatCompletionsStreamResponse
		if err := json.Unmarshal([]byte(dataContent), &streamResp); err != nil {
			logger.SysError("error unmarshalling stream response: " + err.Error())
			continue
		}

		if streamResp.Usage != nil {
			usage = streamResp.Usage
		}

		for _, choice := range streamResp.Choices {
			if contentStr, ok := choice.Delta.Content.(string); ok && contentStr != "" {
				outputText.WriteString(contentStr)
				sendResponsesEvent(c, "response.output_text.delta", map[string]interface{}{
					"type":         "response.output_text.delta",
					"delta":        map[string]string{"text": contentStr},
					"item_id":      assistantItemId,
					"output_index": 0,
				})
			}

			for i, tc := range choice.Delta.ToolCalls {
				if tc.Id != "" {
					toolCallIds[i] = tc.Id
					sendResponsesEvent(c, "response.output_item.added", map[string]interface{}{
						"type": "response.output_item.added",
						"item": map[string]interface{}{
							"type":    "function_call",
							"id":      fmt.Sprintf("fc_%s", tc.Id),
							"call_id": tc.Id,
							"name":    tc.Function.Name,
							"status":  "in_progress",
						},
					})
				}
				if tc.Function.Arguments != nil {
					var argsStr string
					switch v := tc.Function.Arguments.(type) {
					case string:
						argsStr = v
					default:
						argsBytes, _ := json.Marshal(v)
						argsStr = string(argsBytes)
					}
					if argsStr != "" {
						sendResponsesEvent(c, "response.function_call_arguments.delta", map[string]interface{}{
							"type":    "response.function_call_arguments.delta",
							"item_id": fmt.Sprintf("fc_%s", toolCallIds[i]),
							"delta": map[string]string{
								"arguments": argsStr,
							},
						})
					}
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		logger.SysError("error reading stream: " + err.Error())
	}
	resp.Body.Close()
	return usage, nil
}

// responsesNonStreamHandler handles non-stream responses from OpenAI-format upstream directly
func responsesNonStreamHandler(c *gin.Context, resp *http.Response) (*model.Usage, *model.ErrorWithStatusCode) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, openai.ErrorWrapper(err, "read_response_failed", http.StatusInternalServerError)
	}
	resp.Body.Close()

	return responsesConvertNonStreamFromOpenAI(c, body, nil)
}

// Helper functions
func convertResponsesToChatCompletions(req *model.ResponsesRequest) *model.GeneralOpenAIRequest {
	chatReq := &model.GeneralOpenAIRequest{
		Model:       req.Model,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Tools:       model.ResponsesToolsToTools(req.Tools),
	}

	if req.MaxTokens != nil {
		chatReq.MaxTokens = *req.MaxTokens
	}

	// Convert instructions to system message
	var messages []model.Message
	if req.Instructions != nil && *req.Instructions != "" {
		messages = append(messages, model.Message{
			Role:    "system",
			Content: *req.Instructions,
		})
	}

	// Convert input items to messages
	for _, item := range req.Input {
		switch item.Type {
		case "message":
			msg := model.Message{Role: item.Role}
			if len(item.Content) > 0 {
				if len(item.Content) == 1 && item.Content[0].Type == "input_text" {
					msg.Content = item.Content[0].Text
				} else {
					var content []interface{}
					for _, c := range item.Content {
						switch c.Type {
						case "input_text":
							content = append(content, map[string]interface{}{
								"type": "text",
								"text": c.Text,
							})
						case "input_image":
							if c.ImageURL != nil {
								content = append(content, map[string]interface{}{
									"type": "image_url",
									"image_url": map[string]string{
										"url": c.ImageURL.Url,
									},
								})
							}
						}
					}
					msg.Content = content
				}
			}
			messages = append(messages, msg)

		case "function_call":
			messages = append(messages, model.Message{
				Role: "assistant",
				ToolCalls: []model.Tool{
					{
						Id:   item.Id,
						Type: "function",
						Function: model.Function{
							Name:      item.Name,
							Arguments: item.Arguments,
						},
					},
				},
			})

		case "function_call_output":
			messages = append(messages, model.Message{
				Role:       "tool",
				ToolCallId: item.CallId,
				Content:    item.Output,
			})
		}
	}

	chatReq.Messages = messages
	return chatReq
}

// convertUsageToResponsesFormat converts Usage to Responses API format with input_tokens/output_tokens
func convertUsageToResponsesFormat(usage *model.Usage) map[string]interface{} {
	if usage == nil {
		return map[string]interface{}{
			"input_tokens":  0,
			"output_tokens": 0,
			"total_tokens":  0,
		}
	}
	return map[string]interface{}{
		"input_tokens":  usage.PromptTokens,
		"output_tokens": usage.CompletionTokens,
		"total_tokens":  usage.TotalTokens,
	}
}

func generateResponseId() string {
	return fmt.Sprintf("resp_%s", uuid.New().String()[:24])
}

func sendResponsesEvent(c *gin.Context, eventType string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	c.Render(-1, common.CustomEvent{
		Event: "event: " + eventType,
		Data:  "data: " + string(jsonData),
	})
	c.Writer.Flush()
}

// SlimTextResponse for parsing non-stream responses
type SlimTextResponse struct {
	Id      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int           `json:"index"`
		Message      model.Message `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage model.Usage `json:"usage"`
	Error model.Error `json:"error"`
}

// ChatCompletionsStreamResponse for parsing stream responses
type ChatCompletionsStreamResponse struct {
	Id      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int           `json:"index"`
		Delta        model.Message `json:"delta"`
		FinishReason *string       `json:"finish_reason"`
	} `json:"choices"`
	Usage *model.Usage `json:"usage,omitempty"`
}
