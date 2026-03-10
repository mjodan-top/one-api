package controller

import (
	"bufio"
	"bytes"
	"context"
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

func doResponsesResponse(c *gin.Context, resp *http.Response, metaObj *meta.Meta, adaptorObj adaptor.Adaptor, isStream bool) (*model.Usage, *model.ErrorWithStatusCode) {
	if isStream {
		return responsesStreamHandler(c, resp, metaObj)
	}
	return responsesNonStreamHandler(c, resp, metaObj)
}

func responsesStreamHandler(c *gin.Context, resp *http.Response, metaObj *meta.Meta) (*model.Usage, *model.ErrorWithStatusCode) {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Split(bufio.ScanLines)

	responseId := generateResponseId()
	var usage *model.Usage
	var outputText strings.Builder
	var assistantItemId string
	var toolCallIds = make(map[int]string)

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

	// Create assistant message item
	assistantItemId = fmt.Sprintf("msg_%s", uuid.New().String()[:24])

	for scanner.Scan() {
		data := scanner.Text()
		if !strings.HasPrefix(data, "data: ") {
			continue
		}

		dataContent := strings.TrimPrefix(data, "data: ")
		if dataContent == "[DONE]" {
			// Send output_item.done for assistant message
			sendResponsesEvent(c, "response.output_item.done", map[string]interface{}{
				"type": "response.output_item.done",
				"item": map[string]interface{}{
					"type":    "message",
					"id":      assistantItemId,
					"role":    "assistant",
					"status":  "completed",
					"content": []interface{}{
						map[string]interface{}{
							"type": "output_text",
							"text": outputText.String(),
						},
					},
				},
			})

			// Send response.completed event
			sendResponsesEvent(c, "response.completed", map[string]interface{}{
				"type": "response.completed",
				"response": map[string]interface{}{
					"id":       responseId,
					"object":   "response",
					"status":   "completed",
					"output":   []interface{}{},
					"usage":    usage,
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

		// Handle usage
		if streamResp.Usage != nil {
			usage = streamResp.Usage
		}

		// Handle choices
		for _, choice := range streamResp.Choices {
			// Handle content delta
			if choice.Delta.Content != nil && *choice.Delta.Content != "" {
				text := *choice.Delta.Content
				outputText.WriteString(text)

				sendResponsesEvent(c, "response.output_text.delta", map[string]interface{}{
					"type":  "response.output_text.delta",
					"delta": map[string]string{"text": text},
					"item_id": assistantItemId,
					"output_index": 0,
				})
			}

			// Handle tool calls
			for i, tc := range choice.Delta.ToolCalls {
				if tc.Id != "" {
					// New tool call
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
				if tc.Function.Arguments != "" {
					sendResponsesEvent(c, "response.function_call_arguments.delta", map[string]interface{}{
						"type": "response.function_call_arguments.delta",
						"item_id": fmt.Sprintf("fc_%s", toolCallIds[i]),
						"delta": map[string]string{
							"arguments": tc.Function.Arguments,
						},
					})
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

func responsesNonStreamHandler(c *gin.Context, resp *http.Response, metaObj *meta.Meta) (*model.Usage, *model.ErrorWithStatusCode) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, openai.ErrorWrapper(err, "read_response_failed", http.StatusInternalServerError)
	}
	resp.Body.Close()

	var chatResp SlimTextResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, openai.ErrorWrapper(err, "unmarshal_response_failed", http.StatusInternalServerError)
	}

	if chatResp.Error.Type != "" {
		return nil, &model.ErrorWithStatusCode{
			Error:      chatResp.Error,
			StatusCode: resp.StatusCode,
		}
	}

	// Convert to Responses API format
	responseId := generateResponseId()
	outputItems := make([]interface{}, 0)

	for i, choice := range chatResp.Choices {
		// Create message item
		messageItem := map[string]interface{}{
			"type":    "message",
			"id":      fmt.Sprintf("msg_%d", i),
			"role":    "assistant",
			"status":  "completed",
			"content": []interface{}{},
		}

		// Add text content
		if choice.Message.Content != nil {
			messageItem["content"] = []interface{}{
				map[string]interface{}{
					"type": "output_text",
					"text": choice.Message.Content,
				},
			}
		}

		// Add tool call items first
		for _, tc := range choice.Message.ToolCalls {
			toolItem := map[string]interface{}{
				"type":      "function_call",
				"id":        fmt.Sprintf("fc_%s", tc.Id),
				"call_id":   tc.Id,
				"name":      tc.Function.Name,
				"arguments": tc.Function.Arguments,
				"status":    "completed",
			}
			outputItems = append(outputItems, toolItem)
		}

		outputItems = append(outputItems, messageItem)
	}

	responsesResp := map[string]interface{}{
		"id":      responseId,
		"object":  "response",
		"created": chatResp.Created,
		"model":   chatResp.Model,
		"status":  "completed",
		"output":  outputItems,
		"usage":   chatResp.Usage,
	}

	c.JSON(http.StatusOK, responsesResp)
	return &chatResp.Usage, nil
}

// Helper functions
func convertResponsesToChatCompletions(req *model.ResponsesRequest) *model.GeneralOpenAIRequest {
	chatReq := &model.GeneralOpenAIRequest{
		Model:       req.Model,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Tools:       req.Tools,
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
	Id      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []struct {
		Index        int          `json:"index"`
		Message      model.Message `json:"message"`
		FinishReason string       `json:"finish_reason"`
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
		Index        int          `json:"index"`
		Delta        model.Message `json:"delta"`
		FinishReason *string      `json:"finish_reason"`
	} `json:"choices"`
	Usage *model.Usage `json:"usage,omitempty"`
}