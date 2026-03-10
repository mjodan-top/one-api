package model

import "encoding/json"

// ResponsesRequest represents the OpenAI Responses API request format
type ResponsesRequest struct {
	Model        string              `json:"model"`
	Input        ResponsesInput      `json:"input"`
	Instructions *string             `json:"instructions,omitempty"`
	Tools        []Tool              `json:"tools,omitempty"`
	Stream       bool                `json:"stream,omitempty"`
	Temperature  *float64            `json:"temperature,omitempty"`
	TopP         *float64            `json:"top_p,omitempty"`
	MaxTokens    *int                `json:"max_output_tokens,omitempty"`
}

// ResponsesInput can be a string or an array of ResponsesInputItem
type ResponsesInput []ResponsesInputItem

func (r *ResponsesInput) UnmarshalJSON(data []byte) error {
	// Try as string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*r = []ResponsesInputItem{
			{
				Type: "message",
				Role: "user",
				Content: []ContentItem{
					{Type: "input_text", Text: s},
				},
			},
		}
		return nil
	}
	// Try as array
	var items []ResponsesInputItem
	if err := json.Unmarshal(data, &items); err != nil {
		return err
	}
	*r = items
	return nil
}

// ResponsesInputItem represents an item in the input array
type ResponsesInputItem struct {
	Type    string           `json:"type"`
	Role    string           `json:"role,omitempty"`
	Content []ContentItem    `json:"content,omitempty"`
	CallId  string           `json:"call_id,omitempty"`
	Output  string           `json:"output,omitempty"`
	Id      string           `json:"id,omitempty"`
	Name    string           `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
}

// ContentItem represents content in input/output messages
type ContentItem struct {
	Type     string     `json:"type"`
	Text     string     `json:"text,omitempty"`
	ImageURL *ImageURL  `json:"image_url,omitempty"`
}

// ResponsesEvent represents SSE event types for Responses API
type ResponsesEvent struct {
	Type     string          `json:"type"`
	Response *ResponseObject `json:"response,omitempty"`
	Item     *ResponseItem   `json:"item,omitempty"`
	Delta    *ResponseDelta  `json:"delta,omitempty"`
}

// ResponseObject represents the response object in events
type ResponseObject struct {
	Id           string          `json:"id,omitempty"`
	Object       string          `json:"object,omitempty"`
	CreatedAt    int64           `json:"created_at,omitempty"`
	Status       string          `json:"status,omitempty"`
	Output       []ResponseItem  `json:"output,omitempty"`
	Usage        *Usage          `json:"usage,omitempty"`
}

// ResponseItem represents an output item
type ResponseItem struct {
	Type       string         `json:"type"`
	Id         string         `json:"id,omitempty"`
	Role       string         `json:"role,omitempty"`
	Content    []ContentItem  `json:"content,omitempty"`
	CallId     string         `json:"call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
	Arguments  string         `json:"arguments,omitempty"`
	Status     string         `json:"status,omitempty"`
}

// ResponseDelta represents streaming delta content
type ResponseDelta struct {
	Type    string `json:"type,omitempty"`
	Text    string `json:"text,omitempty"`
	Index   int    `json:"index,omitempty"`
}

// ResponseToolCall represents a tool call in the response
type ResponseToolCall struct {
	Id        string `json:"id,omitempty"`
	Type      string `json:"type"`
	Function  struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}