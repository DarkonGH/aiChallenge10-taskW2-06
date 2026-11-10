package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// ToolCall is one function call requested by the model.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Message is one stored dialogue message. It is persisted as-is in the session
// log and converted to wireMessage before an API request.
type Message struct {
	ID         string     `json:"id,omitempty"`
	Seq        int64      `json:"seq,omitempty"`
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
	Model      string     `json:"model,omitempty"`
	TS         time.Time  `json:"ts,omitempty"`
	Tokens     int        `json:"tokens_est,omitempty"`
}

// wireMessage is the API shape. Content is a pointer because the OpenAI
// protocol sends "content": null for an assistant turn that only has tool calls.
type wireMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

func toWire(m Message) wireMessage {
	var content *string
	if !(m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) > 0) {
		s := m.Content
		content = &s
	}
	return wireMessage{
		Role:       m.Role,
		Content:    content,
		ToolCalls:  m.ToolCalls,
		ToolCallID: m.ToolCallID,
		Name:       m.Name,
	}
}

// ToolSchema describes one tool in the request.
type ToolSchema struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction is the function part of a tool schema.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []wireMessage `json:"messages"`
	Tools       []ToolSchema  `json:"tools,omitempty"`
	ToolChoice  string        `json:"tool_choice,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

// usage holds token counts, incl. reasoning tokens.
type usage struct {
	PromptTokens            int `json:"prompt_tokens"`
	CompletionTokens        int `json:"completion_tokens"`
	TotalTokens             int `json:"total_tokens"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

type chatResponseBody struct {
	Choices []struct {
		Message struct {
			Role             string     `json:"role"`
			Content          *string    `json:"content"`
			Reasoning        string     `json:"reasoning"`
			ReasoningContent string     `json:"reasoning_content"`
			ToolCalls        []ToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage usage `json:"usage"`
}

// ChatResponse is one successful model turn.
type ChatResponse struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string
	Usage        usage
	Elapsed      time.Duration
}

// ClientStats accumulates usage across all calls of the process.
type ClientStats struct {
	Calls            int
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	Cost             float64
	ByModel          map[string]int
}

// ChatClient is a minimal OpenAI-compatible client with tool calling.
type ChatClient struct {
	baseURL   string
	apiKey    string
	http      *http.Client
	maxTokens int
	stats     ClientStats
}

// NewChatClient builds a client from LLM_BASE_URL and OPENAI_API_KEY.
func NewChatClient() *ChatClient {
	return &ChatClient{
		baseURL:   os.Getenv("LLM_BASE_URL"),
		apiKey:    os.Getenv("OPENAI_API_KEY"),
		http:      &http.Client{Timeout: 180 * time.Second},
		maxTokens: 4096,
		stats:     ClientStats{ByModel: map[string]int{}},
	}
}

// SetMaxTokens bounds completion length. Reasoning models spend this budget on
// hidden reasoning first, so an unbounded request can hang for minutes.
func (c *ChatClient) SetMaxTokens(n int) {
	if n > 0 {
		c.maxTokens = n
	}
}

// Stats returns a snapshot of accumulated usage.
func (c *ChatClient) Stats() ClientStats {
	s := c.stats
	s.ByModel = make(map[string]int, len(c.stats.ByModel))
	for k, v := range c.stats.ByModel {
		s.ByModel[k] = v
	}
	return s
}

// send performs one HTTP round-trip and returns status code and body.
func (c *ChatClient) send(ctx context.Context, payload []byte) (int, []byte, error) {
	url := c.baseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, err
	}
	return resp.StatusCode, data, nil
}

// Chat sends one request, retrying transient failures (429/5xx) up to 3 times.
func (c *ChatClient) Chat(ctx context.Context, model string, messages []Message, tools []ToolSchema, temperature float64) (*ChatResponse, error) {
	wire := make([]wireMessage, len(messages))
	for i, m := range messages {
		wire[i] = toWire(m)
	}
	body := chatRequest{
		Model:       model,
		Messages:    wire,
		Tools:       tools,
		Temperature: &temperature,
		MaxTokens:   c.maxTokens,
	}
	if len(tools) > 0 {
		body.ToolChoice = "auto"
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * time.Second):
			}
		}

		status, data, err := c.send(ctx, payload)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			lastErr = err
			continue
		}
		if status == http.StatusTooManyRequests || status >= 500 {
			lastErr = fmt.Errorf("%s: status %d: %s", model, status, truncate(string(data), 500))
			continue
		}
		if status != http.StatusOK {
			return nil, fmt.Errorf("%s: status %d: %s", model, status, truncate(string(data), 2000))
		}

		var parsed chatResponseBody
		if err := json.Unmarshal(data, &parsed); err != nil {
			return nil, fmt.Errorf("%s: decode response: %w", model, err)
		}
		if len(parsed.Choices) == 0 {
			return nil, fmt.Errorf("%s: empty choices", model)
		}

		choice := parsed.Choices[0]
		content := ""
		if choice.Message.Content != nil {
			content = *choice.Message.Content
		}

		c.stats.Calls++
		c.stats.PromptTokens += parsed.Usage.PromptTokens
		c.stats.CompletionTokens += parsed.Usage.CompletionTokens
		c.stats.TotalTokens += parsed.Usage.TotalTokens
		c.stats.ByModel[model]++
		if m, ok := FindModel(model); ok {
			c.stats.Cost += float64(parsed.Usage.TotalTokens) * m.RatePerM / 1e6
		}

		if content == "" && len(choice.Message.ToolCalls) == 0 {
			reasoning := choice.Message.Reasoning
			if reasoning == "" {
				reasoning = choice.Message.ReasoningContent
			}
			if choice.FinishReason == "length" && reasoning != "" {
				return nil, fmt.Errorf("%s: весь лимит (%d ток.) ушёл на рассуждения, ответа нет; увеличьте -max-tokens", model, c.maxTokens)
			}
			return nil, fmt.Errorf("%s: пустой ответ модели (finish_reason=%q)", model, choice.FinishReason)
		}

		return &ChatResponse{
			Content:      content,
			ToolCalls:    choice.Message.ToolCalls,
			FinishReason: choice.FinishReason,
			Usage:        parsed.Usage,
			Elapsed:      time.Since(start),
		}, nil
	}
	return nil, fmt.Errorf("%s: запрос не удался после 3 попыток: %w", model, lastErr)
}
