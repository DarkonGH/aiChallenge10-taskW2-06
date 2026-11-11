package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
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
	Model         string         `json:"model"`
	Messages      []wireMessage  `json:"messages"`
	Tools         []ToolSchema   `json:"tools,omitempty"`
	ToolChoice    string         `json:"tool_choice,omitempty"`
	Temperature   *float64       `json:"temperature,omitempty"`
	MaxTokens     int            `json:"max_tokens,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

// streamOptions asks the server to append a final chunk with token usage.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
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

// streamToolCall is one fragment of a tool call inside a stream delta. The
// server sends id and name on the first fragment, arguments arrive in pieces.
type streamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// chatChunk is one SSE data line of a streamed completion.
type chatChunk struct {
	Choices []struct {
		Delta struct {
			Role             string           `json:"role"`
			Content          string           `json:"content"`
			Reasoning        string           `json:"reasoning"`
			ReasoningContent string           `json:"reasoning_content"`
			ToolCalls        []streamToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usage `json:"usage"`
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

// record accumulates usage of one finished API call.
func (c *ChatClient) record(model string, u usage) {
	c.stats.Calls++
	c.stats.PromptTokens += u.PromptTokens
	c.stats.CompletionTokens += u.CompletionTokens
	c.stats.TotalTokens += u.TotalTokens
	c.stats.ByModel[model]++
	if m, ok := FindModel(model); ok {
		c.stats.Cost += float64(u.TotalTokens) * m.RatePerM / 1e6
	}
}

// emptyAnswerErr builds the error for a turn without content and tool calls:
// reasoning models can burn the whole token budget on hidden reasoning.
func (c *ChatClient) emptyAnswerErr(model, finish, reasoning string) error {
	if finish == "length" && reasoning != "" {
		return fmt.Errorf("%s: весь лимит (%d ток.) ушёл на рассуждения, ответа нет; увеличьте -max-tokens", model, c.maxTokens)
	}
	return fmt.Errorf("%s: пустой ответ модели (finish_reason=%q)", model, finish)
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

		c.record(model, parsed.Usage)

		if content == "" && len(choice.Message.ToolCalls) == 0 {
			reasoning := choice.Message.Reasoning
			if reasoning == "" {
				reasoning = choice.Message.ReasoningContent
			}
			return nil, c.emptyAnswerErr(model, choice.FinishReason, reasoning)
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

// streamFail says what to do after a failed streaming attempt.
type streamFail int

const (
	failRetry    streamFail = iota // transient — retry the streaming request
	failFallback                   // endpoint likely cannot stream — repeat without stream
	failStop                       // give up: error is final or text is already on screen
)

// ChatStream is Chat with server-sent events: content deltas are forwarded to
// onDelta as they arrive (a nil callback is allowed), so the console can print
// the answer while the model is still generating. Reasoning deltas are
// swallowed — only the final content is streamed. A plain (non-stream) request
// is used as fallback when the endpoint refuses or ignores streaming.
func (c *ChatClient) ChatStream(ctx context.Context, model string, messages []Message, tools []ToolSchema, temperature float64, onDelta func(string)) (*ChatResponse, error) {
	wire := make([]wireMessage, len(messages))
	for i, m := range messages {
		wire[i] = toWire(m)
	}
	body := chatRequest{
		Model:         model,
		Messages:      wire,
		Tools:         tools,
		Temperature:   &temperature,
		MaxTokens:     c.maxTokens,
		Stream:        true,
		StreamOptions: &streamOptions{IncludeUsage: true},
	}
	if len(tools) > 0 {
		body.ToolChoice = "auto"
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	fallback := false
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		if attempt > 1 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt-1) * time.Second):
			}
		}
		resp, emitted, fail, err := c.streamOnce(ctx, model, payload, onDelta)
		if err == nil {
			resp.Elapsed = time.Since(start)
			return resp, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if emitted || fail == failStop {
			// Part of the text is already on screen — a retry would print it
			// twice, so surface the failure as is.
			return nil, err
		}
		if fail == failFallback {
			fallback = true
			break
		}
		lastErr = err
	}
	if fallback {
		return c.Chat(ctx, model, messages, tools, temperature)
	}
	return nil, fmt.Errorf("%s: запрос не удался после 3 попыток: %w", model, lastErr)
}

// sendStream performs one HTTP round-trip and returns the open response body
// for incremental reading.
func (c *ChatClient) sendStream(ctx context.Context, payload []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Accept", "text/event-stream")
	return c.http.Do(req)
}

// streamOnce runs one streaming request. It returns the accumulated response,
// whether anything was already handed to onDelta, and how the caller should
// treat a failure.
func (c *ChatClient) streamOnce(ctx context.Context, model string, payload []byte, onDelta func(string)) (*ChatResponse, bool, streamFail, error) {
	httpResp, err := c.sendStream(ctx, payload)
	if err != nil {
		return nil, false, failRetry, err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode >= 500 {
		data, _ := io.ReadAll(io.LimitReader(httpResp.Body, 4096))
		return nil, false, failRetry, fmt.Errorf("status %d: %s", httpResp.StatusCode, truncate(string(data), 500))
	}
	if httpResp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(httpResp.Body, 8192))
		return nil, false, failFallback, fmt.Errorf("status %d: %s", httpResp.StatusCode, truncate(string(data), 2000))
	}

	sc := bufio.NewScanner(httpResp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		content   strings.Builder
		reasoning strings.Builder
		calls     = map[int]*streamToolCall{}
		callOrder []int
		finish    string
		use       usage
		hadUsage  bool
		sawData   bool
		emitted   bool
	)
	for sc.Scan() {
		if ctx.Err() != nil {
			return nil, emitted, failStop, ctx.Err()
		}
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var chunk chatChunk
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue // tolerate keep-alive and junk lines
		}
		sawData = true
		if chunk.Usage != nil {
			use, hadUsage = *chunk.Usage, true
		}
		for _, ch := range chunk.Choices {
			d := ch.Delta
			reasoning.WriteString(d.Reasoning)
			reasoning.WriteString(d.ReasoningContent)
			if d.Content != "" {
				content.WriteString(d.Content)
				emitted = true
				if onDelta != nil {
					onDelta(d.Content)
				}
			}
			for _, tc := range d.ToolCalls {
				acc, ok := calls[tc.Index]
				if !ok {
					cp := tc // copy: map values must not alias the loop variable
					calls[tc.Index] = &cp
					callOrder = append(callOrder, tc.Index)
				} else {
					if acc.ID == "" {
						acc.ID = tc.ID
					}
					if acc.Function.Name == "" {
						acc.Function.Name = tc.Function.Name
					}
					acc.Function.Arguments += tc.Function.Arguments
				}
			}
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
		}
	}
	if err := sc.Err(); err != nil {
		fail := failStop
		if !emitted {
			fail = failRetry
		}
		return nil, emitted, fail, fmt.Errorf("чтение потока: %w", err)
	}
	if !sawData {
		// The server answered 200 but not with SSE — it either ignores the
		// stream field or the proxy swallowed it. Repeat without streaming.
		return nil, emitted, failFallback, fmt.Errorf("сервер вернул ответ не в формате SSE")
	}

	var toolCalls []ToolCall
	for _, idx := range callOrder {
		acc := calls[idx]
		typ := acc.Type
		if typ == "" {
			typ = "function"
		}
		toolCalls = append(toolCalls, ToolCall{
			ID:   acc.ID,
			Type: typ,
		})
		toolCalls[len(toolCalls)-1].Function.Name = acc.Function.Name
		toolCalls[len(toolCalls)-1].Function.Arguments = acc.Function.Arguments
	}
	if !hadUsage {
		use = usage{} // no usage chunk: keep zero counts so stats stay honest
	}
	c.record(model, use)

	cnt := content.String()
	if cnt == "" && len(toolCalls) == 0 {
		return nil, emitted, failStop, c.emptyAnswerErr(model, finish, reasoning.String())
	}
	return &ChatResponse{
		Content:      cnt,
		ToolCalls:    toolCalls,
		FinishReason: finish,
		Usage:        use,
	}, emitted, failStop, nil
}
