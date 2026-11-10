package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAgentToolLoop(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}

		if calls == 1 {
			if len(req.Tools) == 0 {
				t.Error("tools missing from the first request")
			}
			if req.ToolChoice != "auto" {
				t.Errorf("tool_choice = %q, want auto", req.ToolChoice)
			}
			if len(req.Messages) == 0 || req.Messages[0].Role != "system" {
				t.Errorf("first message is not system: %+v", req.Messages)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{
					"message": map[string]any{
						"role":    "assistant",
						"content": nil,
						"tool_calls": []any{map[string]any{
							"id":   "call_1",
							"type": "function",
							"function": map[string]any{
								"name":      "calc",
								"arguments": `{"expr":"17*23"}`,
							},
						}},
					},
					"finish_reason": "tool_calls",
				}},
				"usage": map[string]any{"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
			})
			return
		}

		found := false
		for _, m := range req.Messages {
			if m.Role == "tool" && m.ToolCallID == "call_1" && m.Content != nil && *m.Content == "391" {
				found = true
			}
		}
		if !found {
			t.Errorf("tool result was not sent back to the model: %+v", req.Messages)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"role": "assistant", "content": "17*23 = 391"},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
		})
	}))
	defer srv.Close()

	client := &ChatClient{baseURL: srv.URL, apiKey: "test", http: srv.Client(), stats: ClientStats{ByModel: map[string]int{}}}
	h, err := OpenHistory(t.TempDir(), "s", "m", "m", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	agent := NewAgent(client, NewToolRegistry(nil), h, DefaultBudget(), 4, 0, 800)
	got, err := agent.Chat(context.Background(), "сколько будет 17*23?")
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if got != "17*23 = 391" {
		t.Errorf("answer = %q", got)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2", calls)
	}

	// user, assistant(tool_calls), tool, assistant(final)
	if len(h.Active()) != 4 {
		t.Fatalf("history = %d messages, want 4", len(h.Active()))
	}
	if h.Active()[1].Role != "assistant" || len(h.Active()[1].ToolCalls) != 1 {
		t.Errorf("assistant tool call not stored: %+v", h.Active()[1])
	}
	if h.Active()[2].Role != "tool" || h.Active()[2].Content != "391" {
		t.Errorf("tool result not stored: %+v", h.Active()[2])
	}
}

func TestAgentNoTools(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Tools) == 0 {
			t.Error("tools should always be advertised")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"role": "assistant", "content": "Рекурсия — это вызов функцией самой себя."},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
		})
	}))
	defer srv.Close()

	client := &ChatClient{baseURL: srv.URL, apiKey: "test", http: srv.Client(), stats: ClientStats{ByModel: map[string]int{}}}
	h, err := OpenHistory(t.TempDir(), "s", "m", "m", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	agent := NewAgent(client, NewToolRegistry(nil), h, DefaultBudget(), 4, 0, 800)
	got, err := agent.Chat(context.Background(), "что такое рекурсия?")
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if !strings.Contains(got, "Рекурсия") {
		t.Errorf("answer = %q", got)
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}
}

func TestAgentToolErrorIsReturnedToModel(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if calls == 1 {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{
					"message": map[string]any{
						"role": "assistant", "content": nil,
						"tool_calls": []any{map[string]any{
							"id": "call_1", "type": "function",
							"function": map[string]any{"name": "calc", "arguments": `{"expr":"1/0"}`},
						}},
					},
				}},
			})
			return
		}
		found := false
		for _, m := range req.Messages {
			if m.Role == "tool" && m.Content != nil && strings.Contains(*m.Content, "деление на ноль") {
				found = true
			}
		}
		if !found {
			t.Errorf("tool error not sent back: %+v", req.Messages)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{"role": "assistant", "content": "На ноль делить нельзя."},
			}},
		})
	}))
	defer srv.Close()

	client := &ChatClient{baseURL: srv.URL, apiKey: "test", http: srv.Client(), stats: ClientStats{ByModel: map[string]int{}}}
	h, err := OpenHistory(t.TempDir(), "s", "m", "m", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	agent := NewAgent(client, NewToolRegistry(nil), h, DefaultBudget(), 4, 0, 800)
	if _, err := agent.Chat(context.Background(), "сколько будет 1/0?"); err != nil {
		t.Fatalf("chat: %v", err)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2", calls)
	}
}
