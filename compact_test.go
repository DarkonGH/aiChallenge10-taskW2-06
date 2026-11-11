package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestForceCompactArchivesOldTurns(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{"role": "assistant", "content": "КОНСПЕКТ"},
			}},
			"usage": map[string]any{"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
		})
	}))
	defer srv.Close()

	client := &ChatClient{baseURL: srv.URL, apiKey: "test", http: srv.Client(), stats: ClientStats{ByModel: map[string]int{}}}
	h, err := OpenHistory(t.TempDir(), "s", "m", "m", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	for i := 0; i < 10; i++ {
		h.AppendUser(fmt.Sprintf("вопрос номер %d", i))
		h.AppendAssistant("", []ToolCall{call("calc", `{"expr":"1+1"}`)}, "m")
		h.AppendTool("call_1", "calc", "2")
		h.AppendAssistant(fmt.Sprintf("ответ номер %d", i), nil, "m")
	}

	comp := NewCompactor(h, client, Budget{
		MaxPromptTokens: 200,
		ReserveOutput:   0,
		MinTailTokens:   40,
	}, NewCalibrator(), 800)

	ok, err := comp.ForceCompact(context.Background(), "sys")
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	if !ok {
		t.Fatal("expected compaction to happen")
	}
	if len(h.Active()) == 0 || len(h.Active()) == 40 {
		t.Fatalf("active = %d, expected a trimmed window", len(h.Active()))
	}
	if len(h.Archived()) == 0 {
		t.Fatal("nothing archived")
	}
	if h.SummaryText() != "КОНСПЕКТ" {
		t.Errorf("summary = %q", h.SummaryText())
	}
	if h.Active()[0].Role != "user" {
		t.Errorf("window starts with %q, want a user turn", h.Active()[0].Role)
	}
	assertNoOrphanTools(t, h.Active())

	// A second compaction folds the previous summary in and keeps the window valid.
	if _, err := comp.ForceCompact(context.Background(), "sys"); err != nil {
		t.Fatalf("second compact: %v", err)
	}
	assertNoOrphanTools(t, h.Active())
	if covers := h.SummaryCovers(); covers[0] != 1 {
		t.Errorf("summary covers from %d, want 1 (previous coverage must be folded in)", covers[0])
	}
}

func TestSplitIndexNeverBreaksToolGroup(t *testing.T) {
	h, err := OpenHistory(t.TempDir(), "s", "m", "m", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	h.AppendUser("вопрос 1")
	h.AppendAssistant("", []ToolCall{call("calc", `{"expr":"1+1"}`)}, "m")
	h.AppendTool("call_1", "calc", "2")
	h.AppendAssistant("ответ 1", nil, "m")
	h.AppendUser("вопрос 2")
	h.AppendAssistant("", []ToolCall{call("calc", `{"expr":"2+2"}`)}, "m")
	h.AppendTool("call_1", "calc", "4")
	h.AppendAssistant("ответ 2", nil, "m")

	comp := NewCompactor(h, nil, Budget{MinTailTokens: 1}, NewCalibrator(), 800)
	split := comp.splitIndex(h.Active())
	if split <= 0 || split >= len(h.Active()) {
		t.Fatalf("split = %d", split)
	}
	if h.Active()[split].Role != "user" {
		t.Errorf("split points at %q, want user", h.Active()[split].Role)
	}
	assertNoOrphanTools(t, h.Active()[split:])
}

// assertNoOrphanTools verifies that every tool message is preceded by an
// assistant message that requested it.
func assertNoOrphanTools(t *testing.T, msgs []*Message) {
	t.Helper()
	open := map[string]bool{}
	for _, m := range msgs {
		switch m.Role {
		case "assistant":
			for _, tc := range m.ToolCalls {
				open[tc.ID] = true
			}
		case "tool":
			if !open[m.ToolCallID] {
				t.Errorf("tool %q at seq %d has no matching assistant call", m.ToolCallID, m.Seq)
			}
			delete(open, m.ToolCallID)
		}
	}
}

// seedTurns appends n user/assistant turns so there is something to compact.
func seedTurns(t *testing.T, h *History, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		h.AppendUser(fmt.Sprintf("вопрос номер %d", i))
		h.AppendAssistant(fmt.Sprintf("ответ номер %d", i), nil, "m")
	}
}

func TestCompactFallsBackToChatModel(t *testing.T) {
	var models []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		models = append(models, req.Model)
		if req.Model == "sum-model" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"model is not available"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{"role": "assistant", "content": "КОНСПЕКТ ОТ ЧАТ-МОДЕЛИ"},
			}},
			"usage": map[string]any{"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
		})
	}))
	defer srv.Close()

	h, err := OpenHistory(t.TempDir(), "s", "chat-model", "sum-model", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	seedTurns(t, h, 6)

	comp := NewCompactor(h, newTestClient(srv), Budget{MaxPromptTokens: 120, ReserveOutput: 0, MinTailTokens: 20}, NewCalibrator(), 800)
	var traces []string
	comp.trace = func(s string) { traces = append(traces, s) }

	ok, err := comp.ForceCompact(context.Background(), "sys")
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	if !ok {
		t.Fatal("expected compaction")
	}
	if h.SummaryText() != "КОНСПЕКТ ОТ ЧАТ-МОДЕЛИ" {
		t.Errorf("summary = %q, want the fallback note", h.SummaryText())
	}
	if len(h.Archived()) == 0 {
		t.Error("nothing archived")
	}
	if len(models) < 2 || models[0] != "sum-model" || models[1] != "chat-model" {
		t.Errorf("models called = %v, want sum-model then chat-model", models)
	}
	joined := strings.Join(traces, "\n")
	if !strings.Contains(joined, "пробую chat-model") || !strings.Contains(joined, "запасной моделью chat-model") {
		t.Errorf("traces = %v, want fallback notices", traces)
	}
}

func TestCompactFallsBackOnSummaryTimeout(t *testing.T) {
	old := summaryAttemptTimeout
	summaryAttemptTimeout = 120 * time.Millisecond
	defer func() { summaryAttemptTimeout = old }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model == "slow-model" {
			<-r.Context().Done() // hang until the per-attempt timeout fires
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{"role": "assistant", "content": "БЫСТРЫЙ КОНСПЕКТ"},
			}},
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
		})
	}))
	defer srv.Close()

	h, err := OpenHistory(t.TempDir(), "s", "fast-model", "slow-model", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	seedTurns(t, h, 6)

	comp := NewCompactor(h, newTestClient(srv), Budget{MaxPromptTokens: 120, ReserveOutput: 0, MinTailTokens: 20}, NewCalibrator(), 800)
	ok, err := comp.ForceCompact(context.Background(), "sys")
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	if !ok {
		t.Fatal("expected compaction")
	}
	if h.SummaryText() != "БЫСТРЫЙ КОНСПЕКТ" {
		t.Errorf("summary = %q, want the fast-model note", h.SummaryText())
	}
}

func TestCompactStubNamesBothModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"nope"}`))
	}))
	defer srv.Close()

	h, err := OpenHistory(t.TempDir(), "s", "chat-model", "sum-model", "sys")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	seedTurns(t, h, 6)

	comp := NewCompactor(h, newTestClient(srv), Budget{MaxPromptTokens: 120, ReserveOutput: 0, MinTailTokens: 20}, NewCalibrator(), 800)
	ok, err := comp.ForceCompact(context.Background(), "sys")
	if err != nil {
		t.Fatalf("compact should degrade, not fail: %v", err)
	}
	if !ok {
		t.Fatal("expected compaction")
	}
	s := h.SummaryText()
	if !strings.Contains(s, "сжатие недоступно") {
		t.Errorf("summary = %q, want a stub", s)
	}
	if !strings.Contains(s, "sum-model") || !strings.Contains(s, "chat-model") {
		t.Errorf("summary = %q, want both models named", s)
	}
	if len(h.Archived()) == 0 {
		t.Error("messages must be archived even when both summarizers fail")
	}
}
