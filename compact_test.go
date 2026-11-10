package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
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
