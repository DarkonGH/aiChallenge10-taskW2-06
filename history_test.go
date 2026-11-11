package main

import (
	"testing"
)

func TestHistoryPersistAndCompact(t *testing.T) {
	dir := t.TempDir()

	h, err := OpenHistory(dir, "s1", "model-a", "model-b", "системный промпт")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	h.AppendUser("привет")
	h.AppendAssistant("здравствуйте", nil, "model-a")
	h.AppendUser("2+2?")
	h.AppendAssistant("", []ToolCall{call("calc", `{"expr":"2+2"}`)}, "model-a")
	h.AppendTool("call_1", "calc", "4")
	h.AppendAssistant("4", nil, "model-a")
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	h2, err := OpenHistory(dir, "s1", "", "", "")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if got := len(h2.Active()); got != 6 {
		t.Fatalf("active = %d, want 6", got)
	}
	if h2.ChatModel() != "model-a" || h2.SummaryModel() != "model-b" {
		t.Errorf("models = %q/%q", h2.ChatModel(), h2.SummaryModel())
	}
	if h2.SystemPrompt() != "системный промпт" {
		t.Errorf("system = %q", h2.SystemPrompt())
	}
	if h2.Active()[3].Seq != 4 {
		t.Errorf("seq order broken: %d", h2.Active()[3].Seq)
	}

	if err := h2.Compact(1, 2, "конспект первой пары"); err != nil {
		t.Fatalf("compact: %v", err)
	}
	if got := len(h2.Active()); got != 4 {
		t.Errorf("active after compact = %d, want 4", got)
	}
	if got := len(h2.Archived()); got != 2 {
		t.Errorf("archived = %d, want 2", got)
	}
	if h2.SummaryText() != "конспект первой пары" {
		t.Errorf("summary = %q", h2.SummaryText())
	}
	h2.SetChatModel("model-c")
	h2.Close()

	h3, err := OpenHistory(dir, "s1", "", "", "")
	if err != nil {
		t.Fatalf("reopen 2: %v", err)
	}
	defer h3.Close()
	if got := len(h3.Active()); got != 4 {
		t.Errorf("replayed active = %d, want 4", got)
	}
	if got := len(h3.Archived()); got != 2 {
		t.Errorf("replayed archived = %d, want 2", got)
	}
	if h3.SummaryText() != "конспект первой пары" {
		t.Errorf("replayed summary = %q", h3.SummaryText())
	}
	if h3.ChatModel() != "model-c" {
		t.Errorf("replayed model = %q, want model-c", h3.ChatModel())
	}
	if hits := h3.SearchArchived("привет"); len(hits) != 1 {
		t.Errorf("search archived = %d, want 1", len(hits))
	}
}

func TestSummarizeSessions(t *testing.T) {
	dir := t.TempDir()
	h, err := OpenHistory(dir, "s1", "model-a", "model-b", "sys")
	if err != nil {
		t.Fatal(err)
	}
	h.AppendUser("привет")
	h.AppendAssistant("здравствуйте", nil, "model-a")
	h.SetChatModel("model-c")
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}

	infos, err := SummarizeSessions(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 1 {
		t.Fatalf("infos = %d, want 1", len(infos))
	}
	if infos[0].ID != "s1" || infos[0].Msgs != 2 || infos[0].ChatModel != "model-c" {
		t.Errorf("info = %+v", infos[0])
	}
}

func TestLatestSession(t *testing.T) {
	dir := t.TempDir()
	if _, err := LatestSession(dir); err == nil {
		t.Fatal("expected error for empty dir")
	}
	if _, err := OpenHistory(dir, "one", "", "", "s"); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenHistory(dir, "two", "", "", "s"); err != nil {
		t.Fatal(err)
	}
	ids, err := ListSessions(dir)
	if err != nil || len(ids) != 2 {
		t.Fatalf("list = %v, %v", ids, err)
	}
	latest, err := LatestSession(dir)
	if err != nil || latest == "" {
		t.Fatalf("latest = %q, %v", latest, err)
	}
}

func TestSummaryModelSwitchPersists(t *testing.T) {
	dir := t.TempDir()

	h, err := OpenHistory(dir, "s-sum", "model-a", "model-s1", "sys")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	h.AppendUser("вопрос")
	h.SetSummaryModel("model-s2")
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	h2, err := OpenHistory(dir, "s-sum", "", "", "")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer h2.Close()
	if h2.ChatModel() != "model-a" {
		t.Errorf("chat = %q, want model-a", h2.ChatModel())
	}
	if h2.SummaryModel() != "model-s2" {
		t.Errorf("summary = %q, want model-s2", h2.SummaryModel())
	}

	infos, err := SummarizeSessions(dir)
	if err != nil {
		t.Fatalf("summarize: %v", err)
	}
	if len(infos) != 1 {
		t.Fatalf("sessions = %d, want 1", len(infos))
	}
	if infos[0].ChatModel != "model-a" || infos[0].SummaryModel != "model-s2" {
		t.Errorf("info models = %q/%q, want model-a/model-s2", infos[0].ChatModel, infos[0].SummaryModel)
	}
}
