package main

import "testing"

func TestDefaultModelFromEnv(t *testing.T) {
	t.Setenv("LLM_MODEL", "glm-5.3-flash")
	t.Setenv("LLM_SUMMARY_MODEL", "")
	if got := DefaultModel(); got != "glm-5.3-flash" {
		t.Fatalf("DefaultModel() = %q, want LLM_MODEL", got)
	}
	if got := DefaultSummaryModel(); got != "glm-5.3-flash" {
		t.Fatalf("DefaultSummaryModel() = %q, want chat model fallback", got)
	}

	t.Setenv("LLM_SUMMARY_MODEL", "qwen3.8-27b")
	if got := DefaultSummaryModel(); got != "qwen3.8-27b" {
		t.Fatalf("DefaultSummaryModel() = %q, want LLM_SUMMARY_MODEL", got)
	}

	t.Setenv("LLM_MODEL", "")
	t.Setenv("LLM_SUMMARY_MODEL", "")
	if got := DefaultModel(); got != ModelSet[0].ID {
		t.Fatalf("DefaultModel() = %q, want roster fallback %q", got, ModelSet[0].ID)
	}
}
