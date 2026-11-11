package main

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// summaryAttemptTimeout bounds one summarizer request. Without it a hung
// summary model eats the caller's whole deadline (180 s in ask), and the
// fallback model never gets a chance to run.
var summaryAttemptTimeout = 75 * time.Second

// Budget is the token budget of the prompt window.
type Budget struct {
	MaxPromptTokens int // hard ceiling for system + summary + history
	ReserveOutput   int // room kept for the model's answer
	MinTailTokens   int // keep at least this much recent history verbatim
}

// DefaultBudget is tuned for small demo backends (8k–32k context).
func DefaultBudget() Budget {
	return Budget{
		MaxPromptTokens: 6000,
		ReserveOutput:   1024,
		MinTailTokens:   1200,
	}
}

// Compactor implements the context strategy: sliding window + rolling summary
// + archive. It never deletes data: messages leave the prompt, not the log.
type Compactor struct {
	hist         *History
	client       *ChatClient
	budget       Budget
	calib        *Calibrator
	maxToolChars int
	trace        func(string)
}

// NewCompactor builds a compactor for one session.
func NewCompactor(hist *History, client *ChatClient, budget Budget, calib *Calibrator, maxToolChars int) *Compactor {
	return &Compactor{
		hist:         hist,
		client:       client,
		budget:       budget,
		calib:        calib,
		maxToolChars: maxToolChars,
	}
}

// PromptTokens estimates the next request size: system + summary + active
// window, with tool outputs truncated exactly as buildMessages does.
func (c *Compactor) PromptTokens(system string) int {
	n := EstimateTokens(system)
	if s := c.hist.SummaryText(); s != "" {
		n += EstimateTokens("Конспект предыдущего диалога:\n"+s) + 4
	}
	for _, m := range c.hist.Active() {
		msg := *m
		if msg.Role == "tool" {
			msg.Content = truncate(msg.Content, c.maxToolChars)
		}
		n += EstimateMessage(msg)
	}
	return c.calib.Scale(n)
}

// MaybeCompact compacts until the prompt fits the budget (up to 3 rounds).
func (c *Compactor) MaybeCompact(ctx context.Context, system string) (bool, error) {
	limit := c.budget.MaxPromptTokens - c.budget.ReserveOutput
	if limit < 500 {
		limit = 500
	}
	changed := false
	for i := 0; i < 3; i++ {
		if c.PromptTokens(system) <= limit*85/100 {
			break
		}
		ok, err := c.ForceCompact(ctx, system)
		if err != nil {
			return changed, err
		}
		if !ok {
			break
		}
		changed = true
	}
	return changed, nil
}

// ForceCompact archives the oldest turns and folds them into the rolling
// summary. It returns false when there is nothing safe to compact.
func (c *Compactor) ForceCompact(ctx context.Context, system string) (bool, error) {
	msgs := c.hist.Active()
	split := c.splitIndex(msgs)
	if split <= 0 {
		return false, nil
	}
	prefix := msgs[:split]
	from, to := prefix[0].Seq, prefix[len(prefix)-1].Seq
	// The new summary folds in the previous one, so its coverage starts
	// earlier. Compact() only touches active messages, so widening the range
	// is safe: already archived seqs are simply not in the window.
	coversFrom := from
	if c.hist.HasSummary() {
		if prev := c.hist.SummaryCovers(); prev[0] > 0 && prev[0] < coversFrom {
			coversFrom = prev[0]
		}
	}

	summary, err := c.summarize(ctx, prefix)
	if err != nil {
		// Never lose the dialogue because the summarizer is down: keep the old
		// summary, archive the raw messages and leave a pointer instead.
		var b strings.Builder
		if c.hist.HasSummary() {
			b.WriteString(c.hist.SummaryText())
			b.WriteString("\n\n")
		}
		fmt.Fprintf(&b, "(сжатие недоступно: %v; ещё %d сообщений перенесено в архив, доступны через /recall)",
			err, len(prefix))
		summary = b.String()
	}
	if err := c.hist.Compact(coversFrom, to, summary); err != nil {
		return false, err
	}
	if c.trace != nil {
		c.trace(fmt.Sprintf("сжатие контекста: сообщения seq %d..%d → архив, конспект ~%d ток., промпт теперь ~%d ток.",
			from, to, EstimateTokens(summary), c.PromptTokens(system)))
	}
	return true, nil
}

// splitIndex returns the index of the user message that starts the verbatim
// tail. Splitting on a user boundary keeps every assistant+tool group intact.
func (c *Compactor) splitIndex(msgs []*Message) int {
	if len(msgs) < 2 {
		return 0
	}
	tail := 0
	best := 0
	for i := len(msgs) - 1; i >= 0; i-- {
		tail += EstimateMessage(*msgs[i])
		if msgs[i].Role == "user" && tail >= c.budget.MinTailTokens {
			best = i
			break
		}
	}
	if best == 0 {
		// Fewer turns than the tail budget: keep the last user turn.
		for i := len(msgs) - 1; i > 0; i-- {
			if msgs[i].Role == "user" {
				best = i
				break
			}
		}
	}
	return best
}

// summarizePrompt builds the summarizer input: the previous summary plus the
// messages that are leaving the prompt.
func (c *Compactor) summarizePrompt(prefix []*Message) string {
	var b strings.Builder
	if c.hist.HasSummary() {
		b.WriteString("Предыдущий конспект:\n")
		b.WriteString(c.hist.SummaryText())
		b.WriteString("\n\n")
	}
	b.WriteString("Диалог:\n")
	for _, m := range prefix {
		switch m.Role {
		case "user":
			fmt.Fprintf(&b, "Пользователь: %s\n", truncate(m.Content, 1000))
		case "assistant":
			if m.Content != "" {
				fmt.Fprintf(&b, "Ассистент: %s\n", truncate(m.Content, 1000))
			}
			for _, tc := range m.ToolCalls {
				fmt.Fprintf(&b, "  вызов %s(%s)\n", tc.Function.Name, truncate(tc.Function.Arguments, 300))
			}
		case "tool":
			fmt.Fprintf(&b, "  результат %s: %s\n", m.Name, truncate(m.Content, 400))
		}
	}
	return b.String()
}

// summarize folds the previous summary and the archived prefix into one short
// note. If the summary model fails (503, timeout, empty answer), the chat model
// is tried once as a backup: a slightly more expensive note is much better than
// a stub that loses the dialogue.
func (c *Compactor) summarize(ctx context.Context, prefix []*Message) (string, error) {
	prompt := c.summarizePrompt(prefix)
	primary := c.hist.SummaryModel()

	summary, err := c.summarizeWith(ctx, primary, prompt)
	if err == nil {
		return summary, nil
	}

	backup := c.hist.ChatModel()
	if backup == "" || backup == primary || ctx.Err() != nil {
		return "", err
	}
	if c.trace != nil {
		c.trace(fmt.Sprintf("сжатие: %s не справился (%s) — пробую %s", primary, oneLine(err.Error()), backup))
	}
	summary, err2 := c.summarizeWith(ctx, backup, prompt)
	if err2 != nil {
		return "", fmt.Errorf("%s: %v; запасная %s: %v", primary, err, backup, err2)
	}
	if c.trace != nil {
		c.trace(fmt.Sprintf("сжатие выполнено запасной моделью %s", backup))
	}
	return summary, nil
}

// summarizeWith runs one summarizer request on one model under its own timeout.
func (c *Compactor) summarizeWith(ctx context.Context, model, prompt string) (string, error) {
	sys := "Ты сжимаешь историю диалога в компактный конспект. Сохрани: что просил пользователь, " +
		"какие инструменты вызывались и с каким результатом, какие выводы и решения приняты. " +
		"Пиши по-русски, не более 250 слов, без вступлений и без обращения к пользователю."

	actx, cancel := context.WithTimeout(ctx, summaryAttemptTimeout)
	defer cancel()
	resp, err := c.client.Chat(actx, model, []Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: prompt},
	}, nil, 0)
	if err != nil {
		// Our own cap fired, not the caller's deadline: say so, otherwise the
		// error reads as if the whole turn ran out of time.
		if actx.Err() != nil && ctx.Err() == nil {
			return "", fmt.Errorf("%s: таймаут %s", model, summaryAttemptTimeout)
		}
		return "", err
	}
	summary := strings.TrimSpace(resp.Content)
	if summary == "" {
		return "", fmt.Errorf("%s: пустой конспект", model)
	}
	return summary, nil
}
