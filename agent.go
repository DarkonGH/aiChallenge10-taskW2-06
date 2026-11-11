package main

import (
	"context"
	"fmt"
	"strings"
)

// Agent owns the whole loop: request → tool call → tool result → answer.
// main only feeds it user text and renders the result.
type Agent struct {
	client             *ChatClient
	tools              *ToolRegistry
	hist               *History
	comp               *Compactor
	calib              *Calibrator
	maxSteps           int
	temperature        float64
	maxToolPromptChars int
	trace              func(string)

	// Streaming callbacks (nil = plain non-stream requests): onMessageStart
	// before each model request, onContent per streamed delta of the answer,
	// onMessageDone after each assistant message is complete.
	onMessageStart func()
	onContent      func(string)
	onMessageDone  func()
}

// NewAgent wires a session, a tool set and a context compactor together.
func NewAgent(client *ChatClient, tools *ToolRegistry, hist *History, budget Budget, maxSteps int, temperature float64, maxToolChars int) *Agent {
	calib := NewCalibrator()
	a := &Agent{
		client:             client,
		tools:              tools,
		hist:               hist,
		calib:              calib,
		maxSteps:           maxSteps,
		temperature:        temperature,
		maxToolPromptChars: maxToolChars,
	}
	a.comp = NewCompactor(hist, client, budget, calib, maxToolChars)
	a.comp.trace = func(s string) { a.tracef("%s", s) }
	return a
}

// SetTrace installs a callback for tool calls and compaction notices.
func (a *Agent) SetTrace(fn func(string)) {
	a.trace = fn
	a.comp.trace = fn
}

// SetStreamUI installs the streaming callbacks. Until it is called the agent
// uses plain non-stream requests (tests, benchmarks).
func (a *Agent) SetStreamUI(onStart func(), onContent func(string), onDone func()) {
	a.onMessageStart = onStart
	a.onContent = onContent
	a.onMessageDone = onDone
}

func (a *Agent) tracef(format string, args ...any) {
	if a.trace != nil {
		a.trace(fmt.Sprintf(format, args...))
	}
}

// Chat handles one user turn. It may perform several model/tool round-trips
// and returns the final text once the model stops requesting tools.
func (a *Agent) Chat(ctx context.Context, userText string) (string, error) {
	a.hist.AppendUser(userText)

	for step := 1; step <= a.maxSteps; step++ {
		if _, err := a.comp.MaybeCompact(ctx, a.hist.SystemPrompt()); err != nil {
			a.tracef("сжатие контекста: %v", err)
		}

		messages := a.buildMessages()
		estimated := 0
		for _, m := range messages {
			estimated += EstimateMessage(m)
		}

		if a.onMessageStart != nil {
			a.onMessageStart()
		}
		var resp *ChatResponse
		var err error
		if a.onContent != nil {
			resp, err = a.client.ChatStream(ctx, a.hist.ChatModel(), messages, a.tools.Schemas(), a.temperature, a.onContent)
		} else {
			resp, err = a.client.Chat(ctx, a.hist.ChatModel(), messages, a.tools.Schemas(), a.temperature)
		}
		if err != nil {
			return "", err
		}
		a.calib.Update(estimated, resp.Usage.PromptTokens)

		a.hist.AppendAssistant(resp.Content, resp.ToolCalls, a.hist.ChatModel())
		if a.onMessageDone != nil {
			a.onMessageDone()
		}

		if len(resp.ToolCalls) == 0 {
			return resp.Content, nil
		}

		for _, call := range resp.ToolCalls {
			out := a.tools.Execute(call)
			a.tracef("%s(%s) → %s", call.Function.Name, call.Function.Arguments, oneLine(out))
			a.hist.AppendTool(call.ID, call.Function.Name, out)
		}
	}

	return "", fmt.Errorf("агент: превышен лимит шагов (%d), модель продолжает вызывать инструменты", a.maxSteps)
}

// buildMessages assembles the API request: system prompt (+ rolling summary),
// then the active window. Tool outputs are truncated here only — the log keeps
// the full result.
func (a *Agent) buildMessages() []Message {
	system := a.hist.SystemPrompt()
	if s := a.hist.SummaryText(); s != "" {
		system += "\n\nКонспект предыдущего диалога:\n" + s
	}

	out := make([]Message, 0, len(a.hist.Active())+1)
	out = append(out, Message{Role: "system", Content: system})
	for _, m := range a.hist.Active() {
		c := *m
		if c.Role == "tool" {
			c.Content = truncate(c.Content, a.maxToolPromptChars)
		}
		out = append(out, c)
	}
	return out
}

// oneLine flattens a tool result for the trace line.
func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	return truncate(s, 160)
}
