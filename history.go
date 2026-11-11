package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Event is one line of the append-only session log. The log is event-sourced:
// the active window, the rolling summary and the archive are all reconstructed
// by replaying events, so a crash can never corrupt earlier turns.
type Event struct {
	Type         string     `json:"type"`
	V            int        `json:"v,omitempty"`
	ID           string     `json:"id,omitempty"`
	ChatModel    string     `json:"chat_model,omitempty"`
	SummaryModel string     `json:"summary_model,omitempty"`
	System       string     `json:"system,omitempty"`
	Created      *time.Time `json:"created,omitempty"`

	Message *Message `json:"message,omitempty"`
	Covers  []int64  `json:"covers,omitempty"`
	Summary string   `json:"summary,omitempty"`
	Model   string   `json:"model,omitempty"`
	Role    string   `json:"role,omitempty"` // model_switch: пусто/"chat" — чат-модель, "summary" — модель сжатия
}

// History is one chat session backed by <dataDir>/<id>/session.jsonl.
type History struct {
	dir  string
	id   string
	path string
	file *os.File

	chatModel    string
	summaryModel string
	system       string

	seq      int64
	msgs     []*Message // active window, in prompt order (without system)
	archived []*Message // messages moved out of the prompt, kept on disk

	summaryText   string
	summaryCovers [2]int64
	hasSummary    bool

	warnf func(format string, args ...any)
}

// NewSessionID returns a sortable, unique session id.
func NewSessionID() string {
	var b [2]byte
	_, _ = rand.Read(b[:])
	return time.Now().Format("20060102-150405") + "-" + hex.EncodeToString(b[:])
}

// OpenHistory opens an existing session or creates a new one. Empty
// requested* values mean "use what the session already has"; for a brand-new
// session they fall back to the roster defaults.
func OpenHistory(dataDir, id, requestedChat, requestedSummary, system string) (*History, error) {
	if id == "" {
		id = NewSessionID()
	}
	dir := filepath.Join(dataDir, id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}

	h := &History{
		dir:  dir,
		id:   id,
		path: filepath.Join(dir, "session.jsonl"),
		warnf: func(format string, args ...any) {
			fmt.Fprintf(os.Stderr, "предупреждение: "+format+"\n", args...)
		},
	}

	_, statErr := os.Stat(h.path)
	exists := statErr == nil
	if exists {
		if err := h.load(); err != nil {
			return nil, err
		}
	}
	storedChat := h.chatModel

	if h.system == "" {
		h.system = system
	}
	if requestedChat != "" {
		h.chatModel = requestedChat
	}
	if h.chatModel == "" {
		h.chatModel = DefaultModel()
	}
	if requestedSummary != "" {
		h.summaryModel = requestedSummary
	}
	if h.summaryModel == "" {
		h.summaryModel = DefaultSummaryModel()
	}

	f, err := os.OpenFile(h.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	h.file = f

	if !exists {
		now := time.Now()
		return h, h.writeEvent(Event{
			Type:         "session",
			V:            1,
			ID:           id,
			ChatModel:    h.chatModel,
			SummaryModel: h.summaryModel,
			System:       h.system,
			Created:      &now,
		})
	}
	if requestedChat != "" && requestedChat != storedChat {
		h.writeEvent(Event{Type: "model_switch", Model: requestedChat})
	}
	return h, nil
}

func (h *History) load() error {
	f, err := os.Open(h.path)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		b := sc.Bytes()
		if len(strings.TrimSpace(string(b))) == 0 {
			continue
		}
		var ev Event
		if err := json.Unmarshal(b, &ev); err != nil {
			// A torn last line after a crash is expected; skip it loudly.
			h.warnf("session.jsonl:%d: пропущена битая строка: %v", line, err)
			continue
		}
		h.apply(&ev)
	}
	return sc.Err()
}

func (h *History) apply(ev *Event) {
	switch ev.Type {
	case "session":
		if ev.ChatModel != "" {
			h.chatModel = ev.ChatModel
		}
		if ev.SummaryModel != "" {
			h.summaryModel = ev.SummaryModel
		}
		if ev.System != "" {
			h.system = ev.System
		}
	case "message":
		if ev.Message != nil {
			h.msgs = append(h.msgs, ev.Message)
			if ev.Message.Seq > h.seq {
				h.seq = ev.Message.Seq
			}
		}
	case "compact":
		if len(ev.Covers) == 2 {
			h.applyCompact([2]int64{ev.Covers[0], ev.Covers[1]}, ev.Summary)
		}
	case "model_switch":
		if ev.Model != "" {
			if ev.Role == "summary" {
				h.summaryModel = ev.Model
			} else {
				h.chatModel = ev.Model
			}
		}
	}
}

func (h *History) applyCompact(covers [2]int64, summary string) {
	keep := make([]*Message, 0, len(h.msgs))
	for _, m := range h.msgs {
		if m.Seq >= covers[0] && m.Seq <= covers[1] {
			h.archived = append(h.archived, m)
			continue
		}
		keep = append(keep, m)
	}
	h.msgs = keep
	h.summaryText = summary
	h.summaryCovers = covers
	h.hasSummary = true
}

func (h *History) writeEvent(ev Event) error {
	if h.file == nil {
		return nil
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if _, err := h.file.Write(b); err != nil {
		h.warnf("session.jsonl: запись не удалась: %v", err)
		return err
	}
	return nil
}

func (h *History) appendMsg(m *Message) *Message {
	h.seq++
	m.Seq = h.seq
	m.TS = time.Now()
	m.Tokens = EstimateMessage(*m)
	h.msgs = append(h.msgs, m)
	_ = h.writeEvent(Event{Type: "message", Message: m})
	return m
}

// AppendUser records a user turn.
func (h *History) AppendUser(content string) *Message {
	return h.appendMsg(&Message{Role: "user", Content: content})
}

// AppendAssistant records an assistant turn verbatim, including tool calls.
func (h *History) AppendAssistant(content string, calls []ToolCall, model string) *Message {
	return h.appendMsg(&Message{Role: "assistant", Content: content, ToolCalls: calls, Model: model})
}

// AppendTool records one tool result.
func (h *History) AppendTool(callID, name, content string) *Message {
	return h.appendMsg(&Message{Role: "tool", ToolCallID: callID, Name: name, Content: content})
}

// Compact moves messages with seq in [from,to] out of the prompt and replaces
// the rolling summary. Raw messages stay in the log forever.
func (h *History) Compact(from, to int64, summary string) error {
	if from > to {
		return fmt.Errorf("history: неверный диапазон %d..%d", from, to)
	}
	if err := h.writeEvent(Event{Type: "compact", Covers: []int64{from, to}, Summary: summary}); err != nil {
		return err
	}
	h.applyCompact([2]int64{from, to}, summary)
	return nil
}

// SetChatModel switches the chat model and records the switch in the log.
func (h *History) SetChatModel(model string) {
	if model == "" || model == h.chatModel {
		return
	}
	h.chatModel = model
	_ = h.writeEvent(Event{Type: "model_switch", Model: model})
}

// SetSummaryModel switches the compaction model and records the switch in the
// log. Takes effect on the next compaction: the compactor reads
// SummaryModel() at call time.
func (h *History) SetSummaryModel(model string) {
	if model == "" || model == h.summaryModel {
		return
	}
	h.summaryModel = model
	_ = h.writeEvent(Event{Type: "model_switch", Model: model, Role: "summary"})
}

// Active returns the prompt window, excluding the system prompt.
func (h *History) Active() []*Message { return h.msgs }

// Archived returns messages moved out of the prompt.
func (h *History) Archived() []*Message { return h.archived }

// SummaryText returns the rolling summary (empty when nothing was compacted).
func (h *History) SummaryText() string { return h.summaryText }

// HasSummary reports whether a rolling summary exists.
func (h *History) HasSummary() bool { return h.hasSummary }

// SummaryCovers returns the seq range covered by the rolling summary.
func (h *History) SummaryCovers() [2]int64 { return h.summaryCovers }

// SystemPrompt returns the session system prompt.
func (h *History) SystemPrompt() string { return h.system }

// ChatModel returns the current chat model.
func (h *History) ChatModel() string { return h.chatModel }

// SummaryModel returns the model used for compaction.
func (h *History) SummaryModel() string { return h.summaryModel }

// ID returns the session id.
func (h *History) ID() string { return h.id }

// Path returns the session log path.
func (h *History) Path() string { return h.path }

// Stats returns active/archived message and token counts.
func (h *History) Stats() (activeMsgs, activeTokens, archivedMsgs, archivedTokens int) {
	for _, m := range h.msgs {
		activeMsgs++
		activeTokens += m.Tokens
	}
	for _, m := range h.archived {
		archivedMsgs++
		archivedTokens += m.Tokens
	}
	return
}

// SearchArchived returns archived messages containing q (case-insensitive).
func (h *History) SearchArchived(q string) []*Message {
	q = strings.ToLower(strings.TrimSpace(q))
	if q == "" {
		return nil
	}
	var out []*Message
	for _, m := range h.archived {
		if strings.Contains(strings.ToLower(m.Content), q) {
			out = append(out, m)
		}
	}
	return out
}

// Close flushes and closes the session log.
func (h *History) Close() error {
	if h.file == nil {
		return nil
	}
	err := h.file.Close()
	h.file = nil
	return err
}

// ListSessions returns session ids under dataDir, newest first.
func ListSessions(dataDir string) ([]string, error) {
	entries, err := os.ReadDir(dataDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	type session struct {
		id      string
		modTime time.Time
	}
	var sessions []session
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		info, err := os.Stat(filepath.Join(dataDir, e.Name(), "session.jsonl"))
		if err != nil {
			continue
		}
		sessions = append(sessions, session{id: e.Name(), modTime: info.ModTime()})
	}
	sort.Slice(sessions, func(i, j int) bool { return sessions[i].modTime.After(sessions[j].modTime) })
	ids := make([]string, len(sessions))
	for i, s := range sessions {
		ids[i] = s.id
	}
	return ids, nil
}

// LatestSession returns the most recently modified session id.
func LatestSession(dataDir string) (string, error) {
	ids, err := ListSessions(dataDir)
	if err != nil {
		return "", err
	}
	if len(ids) == 0 {
		return "", fmt.Errorf("сессий пока нет")
	}
	return ids[0], nil
}

// SessionInfo is a one-line summary of a stored session for /session.
type SessionInfo struct {
	ID           string
	Created      time.Time
	ChatModel    string
	SummaryModel string
	Msgs         int
}

// SummarizeSessions lists sessions newest first with a little metadata,
// read straight from the JSONL logs.
func SummarizeSessions(dataDir string) ([]SessionInfo, error) {
	ids, err := ListSessions(dataDir)
	if err != nil {
		return nil, err
	}
	out := make([]SessionInfo, 0, len(ids))
	for _, id := range ids {
		info := SessionInfo{ID: id}
		if b, err := os.ReadFile(filepath.Join(dataDir, id, "session.jsonl")); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				var ev Event
				if json.Unmarshal([]byte(line), &ev) != nil {
					continue
				}
				switch ev.Type {
				case "session":
					if ev.Created != nil {
						info.Created = *ev.Created
					}
					info.ChatModel = ev.ChatModel
					info.SummaryModel = ev.SummaryModel
				case "message":
					info.Msgs++
				case "model_switch":
					if ev.Role == "summary" {
						info.SummaryModel = ev.Model
					} else {
						info.ChatModel = ev.Model
					}
				}
			}
		}
		out = append(out, info)
	}
	return out, nil
}
