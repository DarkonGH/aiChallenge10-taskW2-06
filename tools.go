package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Tool is one callable function exposed to the model.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any
	Run         func(args json.RawMessage) (string, error)
}

// ToolRegistry maps names to tools and builds their JSON schemas.
type ToolRegistry struct {
	tools     map[string]*Tool
	order     []string
	maxResult int
}

// NewToolRegistry builds the agent's tool set. tasks is the in-memory list from
// tasks.txt, exposed through find_task.
func NewToolRegistry(tasks []Task) *ToolRegistry {
	r := &ToolRegistry{tools: map[string]*Tool{}, maxResult: 8000}

	r.add(&Tool{
		Name:        "now",
		Description: "Возвращает текущую дату и время (локальное и UTC). Аргументов не требует.",
		Parameters: map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"additionalProperties": false,
		},
		Run: runNow,
	})

	r.add(&Tool{
		Name:        "calc",
		Description: "Вычисляет арифметическое выражение с + - * / и скобками. Используй для любых вычислений, не считай в уме.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"expr": map[string]any{
					"type":        "string",
					"description": "Выражение, например (17*23)+5",
				},
			},
			"required":             []string{"expr"},
			"additionalProperties": false,
		},
		Run: runCalc,
	})

	r.add(&Tool{
		Name:        "find_task",
		Description: "Ищет задачу в списке задач по названию или условию. Возвращает номер, название и ответ.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "Ключевые слова, например улитка или поезд",
				},
			},
			"required":             []string{"query"},
			"additionalProperties": false,
		},
		Run: runFindTask(tasks),
	})

	return r
}

func (r *ToolRegistry) add(t *Tool) {
	r.tools[t.Name] = t
	r.order = append(r.order, t.Name)
}

// Schemas returns the tool schemas for the API request.
func (r *ToolRegistry) Schemas() []ToolSchema {
	out := make([]ToolSchema, 0, len(r.order))
	for _, name := range r.order {
		t := r.tools[name]
		out = append(out, ToolSchema{
			Type: "function",
			Function: ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}
	return out
}

// Names returns tool names in registration order.
func (r *ToolRegistry) Names() []string {
	return append([]string(nil), r.order...)
}

// Describe returns "name — description" lines for /tools.
func (r *ToolRegistry) Describe() []string {
	out := make([]string, 0, len(r.order))
	for _, name := range r.order {
		t := r.tools[name]
		out = append(out, fmt.Sprintf("%s — %s", t.Name, t.Description))
	}
	return out
}

// Execute runs a tool call and always returns a string for the model: errors
// and panics become text so the model can correct itself instead of crashing
// the loop.
func (r *ToolRegistry) Execute(call ToolCall) (out string) {
	name := call.Function.Name
	t, ok := r.tools[name]
	if !ok {
		return fmt.Sprintf("ошибка: неизвестный инструмент %q; доступны: %s", name, strings.Join(r.order, ", "))
	}

	args := json.RawMessage(call.Function.Arguments)
	if len(strings.TrimSpace(string(args))) == 0 {
		args = json.RawMessage("{}")
	}

	defer func() {
		if rec := recover(); rec != nil {
			out = fmt.Sprintf("ошибка инструмента %s: %v", name, rec)
		}
	}()

	res, err := t.Run(args)
	if err != nil {
		return fmt.Sprintf("ошибка инструмента %s: %v", name, err)
	}
	return truncate(res, r.maxResult)
}

func runNow(json.RawMessage) (string, error) {
	n := time.Now()
	return fmt.Sprintf("локальное: %s\nUTC: %s",
		n.Format("2006-01-02 15:04:05 -07:00"),
		n.UTC().Format(time.RFC3339)), nil
}

func runCalc(raw json.RawMessage) (string, error) {
	var a struct {
		Expr string `json:"expr"`
	}
	if err := json.Unmarshal(raw, &a); err != nil {
		return "", fmt.Errorf("аргумент expr: %v", err)
	}
	if strings.TrimSpace(a.Expr) == "" {
		return "", fmt.Errorf("пустое выражение")
	}
	v, err := evalExpr(a.Expr)
	if err != nil {
		return "", err
	}
	return strconv.FormatFloat(v, 'f', -1, 64), nil
}

func runFindTask(tasks []Task) func(json.RawMessage) (string, error) {
	return func(raw json.RawMessage) (string, error) {
		var a struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return "", fmt.Errorf("аргумент query: %v", err)
		}
		q := strings.ToLower(strings.TrimSpace(a.Query))
		if q == "" {
			return fmt.Sprintf("в списке %d задач; уточни запрос", len(tasks)), nil
		}

		hits := matchTasks(tasks, q)
		if len(hits) == 0 {
			// Try individual words as a looser fallback.
			for _, w := range strings.Fields(q) {
				if len([]rune(w)) < 3 {
					continue
				}
				if found := matchTasks(tasks, w); len(found) > 0 {
					hits = append(hits, found...)
				}
			}
		}
		if len(hits) == 0 {
			return "ничего не найдено", nil
		}

		var b strings.Builder
		seen := map[int]bool{}
		shown := 0
		for _, t := range hits {
			if seen[t.ID] {
				continue
			}
			seen[t.ID] = true
			if shown == 5 {
				b.WriteString("…есть ещё совпадения, уточни запрос\n")
				break
			}
			fmt.Fprintf(&b, "%d. %s — ответ %s\n", t.ID, t.Label, formatAnswer(t.Answer))
			shown++
		}
		return strings.TrimRight(b.String(), "\n"), nil
	}
}

func matchTasks(tasks []Task, q string) []Task {
	var out []Task
	for _, t := range tasks {
		hay := strings.ToLower(t.Label + " " + t.Condition)
		if strings.Contains(hay, q) {
			out = append(out, t)
		}
	}
	return out
}

// calcParser is a tiny recursive-descent evaluator. It deliberately avoids
// eval-like constructs: tool input is untrusted model output.
type calcParser struct {
	s   string
	pos int
}

func evalExpr(s string) (float64, error) {
	p := &calcParser{s: strings.ReplaceAll(strings.ReplaceAll(s, " ", ""), ",", ".")}
	v, err := p.expr()
	if err != nil {
		return 0, err
	}
	if p.pos != len(p.s) {
		return 0, fmt.Errorf("неожиданный символ %q в позиции %d", string(p.s[p.pos]), p.pos)
	}
	return v, nil
}

func (p *calcParser) expr() (float64, error) {
	v, err := p.term()
	if err != nil {
		return 0, err
	}
	for p.pos < len(p.s) && (p.s[p.pos] == '+' || p.s[p.pos] == '-') {
		op := p.s[p.pos]
		p.pos++
		rhs, err := p.term()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			v += rhs
		} else {
			v -= rhs
		}
	}
	return v, nil
}

func (p *calcParser) term() (float64, error) {
	v, err := p.factor()
	if err != nil {
		return 0, err
	}
	for p.pos < len(p.s) && (p.s[p.pos] == '*' || p.s[p.pos] == '/') {
		op := p.s[p.pos]
		p.pos++
		rhs, err := p.factor()
		if err != nil {
			return 0, err
		}
		if op == '*' {
			v *= rhs
		} else {
			if rhs == 0 {
				return 0, fmt.Errorf("деление на ноль")
			}
			v /= rhs
		}
	}
	return v, nil
}

func (p *calcParser) factor() (float64, error) {
	if p.pos >= len(p.s) {
		return 0, fmt.Errorf("неожиданный конец выражения")
	}
	c := p.s[p.pos]
	switch {
	case c == '+':
		p.pos++
		return p.factor()
	case c == '-':
		p.pos++
		v, err := p.factor()
		return -v, err
	case c == '(':
		p.pos++
		v, err := p.expr()
		if err != nil {
			return 0, err
		}
		if p.pos >= len(p.s) || p.s[p.pos] != ')' {
			return 0, fmt.Errorf("нет закрывающей скобки")
		}
		p.pos++
		return v, nil
	case c == '.' || (c >= '0' && c <= '9'):
		start := p.pos
		for p.pos < len(p.s) && ((p.s[p.pos] >= '0' && p.s[p.pos] <= '9') || p.s[p.pos] == '.') {
			p.pos++
		}
		f, err := strconv.ParseFloat(p.s[start:p.pos], 64)
		if err != nil {
			return 0, fmt.Errorf("не число %q", p.s[start:p.pos])
		}
		return f, nil
	default:
		return 0, fmt.Errorf("неожиданный символ %q", string(c))
	}
}
