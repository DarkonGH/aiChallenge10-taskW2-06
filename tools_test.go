package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// call builds a tool call for tests.
func call(name, args string) ToolCall {
	var tc ToolCall
	tc.ID = "call_1"
	tc.Type = "function"
	tc.Function.Name = name
	tc.Function.Arguments = args
	return tc
}

func TestEvalExpr(t *testing.T) {
	ok := []struct {
		in   string
		want float64
	}{
		{"17*23", 391},
		{"(2+3)*4", 20},
		{"-3+5", 2},
		{"10/4", 2.5},
		{"2*(3+4)-1", 13},
		{"7,5*2", 15},
	}
	for _, c := range ok {
		got, err := evalExpr(c.in)
		if err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		if got != c.want {
			t.Errorf("%s = %v, want %v", c.in, got, c.want)
		}
	}

	for _, bad := range []string{"", "1/0", "2+", "(1+2", "abc", "2**3"} {
		if _, err := evalExpr(bad); err == nil {
			t.Errorf("%q: expected an error", bad)
		}
	}
}

func TestRegistryExecute(t *testing.T) {
	r := NewToolRegistry([]Task{
		{ID: 8, Label: "улитка на столбе", Condition: "Улитка ползёт вверх по столбу", Answer: 28},
	})

	if out := r.Execute(call("calc", `{"expr":"17*23"}`)); out != "391" {
		t.Errorf("calc = %q, want 391", out)
	}
	if out := r.Execute(call("calc", `{"expr":"1/0"}`)); !strings.Contains(out, "деление на ноль") {
		t.Errorf("calc error = %q", out)
	}
	if out := r.Execute(call("find_task", `{"query":"улитка"}`)); !strings.Contains(out, "улитка на столбе") || !strings.Contains(out, "28") {
		t.Errorf("find_task = %q", out)
	}
	if out := r.Execute(call("find_task", `{"query":"дракон"}`)); !strings.Contains(out, "ничего не найдено") {
		t.Errorf("find_task miss = %q", out)
	}
	if out := r.Execute(call("nope", `{}`)); !strings.Contains(out, "неизвестный инструмент") {
		t.Errorf("unknown tool = %q", out)
	}
	if out := r.Execute(call("calc", `not json`)); !strings.Contains(out, "ошибка") {
		t.Errorf("bad args = %q", out)
	}
	if out := r.Execute(call("now", `{}`)); !strings.Contains(out, "UTC") {
		t.Errorf("now = %q", out)
	}
}

func TestToolSchemas(t *testing.T) {
	r := NewToolRegistry(nil)
	schemas := r.Schemas()
	if len(schemas) != 3 {
		t.Fatalf("got %d schemas, want 3", len(schemas))
	}
	for _, s := range schemas {
		if s.Type != "function" || s.Function.Name == "" {
			t.Errorf("bad schema: %+v", s)
		}
		b, err := json.Marshal(s.Function.Parameters)
		if err != nil {
			t.Fatalf("schema parameters: %v", err)
		}
		if !json.Valid(b) {
			t.Errorf("invalid parameters JSON: %s", b)
		}
	}
}
