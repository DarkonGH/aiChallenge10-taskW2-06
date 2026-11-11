package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLineEditorHistoryRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "history")

	ed := newLineEditor(false, path)
	ed.append("привет")
	ed.append("сколько будет 17*23?")
	ed.append("сколько будет 17*23?") // соседний дубликат — не пишется
	ed.append("")                     // пустая — не пишется

	// Файл: две строки.
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	if got := strings.Count(strings.TrimRight(string(b), "\n"), "\n") + 1; got != 2 {
		t.Fatalf("в файле %d строк, хотим 2:\n%s", got, b)
	}

	// Новый редактор подхватывает историю.
	ed2 := newLineEditor(true, path)
	if len(ed2.hist) != 2 || ed2.hist[0] != "привет" || ed2.hist[1] != "сколько будет 17*23?" {
		t.Fatalf("hist = %q", ed2.hist)
	}
}

func TestLoadHistoryFileTrimsAndCaps(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "history")

	var sb strings.Builder
	for i := 0; i < maxHistoryLines+50; i++ {
		sb.WriteString("строка-")
		sb.WriteString(fmt.Sprintf("%04d", i))
		sb.WriteString("\n")
		if i%10 == 0 {
			sb.WriteString("\n") // пустые строки пропускаются
		}
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o600); err != nil {
		t.Fatal(err)
	}

	h := loadHistoryFile(path)
	if len(h) != maxHistoryLines {
		t.Fatalf("len = %d, хотим %d", len(h), maxHistoryLines)
	}
}

func TestRuneBuf(t *testing.T) {
	var rb runeBuf

	// ASCII по байту.
	if r, ok := rb.push('a'); !ok || r != 'a' {
		t.Fatalf("ascii: %q %v", r, ok)
	}

	// Кириллица собирается из двух байтов.
	var assembled []rune
	for _, b := range []byte("привет") {
		if r, ok := rb.push(b); ok {
			assembled = append(assembled, r)
		}
	}
	if string(assembled) != "привет" {
		t.Fatalf("utf8: %q", string(assembled))
	}

	// Битая последовательность: 0xD0 без продолжения, затем управляющий байт —
	// буфер сбрасывается и не «съедает» следующий ввод.
	rb.push(0xD0)
	if _, ok := rb.push(0x03); ok {
		t.Fatal("управляющий байт не должен порождать руну")
	}
	if r, ok := rb.push('z'); !ok || r != 'z' {
		t.Fatalf("после сброса: %q %v", r, ok)
	}
}
