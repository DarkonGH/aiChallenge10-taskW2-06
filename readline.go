package main

// lineEditor — минимальный readline для интерактивного REPL: история по
// стрелкам ↑/↓ (с сохранением в файл между запусками), курсор ←/→, правка
// строки и несколько управляющих клавиш. Терминал на время чтения
// переключается в сырой режим (readline_raw_darwin.go), поэтому Backspace и
// escape-последовательности долетают до программы как есть, а не как
// отредактированная строка. Вне TTY (пайпы, -script) ввод остаётся прежним —
// обычный построчный сканер.

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"unicode/utf8"
)

// maxHistoryLines — сколько последних строк держать в файле истории.
const maxHistoryLines = 1000

type lineEditor struct {
	tty      bool
	histPath string
	hist     []string
	histPos  int // позиция в истории; len(hist) — «свежая» строка (черновик)
	draft    string

	in *bufio.Reader  // tty: побайтовое чтение
	sc *bufio.Scanner // не-tty: построчное чтение
}

func newLineEditor(tty bool, histPath string) *lineEditor {
	ed := &lineEditor{tty: tty, histPath: histPath}
	if tty {
		ed.hist = loadHistoryFile(histPath)
		ed.in = bufio.NewReader(os.Stdin)
	}
	return ed
}

// read возвращает одну строку ввода. В tty-режиме рисует промпт и обрабатывает
// клавиши; вне tty просто читает строку из stdin. Ошибка io.EOF — конец ввода
// (Ctrl-D на пустой строке или закрытый stdin).
func (ed *lineEditor) read(prompt string) (string, error) {
	if !ed.tty {
		return ed.readPlain()
	}
	raw, err := makeRaw(os.Stdin.Fd())
	if err != nil {
		// Не удалось перевести терминал в сырой режим — деградируем до
		// обычного построчного ввода без истории.
		return ed.readPlain()
	}
	defer raw.restore()

	ed.histPos = len(ed.hist)
	ed.draft = ""

	fmt.Print("\n")
	line := make([]rune, 0, 64)
	cur := 0 // позиция курсора в рунах
	draw := func() {
		fmt.Printf("\r\x1b[K%s%s", prompt, string(line))
		if cur < len(line) {
			fmt.Printf("\x1b[%dD", len(line)-cur)
		}
	}
	draw()

	var rb runeBuf
	for {
		b, err := ed.in.ReadByte()
		if err != nil {
			return "", err
		}
		switch b {
		case 0x1b: // escape-последовательность стрелок
			b2, err := ed.in.ReadByte()
			if err != nil {
				return "", err
			}
			if b2 != '[' && b2 != 'O' {
				rb.reset()
				continue // одиночный ESC — игнорируем
			}
			b3, err := ed.in.ReadByte()
			if err != nil {
				return "", err
			}
			switch b3 {
			case 'A': // ↑ — назад по истории
				if len(ed.hist) == 0 {
					continue
				}
				if ed.histPos == len(ed.hist) {
					ed.draft = string(line) // запомнить недосланный ввод
				}
				if ed.histPos > 0 {
					ed.histPos--
				}
				line = []rune(ed.hist[ed.histPos])
				cur = len(line)
				draw()
			case 'B': // ↓ — вперёд по истории
				if ed.histPos >= len(ed.hist) {
					continue
				}
				ed.histPos++
				if ed.histPos == len(ed.hist) {
					line = []rune(ed.draft) // вернуться к черновику
				} else {
					line = []rune(ed.hist[ed.histPos])
				}
				cur = len(line)
				draw()
			case 'C': // →
				if cur < len(line) {
					cur++
					draw()
				}
			case 'D': // ←
				if cur > 0 {
					cur--
					draw()
				}
			}
		case '\r', '\n':
			fmt.Print("\n") // завершить строку ввода, иначе следующий вывод клеится к промпту
			s := strings.TrimSpace(string(line))
			ed.append(s)
			return s, nil
		case 0x7f, 0x08: // Backspace
			if cur > 0 {
				line = append(line[:cur-1], line[cur:]...)
				cur--
				draw()
			}
		case 0x03: // Ctrl-C — отменить строку, вернуться к пустому промпту
			fmt.Print("\r\x1b[K^C")
			rb.reset()
			return "", nil
		case 0x04: // Ctrl-D на пустой строке — выход
			if len(line) == 0 {
				fmt.Println()
				return "", io.EOF
			}
		case 0x01: // Ctrl-A — в начало строки
			cur = 0
			draw()
		case 0x05: // Ctrl-E — в конец строки
			cur = len(line)
			draw()
		case 0x15: // Ctrl-U — стереть всю строку
			line = line[:0]
			cur = 0
			draw()
		case 0x17: // Ctrl-W — стереть слово перед курсором
			j := cur
			for j > 0 && line[j-1] == ' ' {
				j--
			}
			for j > 0 && line[j-1] != ' ' {
				j--
			}
			line = append(line[:j], line[cur:]...)
			cur = j
			draw()
		default:
			if b < 0x20 { // Tab и прочее непечатное — игнорируем
				rb.reset()
				continue
			}
			r, ok := rb.push(b)
			if !ok {
				continue // ждём продолжение многобайтного символа
			}
			line = append(line, 0)
			copy(line[cur+1:], line[cur:])
			line[cur] = r
			cur++
			draw()
		}
	}
}

// readPlain — построчный ввод без сырого режима.
func (ed *lineEditor) readPlain() (string, error) {
	if ed.sc == nil {
		ed.sc = bufio.NewScanner(os.Stdin)
		ed.sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	}
	if !ed.sc.Scan() {
		return "", io.EOF
	}
	return ed.sc.Text(), nil
}

// append добавляет строку в историю (без пустых и соседних дубликатов) и
// дописывает её в файл истории.
func (ed *lineEditor) append(line string) {
	if line == "" {
		return
	}
	if n := len(ed.hist); n > 0 && ed.hist[n-1] == line {
		return
	}
	ed.hist = append(ed.hist, line)
	if len(ed.hist) > maxHistoryLines {
		ed.hist = ed.hist[len(ed.hist)-maxHistoryLines:]
	}
	if ed.histPath == "" {
		return
	}
	_ = os.MkdirAll(filepath.Dir(ed.histPath), 0o755)
	f, err := os.OpenFile(ed.histPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return
	}
	fmt.Fprintln(f, line)
	_ = f.Close()
}

// loadHistoryFile читает файл истории: по строке на ввод, без пустых и
// соседних дубликатов, максимум maxHistoryLines последних строк.
func loadHistoryFile(path string) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []string
	for _, l := range strings.Split(string(b), "\n") {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		if len(out) > 0 && out[len(out)-1] == l {
			continue
		}
		out = append(out, l)
	}
	if len(out) > maxHistoryLines {
		out = out[len(out)-maxHistoryLines:]
	}
	return out
}

// runeBuf собирает многобайтный UTF-8 символ из отдельных байтов сырого
// ввода и отдаёт готовую руну.
type runeBuf struct {
	buf []byte
}

func (r *runeBuf) reset() { r.buf = r.buf[:0] }

func (r *runeBuf) push(b byte) (rune, bool) {
	if len(r.buf) == 0 && b < utf8.RuneSelf {
		return rune(b), true // ASCII — сразу
	}
	// Управляющий байт внутри последовательности — последовательность битая.
	if b < 0x20 || b == 0x7f {
		r.reset()
		return 0, false
	}
	r.buf = append(r.buf, b)
	if len(r.buf) > utf8.UTFMax {
		r.reset()
		return 0, false
	}
	// Незаконченная последовательность: ждём, пока наберётся длина,
	// заявленная первым байтом (DecodeRune не различает «нужно ещё» и «битое»).
	if want := utf8SeqLen(r.buf[0]); len(r.buf) < want {
		return 0, false
	}
	ru, size := utf8.DecodeRune(r.buf)
	r.reset()
	if size == 0 || ru == utf8.RuneError {
		return 0, false // некорректная последовательность — пропускаем
	}
	return ru, true
}

// utf8SeqLen — сколько байтов занимает символ с таким начальным байтом.
func utf8SeqLen(b byte) int {
	switch {
	case b&0x80 == 0x00:
		return 1
	case b&0xE0 == 0xC0:
		return 2
	case b&0xF0 == 0xE0:
		return 3
	case b&0xF8 == 0xF0:
		return 4
	}
	return 1 // некорректный стартовый байт — пусть DecodeRune отбракует
}

// Восстановление терминала при внешнем завершении процесса (kill): сырой
// режим без этого остался бы включённым после смерти агента.
var (
	rawMu      sync.Mutex
	rawCurrent *rawState
)

func init() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)
	go func() {
		for sig := range c {
			rawMu.Lock()
			st := rawCurrent
			rawMu.Unlock()
			if st == nil {
				// Сырой режим не включён — возвращаем системе поведение
				// по умолчанию и переотправляем сигнал себе.
				signal.Reset(sig)
				if s, ok := sig.(syscall.Signal); ok {
					_ = syscall.Kill(syscall.Getpid(), s)
				}
				continue
			}
			_ = st.restore()
			os.Exit(128 + int(sig.(syscall.Signal)))
		}
	}()
}

func rawSetActive(st *rawState) {
	rawMu.Lock()
	rawCurrent = st
	rawMu.Unlock()
}
