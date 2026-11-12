package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// app wires the CLI, the agent and the session together.
type app struct {
	client       *ChatClient
	tools        *ToolRegistry
	dataDir      string
	system       string
	budget       Budget
	maxSteps     int
	temperature  float64
	maxToolChars int
	noCompact    bool

	hist  *History
	agent *Agent
	ui    *ui
}

// ui owns the single "status line" under the prompt: the one-character
// spinner and the trace lines (tool calls, compaction notices) share it, so
// every write is serialized through one mutex. Streamed answer deltas also go
// through it: the first token stops the spinner, and every consumer that
// prints a full line closes the streamed line first.
type ui struct {
	mu        sync.Mutex
	tty       bool
	spinning  bool
	streamed  bool // a streamed answer fragment is on screen, line not closed
	sawStream bool // any answer text was streamed during the current ask turn
}

const spinLine = "  / "

// spinFrames is the one-character spinner: / - \ |, one cell, in place.
const spinFrames = "/-\\|"

// begin starts the spinner. It returns a stop channel (nil when the spinner
// is off); waiting on it guarantees the line is cleared before the next print.
func (u *ui) begin(done <-chan struct{}) <-chan struct{} {
	if !u.tty {
		return nil
	}
	stop := make(chan struct{})
	u.mu.Lock()
	u.spinning = true
	u.mu.Unlock()
	go func() {
		defer close(stop)
		for i := 0; ; i++ {
			select {
			case <-done:
				u.mu.Lock()
				if u.spinning {
					fmt.Print("\r" + strings.Repeat(" ", len(spinLine)) + "\r")
					u.spinning = false
				}
				u.mu.Unlock()
				return
			case <-time.After(130 * time.Millisecond):
				u.mu.Lock()
				if u.spinning {
					fmt.Printf("\r  %s ", string(spinFrames[i%len(spinFrames)]))
				}
				u.mu.Unlock()
			}
		}
	}()
	return stop
}

// rearm turns the spinner back on before the next model request (the painting
// goroutine from begin() keeps running until done is closed).
func (u *ui) rearm() {
	if !u.tty {
		return
	}
	u.mu.Lock()
	u.spinning = true
	u.mu.Unlock()
}

// beginTurn marks the start of one ask() turn: nothing streamed yet.
func (u *ui) beginTurn() {
	u.mu.Lock()
	u.streamed = false
	u.sawStream = false
	u.mu.Unlock()
}

// content prints a streamed piece of the assistant answer, stopping the
// spinner on the first visible token.
func (u *ui) content(s string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.spinning {
		fmt.Print("\r" + strings.Repeat(" ", len(spinLine)) + "\r")
		u.spinning = false
	}
	u.streamed = true
	u.sawStream = true
	fmt.Print(s)
}

// endLine closes a streamed line if one is open and reports whether it did.
func (u *ui) endLine() bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.streamed {
		fmt.Println()
		u.streamed = false
		return true
	}
	return false
}

// didStream reports whether answer text was streamed in the current turn.
func (u *ui) didStream() bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.sawStream
}

// trace prints a tool/compaction line, wiping the spinner frame first and
// closing an unterminated streamed line.
func (u *ui) trace(s string) {
	u.mu.Lock()
	if u.spinning {
		fmt.Print("\r" + strings.Repeat(" ", len(spinLine)) + "\r")
	}
	if u.streamed {
		fmt.Println()
		u.streamed = false
	}
	fmt.Println("  ↳ " + s)
	u.mu.Unlock()
}

func main() {
	modelFlag := flag.String("model", "", "чат-модель (по умолчанию LLM_MODEL или deepseek-v4.1-flash)")
	summaryFlag := flag.String("summary-model", "", "модель для сжатия контекста (по умолчанию LLM_SUMMARY_MODEL или самая дешёвая в списке)")
	sessionFlag := flag.String("session", "", "id сессии")
	continueFlag := flag.Bool("continue", false, "продолжить последнюю сессию")
	dataFlag := flag.String("data", "", "каталог сессий")
	tasksFlag := flag.String("tasks", "tasks.txt", "список задач для find_task")
	maxSteps := flag.Int("max-steps", 8, "максимум шагов с инструментами на один запрос")
	maxPrompt := flag.Int("max-prompt-tokens", 6000, "бюджет токенов промпта")
	reserve := flag.Int("reserve-output", 1024, "токены, зарезервированные под ответ")
	minTail := flag.Int("min-tail-tokens", 1200, "сколько свежей истории держать дословно")
	temp := flag.Float64("temperature", 0.2, "температура сэмплирования")
	maxTokens := flag.Int("max-tokens", 4096, "лимит токенов ответа (включая рассуждения reasoning-моделей)")
	maxToolChars := flag.Int("max-tool-chars", 800, "сколько символов результата инструмента оставлять в промпте")
	noCompact := flag.Bool("no-compact", false, "отключить авто-сжатие контекста (демо переполнения)")
	systemFlag := flag.String("system", "", "файл с системным промптом")
	scriptFlag := flag.String("script", "", "проиграть запросы из файла и выйти")
	listModels := flag.Bool("list-models", false, "показать список моделей и выйти")
	listSessions := flag.Bool("list-sessions", false, "показать сохранённые сессии и выйти")
	printSession := flag.String("print-session", "", "показать весь лог сессии и выйти (офлайн, без модели)")
	flag.Parse()

	dataDir := *dataFlag
	if dataDir == "" {
		dataDir = defaultDataDir()
	}

	if *listModels {
		printModels()
		return
	}
	if *listSessions {
		printSessions(dataDir)
		return
	}
	if *printSession != "" {
		id := *printSession
		if id == "last" {
			var err error
			if id, err = LatestSession(dataDir); err != nil {
				fatalf("%v", err)
			}
		}
		if err := PrintSessionLog(dataDir, id, os.Stdout); err != nil {
			fatalf("%v", err)
		}
		return
	}

	system := DefaultSystemPrompt
	if *systemFlag != "" {
		b, err := os.ReadFile(*systemFlag)
		if err != nil {
			fatalf("system prompt: %v", err)
		}
		system = strings.TrimSpace(string(b))
	}

	tasks, err := loadTasks(*tasksFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "предупреждение: список задач не загружен (%v); find_task будет пустым\n", err)
	}

	client := NewChatClient()
	if client.baseURL == "" || client.apiKey == "" {
		fatalf("нужны переменные окружения LLM_BASE_URL и OPENAI_API_KEY")
	}
	client.SetMaxTokens(*maxTokens)

	a := &app{
		client:       client,
		tools:        NewToolRegistry(tasks),
		dataDir:      dataDir,
		system:       system,
		budget:       Budget{MaxPromptTokens: *maxPrompt, ReserveOutput: *reserve, MinTailTokens: *minTail},
		maxSteps:     *maxSteps,
		temperature:  *temp,
		maxToolChars: *maxToolChars,
		noCompact:    *noCompact,
		ui:           &ui{tty: isTerminal(os.Stdout)},
	}

	sessionID := *sessionFlag
	if *continueFlag {
		id, err := LatestSession(dataDir)
		if err != nil {
			fatalf("%v", err)
		}
		sessionID = id
	}
	if err := a.newSession(sessionID, *modelFlag, *summaryFlag); err != nil {
		fatalf("%v", err)
	}
	defer a.hist.Close()

	if *scriptFlag != "" {
		if err := a.runScript(*scriptFlag); err != nil {
			fatalf("%v", err)
		}
		return
	}
	a.repl()
}

// newSession opens (or creates) a session and rebuilds the agent around it.
func (a *app) newSession(id, requestedChat, requestedSummary string) error {
	if a.hist != nil {
		_ = a.hist.Close()
	}
	h, err := OpenHistory(a.dataDir, id, requestedChat, requestedSummary, a.system)
	if err != nil {
		return err
	}
	a.hist = h
	a.agent = NewAgent(a.client, a.tools, h, a.budget, a.maxSteps, a.temperature, a.maxToolChars)
	a.agent.SetTrace(a.ui.trace)
	a.agent.SetStreamUI(a.ui.rearm, a.ui.content, func() { a.ui.endLine() })
	a.agent.SetNoCompact(a.noCompact)
	return nil
}

func (a *app) repl() {
	fmt.Printf("сессия %s · модель %s · сжатие %s\n/help — команды: /model — модель, /session — сессия, /exit — выход; статистика — после каждого ответа\n", a.hist.ID(), a.hist.ChatModel(), a.hist.SummaryModel())
	if envModel := strings.TrimSpace(os.Getenv("LLM_MODEL")); envModel != "" && envModel != a.hist.ChatModel() {
		fmt.Printf("внимание: сессия сохранена с моделью %s, а LLM_MODEL=%s; переключить: /model %s\n",
			a.hist.ChatModel(), envModel, envModel)
	}

	ed := newLineEditor(isTerminal(os.Stdin), filepath.Join(a.dataDir, "history"))
	for {
		line, err := ed.read("agent> ")
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "/") {
			exit, err := a.command(line)
			if err != nil {
				fmt.Println("ошибка:", err)
			}
			if exit {
				break
			}
			continue
		}
		a.ask(line)
	}
	fmt.Println()
}

func (a *app) ask(text string) {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	done := make(chan struct{})
	a.ui.beginTurn()

	if a.ui.tty {
		// One-character spinner on the line under the prompt; the streamed
		// answer replaces it as soon as the first token arrives.
		stop := a.ui.begin(done)
		answer, err := a.agent.Chat(ctx, text)
		close(done)
		if stop != nil {
			<-stop
		}
		streamed := a.ui.didStream()
		a.ui.endLine()
		if err != nil {
			fmt.Println("ошибка:", err)
			return
		}
		if !streamed {
			fmt.Println()
			fmt.Println(answer)
		}
		a.footer()
		return
	}

	// Non-TTY (script mode): no spinner, deltas still stream to stdout; the
	// ticker only speaks while nothing has arrived yet.
	go func() {
		t := time.NewTicker(10 * time.Second)
		defer t.Stop()
		start := time.Now()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				if a.ui.didStream() {
					continue
				}
				fmt.Printf("  … ждём ответа (%d с)\n", int(time.Since(start).Seconds()))
			}
		}
	}()

	answer, err := a.agent.Chat(ctx, text)
	close(done)
	streamed := a.ui.didStream()
	a.ui.endLine()
	if err != nil {
		fmt.Println("ошибка:", err)
		return
	}
	if !streamed {
		fmt.Println()
		fmt.Println(answer)
	}
	a.footer()
}

// footer is the one-line stats line printed after every turn: model, prompt
// estimate vs budget, last answer size, active/archive split and accumulated
// API cost. An overflowing prompt is flagged explicitly.
func (a *app) footer() {
	activeMsgs, _, archivedMsgs, _ := a.hist.Stats()
	extra := ""
	if a.hist.HasSummary() {
		extra = fmt.Sprintf(" · конспект ~%d ток", EstimateTokens(a.hist.SummaryText()))
	}
	prompt := a.agent.comp.PromptTokens(a.hist.SystemPrompt())
	over := ""
	if prompt > a.budget.MaxPromptTokens {
		over = " · ПЕРЕПОЛНЕНИЕ"
	}
	s := a.client.Stats()
	fmt.Printf("· %s · промпт ~%d/%d ток · ответ ~%d ток · актив %d, архив %d сообщ%s · API %d выз., $%.4f%s\n",
		a.hist.ChatModel(),
		prompt, a.budget.MaxPromptTokens,
		a.agent.LastAnswerTokens(),
		activeMsgs, archivedMsgs, extra, s.Calls, s.Cost, over)
}

func (a *app) command(line string) (bool, error) {
	fields := strings.Fields(line)
	cmd := strings.ToLower(fields[0])
	arg := strings.TrimSpace(strings.TrimPrefix(line, fields[0]))

	switch cmd {
	case "/exit", "/quit", "/q":
		return true, nil

	case "/help":
		printHelp()

	case "/new":
		chat, summary := a.hist.ChatModel(), a.hist.SummaryModel()
		if err := a.newSession("", chat, summary); err != nil {
			return false, err
		}
		fmt.Println("новая сессия:", a.hist.ID())

	case "/session":
		if arg == "" {
			a.printSessionList()
			return false, nil
		}
		id := arg
		infos, err := SummarizeSessions(a.dataDir)
		if err != nil {
			return false, err
		}
		exists := false
		for _, inf := range infos {
			if inf.ID == id {
				exists = true
				break
			}
		}
		if !exists {
			return false, fmt.Errorf("нет сессии %s; /session без аргумента — список сессий", id)
		}
		if id == a.hist.ID() {
			fmt.Println("уже в этой сессии:", id)
			return false, nil
		}
		if err := a.newSession(id, "", ""); err != nil {
			return false, err
		}
		activeMsgs, _, archivedMsgs, _ := a.hist.Stats()
		fmt.Printf("сессия: %s · модель %s · сжатие %s · сообщений %d (актив %d, архив %d)\n",
			a.hist.ID(), a.hist.ChatModel(), a.hist.SummaryModel(), activeMsgs+archivedMsgs, activeMsgs, archivedMsgs)

	case "/model":
		if arg == "" {
			printModels()
			fmt.Println("текущая чат-модель:", a.hist.ChatModel())
			fmt.Printf("переключить: /model <id>, например /model %s\n", ModelSet[0].ID)
			return false, nil
		}
		if _, ok := FindModel(arg); !ok {
			return false, fmt.Errorf("неизвестная модель %q; доступны: %s", arg, strings.Join(ModelIDs(), ", "))
		}
		a.hist.SetChatModel(arg)
		fmt.Println("чат-модель:", arg)

	case "/summary":
		if arg == "" {
			printModels()
			fmt.Println("текущая модель сжатия:", a.hist.SummaryModel())
			fmt.Println("переключить: /summary <id>; вступит в силу со следующего сжатия, /compact — сжать сейчас")
			return false, nil
		}
		if _, ok := FindModel(arg); !ok {
			return false, fmt.Errorf("неизвестная модель %q; доступны: %s", arg, strings.Join(ModelIDs(), ", "))
		}
		if arg == a.hist.SummaryModel() {
			fmt.Println("модель сжатия уже:", arg)
			return false, nil
		}
		a.hist.SetSummaryModel(arg)
		fmt.Println("модель сжатия:", arg, "— вступит в силу со следующего сжатия (/compact — сжать сейчас)")

	case "/system":
		fmt.Println(a.hist.SystemPrompt())

	case "/tools":
		for _, s := range a.tools.Describe() {
			fmt.Println("•", s)
		}

	case "/history":
		a.printHistory()

	case "/tokens":
		a.printTokens()

	case "/compact":
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()
		ok, err := a.agent.comp.ForceCompact(ctx, a.hist.SystemPrompt())
		if err != nil {
			return false, err
		}
		if !ok {
			fmt.Println("сжимать нечего: история слишком короткая")
		}
		a.printTokens()

	case "/recall":
		if arg == "" {
			return false, fmt.Errorf("использование: /recall <текст>")
		}
		hits := a.hist.SearchArchived(arg)
		if len(hits) == 0 {
			fmt.Println("в архиве ничего не найдено")
			return false, nil
		}
		for i, m := range hits {
			if i == 10 {
				fmt.Println("…")
				break
			}
			fmt.Printf("[%d] %s: %s\n", m.Seq, m.Role, oneLine(m.Content))
		}

	default:
		fmt.Println("неизвестная команда; /help — список команд")
	}
	return false, nil
}

func (a *app) printHistory() {
	msgs := a.hist.Active()
	start := 0
	if len(msgs) > 20 {
		start = len(msgs) - 20
	}
	for _, m := range msgs[start:] {
		tok := m.Tokens
		if tok == 0 {
			tok = EstimateMessage(*m)
		}
		fmt.Printf("[%d] %-9s ~%-5d ток. %s\n", m.Seq, m.Role, tok, oneLine(m.Content))
	}
	if a.hist.HasSummary() {
		fmt.Println("(плюс конспект предыдущего диалога)")
	}
}

func (a *app) printTokens() {
	activeMsgs, activeTokens, archivedMsgs, archivedTokens := a.hist.Stats()
	fmt.Printf("активное окно: %d сообщений, ~%d ток.\n", activeMsgs, activeTokens)
	fmt.Printf("архив: %d сообщений, ~%d ток.\n", archivedMsgs, archivedTokens)
	if a.hist.HasSummary() {
		covers := a.hist.SummaryCovers()
		fmt.Printf("конспект: ~%d ток., покрывает seq %d..%d\n",
			EstimateTokens(a.hist.SummaryText()), covers[0], covers[1])
	}
	fmt.Printf("оценка промпта: ~%d ток. (бюджет %d, калибровка ×%.2f)\n",
		a.agent.comp.PromptTokens(a.hist.SystemPrompt()), a.budget.MaxPromptTokens, a.agent.calib.Ratio())
	fmt.Printf("модель сжатия: %s\n", a.hist.SummaryModel())

	s := a.client.Stats()
	fmt.Printf("API: %d вызовов, промпт %d, ответ %d, всего %d ток., $%.4f\n",
		s.Calls, s.PromptTokens, s.CompletionTokens, s.TotalTokens, s.Cost)
}

func (a *app) runScript(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fmt.Printf("agent> %s\n", line)
		a.ask(line)
	}
	return sc.Err()
}

// printSessionList shows all stored sessions with the current one marked.
func (a *app) printSessionList() {
	infos, err := SummarizeSessions(a.dataDir)
	if err != nil {
		fmt.Println("ошибка:", err)
		return
	}
	if len(infos) == 0 {
		fmt.Println("сессий пока нет")
		return
	}
	fmt.Println("сессии (сначала новые):")
	for _, inf := range infos {
		mark := " "
		if inf.ID == a.hist.ID() {
			mark = "*"
		}
		fmt.Printf(" %s %s · %s · модель %s · сжатие %s\n", mark, inf.ID, inf.Created.Format("02.01 15:04"), inf.ChatModel, inf.SummaryModel)
	}
	fmt.Println("* — текущая; переключиться: /session <id>")
}

func printModels() {
	fmt.Println("Модели — id это имя в первом столбце:")
	for _, m := range ModelSet {
		mark := " "
		if m.ID == DefaultModel() {
			mark = "*"
		}
		fmt.Printf(" %s %-22s %-10s $%.4f/1M\n", mark, m.ID, m.Tier, m.RatePerM)
	}
	fmt.Println("(* — по умолчанию)")
}

func printSessions(dataDir string) {
	infos, err := SummarizeSessions(dataDir)
	if err != nil {
		fatalf("%v", err)
	}
	if len(infos) == 0 {
		fmt.Println("сессий пока нет")
		return
	}
	for _, inf := range infos {
		fmt.Printf("%s · %s · модель %s · сжатие %s · сообщений %d\n",
			inf.ID, inf.Created.Format("02.01 15:04"), inf.ChatModel, inf.SummaryModel, inf.Msgs)
	}
}

func printHelp() {
	fmt.Print(`Команды:
  /help            эта справка
  /exit            выход (также /quit)
  /new             новая сессия
  /session [id]    список сессий или переключение на сессию
  /model [id]      список моделей или смена чат-модели
  /summary [id]    модель сжатия: список или смена
  /system          показать системный промпт
  /tools           список инструментов
  /history         последние сообщения активного окна
  /tokens          токены, архив, конспект, расходы API
  /compact         принудительно сжать контекст
  /recall <текст>  поиск по архиву

Полный лог сессии (все сообщения, архив, сжатия) — из оболочки, офлайн:
  ./task2-06 -print-session <id|last>

Ввод: ↑/↓ — история запросов (хранится между запусками), ←/→ — курсор,
Backspace, Ctrl-A/E — начало/конец строки, Ctrl-U/W — стереть строку/слово,
Ctrl-C — отменить ввод, Ctrl-D — выход.

Всё остальное отправляется модели.
`)
}

func defaultDataDir() string {
	if d := os.Getenv("XDG_DATA_HOME"); d != "" {
		return filepath.Join(d, "goagent", "sessions")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "data/sessions"
	}
	return filepath.Join(home, ".local", "share", "goagent", "sessions")
}

func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ошибка: "+format+"\n", args...)
	os.Exit(1)
}
