package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

	hist  *History
	agent *Agent
}

func main() {
	modelFlag := flag.String("model", "", "чат-модель (по умолчанию LLM_MODEL или deepseek-v4.1-flash)")
	summaryFlag := flag.String("summary-model", "", "модель для сжатия контекста (по умолчанию LLM_SUMMARY_MODEL или чат-модель)")
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
	systemFlag := flag.String("system", "", "файл с системным промптом")
	scriptFlag := flag.String("script", "", "проиграть запросы из файла и выйти")
	listModels := flag.Bool("list-models", false, "показать ростер моделей и выйти")
	listSessions := flag.Bool("list-sessions", false, "показать сохранённые сессии и выйти")
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
	a.agent.SetTrace(func(s string) { fmt.Println("  ↳ " + s) })
	return nil
}

func (a *app) repl() {
	interactive := isTerminal(os.Stdin)
	fmt.Printf("сессия %s · модель %s\n/help — команды, /exit — выход\n", a.hist.ID(), a.hist.ChatModel())
	if envModel := strings.TrimSpace(os.Getenv("LLM_MODEL")); envModel != "" && envModel != a.hist.ChatModel() {
		fmt.Printf("внимание: сессия сохранена с моделью %s, а LLM_MODEL=%s; переключить: /model %s\n",
			a.hist.ChatModel(), envModel, envModel)
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for {
		if interactive {
			fmt.Print("\nagent> ")
		}
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
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

	// A reasoning model can think for a while before the first byte, so show
	// that the process is alive instead of looking frozen.
	done := make(chan struct{})
	go func() {
		t := time.NewTicker(10 * time.Second)
		defer t.Stop()
		start := time.Now()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				fmt.Printf("  … ждём ответа (%d с)\n", int(time.Since(start).Seconds()))
			}
		}
	}()

	answer, err := a.agent.Chat(ctx, text)
	close(done)
	if err != nil {
		fmt.Println("ошибка:", err)
		return
	}
	fmt.Println()
	fmt.Println(answer)
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
		fmt.Printf("сессия: %s\nфайл: %s\nчат-модель: %s\nмодель сжатия: %s\n",
			a.hist.ID(), a.hist.Path(), a.hist.ChatModel(), a.hist.SummaryModel())

	case "/model":
		if arg == "" {
			printModels()
			fmt.Println("текущая:", a.hist.ChatModel())
			return false, nil
		}
		if _, ok := FindModel(arg); !ok {
			return false, fmt.Errorf("неизвестная модель %q; доступны: %s", arg, strings.Join(ModelIDs(), ", "))
		}
		a.hist.SetChatModel(arg)
		fmt.Println("чат-модель:", arg)

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
		fmt.Printf("[%d] %-9s %s\n", m.Seq, m.Role, oneLine(m.Content))
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

func printModels() {
	fmt.Println("Модели:")
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
	ids, err := ListSessions(dataDir)
	if err != nil {
		fatalf("%v", err)
	}
	if len(ids) == 0 {
		fmt.Println("сессий пока нет")
		return
	}
	for _, id := range ids {
		fmt.Println(id)
	}
}

func printHelp() {
	fmt.Print(`Команды:
  /help            эта справка
  /exit            выход (также /quit)
  /new             новая сессия
  /session         id, файл, модели
  /model [id]      показать ростер или переключить чат-модель
  /system          показать системный промпт
  /tools           список инструментов
  /history         последние сообщения активного окна
  /tokens          токены, архив, конспект, расходы API
  /compact         принудительно сжать контекст
  /recall <текст>  поиск по архиву

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
