//go:build !darwin

package main

// Заглушка сырого режима для платформ, отличных от macOS: lineEditor
// деградирует до обычного построчного ввода без стрелок.

import "fmt"

type rawState struct{}

func makeRaw(fd uintptr) (*rawState, error) {
	return nil, fmt.Errorf("raw mode не поддерживается на этой платформе")
}

func (s *rawState) restore() error { return nil }
