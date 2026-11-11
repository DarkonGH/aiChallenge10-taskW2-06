//go:build darwin

package main

// Сырой режим терминала для lineEditor на macOS: termios через stdlib
// syscall, без внешних зависимостей.

import (
	"fmt"
	"syscall"
	"unsafe"
)

type rawState struct {
	saved syscall.Termios
	fd    uintptr
}

// makeRaw отключает канонический режим, эхо и генерацию сигналов (ISIG), чтобы
// программа сама видела Ctrl-C/Ctrl-D и escape-последовательности стрелок.
func makeRaw(fd uintptr) (*rawState, error) {
	var old syscall.Termios
	if err := ioctlTermios(fd, syscall.TIOCGETA, &old); err != nil {
		return nil, err
	}
	raw := old
	raw.Lflag &^= syscall.ECHO | syscall.ICANON | syscall.ISIG | syscall.IEXTEN
	raw.Iflag &^= syscall.IXON | syscall.ICRNL | syscall.INLCR | syscall.IGNCR
	raw.Cc[syscall.VMIN] = 1
	raw.Cc[syscall.VTIME] = 0
	if err := ioctlTermios(fd, syscall.TIOCSETA, &raw); err != nil {
		return nil, err
	}
	st := &rawState{saved: old, fd: fd}
	rawSetActive(st)
	return st, nil
}

func (s *rawState) restore() error {
	rawSetActive(nil)
	return ioctlTermios(s.fd, syscall.TIOCSETA, &s.saved)
}

func ioctlTermios(fd uintptr, req uint, t *syscall.Termios) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, fd, uintptr(req), uintptr(unsafe.Pointer(t)))
	if errno != 0 {
		return fmt.Errorf("ioctl: %v", errno)
	}
	return nil
}
