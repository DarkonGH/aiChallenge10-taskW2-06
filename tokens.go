package main

import (
	"math"
	"unicode/utf8"
)

// charsPerToken is a rough ratio for mixed Russian/English text. It is only a
// starting point: the calibrator corrects it against real usage.prompt_tokens.
const charsPerToken = 2.7

// EstimateTokens returns a cheap token estimate for a string.
func EstimateTokens(s string) int {
	if s == "" {
		return 0
	}
	r := utf8.RuneCountInString(s)
	n := int(math.Ceil(float64(r) / charsPerToken))
	if n < 1 {
		n = 1
	}
	return n
}

// EstimateMessage estimates the prompt cost of one message, including the
// per-message overhead and tool call metadata.
func EstimateMessage(m Message) int {
	n := EstimateTokens(m.Content) + 4
	for _, tc := range m.ToolCalls {
		n += EstimateTokens(tc.Function.Name) + EstimateTokens(tc.Function.Arguments) + 4
	}
	return n
}

// Calibrator scales estimates by the observed estimate/actual ratio, so the
// budget reacts to the real tokenizer without shipping one.
type Calibrator struct {
	ratio float64
}

// NewCalibrator returns a calibrator that trusts raw estimates.
func NewCalibrator() *Calibrator {
	return &Calibrator{ratio: 1}
}

// Scale converts a raw estimate into a calibrated one.
func (c *Calibrator) Scale(n int) int {
	v := int(math.Ceil(float64(n) * c.ratio))
	if v < 1 {
		v = 1
	}
	return v
}

// Update folds one real measurement into the ratio.
func (c *Calibrator) Update(estimated, actual int) {
	if estimated <= 0 || actual <= 0 {
		return
	}
	observed := float64(actual) / float64(estimated)
	if observed < 0.3 {
		observed = 0.3
	}
	if observed > 3 {
		observed = 3
	}
	c.ratio = 0.7*c.ratio + 0.3*observed
}

// Ratio exposes the current calibration factor.
func (c *Calibrator) Ratio() float64 { return c.ratio }
