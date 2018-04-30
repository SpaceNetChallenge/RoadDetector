package common

import (
	"testing"
)

func TestBresenhamSlope1(t *testing.T) {
	t.Fatalf("%v", DrawLineOnCells(5, 5, 10, 15, 20, 20))
}
