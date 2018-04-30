package common

import (
	"math"
	"testing"
)

func TestEdgeAngleTo(t *testing.T) {
	zero := &Node{Point: Point{0, 0}}
	v1 := &Node{Point: Point{1, 1}}
	v2 := &Node{Point: Point{1, 0}}
	v3 := &Node{Point: Point{1, -1}}
	e1 := &Edge{Src: zero, Dst: v1}
	e2 := &Edge{Src: zero, Dst: v2}
	e3 := &Edge{Src: zero, Dst: v3}
	e4 := &Edge{Src: v1, Dst: zero}
	check := func(label string, expected float64, got float64) {
		if math.Abs(expected - got) > 0.001 {
			t.Fatalf("%s: expected %f but got %f", label, expected, got)
		}
	}
	check("e1->e2", math.Pi / 4, e1.AngleTo(e2))
	check("e2->e1", math.Pi / 4, e2.AngleTo(e1))
	check("e1->e3", math.Pi / 2, e1.AngleTo(e3))
	check("e1->e4", math.Pi, e1.AngleTo(e4))
}
