package common

import (
	"math"
	"testing"
)

func TestPointAngleTo(t *testing.T) {
	point1 := Point{1, 1}
	point2 := Point{1, 0}
	point3 := Point{-1, 0}
	check := func(a Point, b Point, expected float64) {
		got := a.AngleTo(b)
		if math.Abs(got - expected) > 0.001 {
			t.Fatalf("expected %f for angle from %v to %v, but got %f", expected, a, b, got)
		}
	}
	check(point1, point1, 0)
	check(point1, point2, math.Pi / 4)
	check(point2, point3, math.Pi)
	check(point1, point3, math.Pi * 3 / 4)
	check(point3, point1, math.Pi * 3 / 4)
}

func TestSegmentDistance(t *testing.T) {
	segment := Segment{
		Point{0, 0},
		Point{1, 0},
	}
	check := func(point Point, expected float64) {
		got := segment.Distance(point)
		if math.Abs(got - expected) > 0.001 {
			t.Fatalf("expected %f for distance to %v, but got %f", expected, point, got)
		}
	}
	check(Point{0, 0}, 0)
	check(Point{0.5, 0}, 0)
	check(Point{-0.5, 0}, 0.5)
	check(Point{0.5, 0.5}, 0.5)
}

func TestSegmentDistanceToSegment(t *testing.T) {
	check := func(a Segment, b Segment, expected float64) {
		got := a.DistanceToSegment(b)
		if math.Abs(got - expected) > 0.001 {
			t.Fatalf("expected %f for distance from %v to %v, but got %f", expected, a, b, got)
		}
	}
	segment1 := Segment{Point{0, 0}, Point{1, 0}}
	segment2 := Segment{Point{0, 0.5}, Point{1, -0.5}}
	check(segment1, segment2, 0)
	segment3 := Segment{Point{0, 1}, Point{1, 1}}
	check(segment1, segment3, 1)
	check(segment2, segment3, 0.5)
	segment4 := Segment{Point{-1, 0}, Point{0, 1}}
	check(segment1, segment4, math.Sqrt(0.5))
}

func TestSegmentIntersection(t *testing.T) {
	check := func(a Segment, b Segment, expected *Point) {
		got := a.Intersection(b)
		if expected == nil && got != nil {
			t.Fatalf("expected nil from %v to %v, but got %v", a, b, *got)
		} else if expected != nil && got == nil {
			t.Fatalf("expected %v from %v to %v, but got nil", *expected, a, b)
		} else if expected != nil && got != nil {
			ep := *expected
			gp := *got
			if ep.Distance(gp) > 0.001 {
				t.Fatalf("expected %v from %v to %v, but got %v", ep, a, b, gp)
			}
		}
	}
	segment1 := Segment{Point{0, 0}, Point{1, 0}}
	segment2 := Segment{Point{0, 0.5}, Point{1, -0.5}}
	check(segment1, segment2, &Point{0.5, 0})
	segment3 := Segment{Point{0, 1}, Point{1, 1}}
	check(segment1, segment3, nil)
	check(segment2, segment3, nil)
	segment4 := Segment{Point{0, -1}, Point{0, 1}}
	check(segment1, segment4, &Point{0, 0})
}

func TestSegmentProjectWithWidth(t *testing.T) {
	segment := Segment{Point{10, 10}, Point{20, 10}}
	width := 2.0
	check := func(p Point, expected Point) {
		got := segment.ProjectWithWidth(p, width)
		if math.Abs(got.X - expected.X) > 0.001 || math.Abs(got.Y - expected.Y) > 0.001 {
			t.Fatalf("expected %v for projection of %v, but got %v", expected, p, got)
		}
	}
	check(Point{10, 5}, Point{10, 9})
	check(Point{10, 50}, Point{10, 11})
	check(Point{15, 15}, Point{15, 11})
	check(Point{15, 10.5}, Point{15, 10.5})
	check(Point{100, 10}, Point{21, 10})
}

func TestLineProjectPoint(t *testing.T) {
	line := Line{Point{10, 10}, Point{20, 10}}
	check := func(p Point, expected Point) {
		got := line.ProjectPoint(p)
		if got.Distance(expected) > 0.001 {
			t.Fatalf("expected %v for projection of %v, but got %v", expected, p, got)
		}
	}
	check(Point{15, 5}, Point{15, 10})
	check(Point{100, 5}, Point{100, 10})
}

func TestRectangleIntersects(t *testing.T) {
	check := func(a Rectangle, b Rectangle, expected bool) {
		got := a.Intersects(b)
		if got != expected {
			t.Fatalf("expected %v for %v.Intersects(%v), but got %v", expected, a, b, got)
		}
	}
	check(Rectangle{Point{0, 0}, Point{1, 1}}, Rectangle{Point{2, 0}, Point{3, 1}}, false)
	check(Rectangle{Point{0, 0}, Point{1, 1}}, Rectangle{Point{-1, -1}, Point{0.1, 0.1}}, true)
	check(Rectangle{Point{0, 0}, Point{1, 1}}, Rectangle{Point{0.5, 0.5}, Point{0.6, 0.6}}, true)
}
