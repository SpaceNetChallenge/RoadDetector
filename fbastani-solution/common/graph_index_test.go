package common

import (
	"testing"
)

func TestGridIndexSearch(t *testing.T) {
	graph := &Graph{}
	v00 := graph.AddNode(Point{0, 0})
	v01 := graph.AddNode(Point{1, 0})
	v02 := graph.AddNode(Point{2, 0})
	v11 := graph.AddNode(Point{1, 1})
	e00t01 := graph.AddEdge(v00, v01)
	e00t02 := graph.AddEdge(v00, v02)
	e00t11 := graph.AddEdge(v00, v11)
	e01t11 := graph.AddEdge(v01, v11)
	idx := graph.GridIndex(0.3)
	check := func(rect Rectangle, edges []*Edge) {
		edgeIDs := make(map[int]bool)
		for _, edge := range edges {
			edgeIDs[edge.ID] = true
		}
		got := idx.Search(rect)
		if len(got) != len(edgeIDs) {
			t.Errorf("expected %d edges but got %d for %v", len(edgeIDs), len(got), rect)
			return
		}
		for _, edge := range got {
			if !edgeIDs[edge.ID] {
				t.Errorf("got edge %d (%v) unexpectedly for %v", edge.ID, edge.Segment(), rect)
				return
			}
		}
	}
	check(Rectangle{Point{-0.1, -0.1}, Point{0.1, 0.1}}, []*Edge{e00t01, e00t02, e00t11})
	check(Rectangle{Point{0.4, 0.4}, Point{0.6, 0.6}}, []*Edge{e00t11})
	check(Rectangle{Point{1.7, -0.01}, Point{1.71, 0.01}}, []*Edge{e00t02})
	check(Rectangle{Point{0.4, 0.4}, Point{1.1, 0.6}}, []*Edge{e00t11, e01t11})
}
