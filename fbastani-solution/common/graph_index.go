package common

import (
	"math"
)

type GridIndex struct {
	gridSize float64
	grid map[[2]int][]int
	graph *Graph
}

func (idx GridIndex) eachCell(rect Rectangle, f func(i int, j int)) {
	for i := int(math.Floor(rect.Min.X / idx.gridSize)); i <= int(math.Floor(rect.Max.X / idx.gridSize)); i++ {
		for j := int(math.Floor(rect.Min.Y / idx.gridSize)); j <= int(math.Floor(rect.Max.Y / idx.gridSize)); j++ {
			f(i, j)
		}
	}
}

func (idx GridIndex) Search(rect Rectangle) []*Edge {
	edgeIDs := make(map[int]bool)
	idx.eachCell(rect, func(i int, j int) {
		for _, edgeID := range idx.grid[[2]int{i, j}] {
			edgeIDs[edgeID] = true
		}
	})
	edges := make([]*Edge, 0, len(edgeIDs))
	for edgeID := range edgeIDs {
		edges = append(edges, idx.graph.Edges[edgeID])
	}
	return edges
}

func (graph *Graph) GridIndex(gridSize float64) GridIndex {
	idx := GridIndex{
		gridSize: gridSize,
		grid: make(map[[2]int][]int),
		graph: graph,
	}
	for _, edge := range graph.Edges {
		idx.eachCell(edge.Segment().Bounds(), func(i int, j int) {
			idx.grid[[2]int{i, j}] = append(idx.grid[[2]int{i, j}], edge.ID)
		})
	}
	return idx
}
