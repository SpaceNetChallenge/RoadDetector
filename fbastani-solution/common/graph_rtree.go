package common

import (
	"github.com/dhconnelly/rtreego"

	"math"
)

func RtreegoRect(r Rectangle) *rtreego.Rect {
	dx := math.Max(0.00000001, r.Max.X - r.Min.X)
	dy := math.Max(0.00000001, r.Max.Y - r.Min.Y)
	rect, err := rtreego.NewRect(rtreego.Point{r.Min.X, r.Min.Y}, []float64{dx, dy})
	if err != nil {
		panic(err)
	}
	return rect
}

type edgeSpatial struct {
	edge *Edge
	rect *rtreego.Rect
}

func (e *edgeSpatial) Bounds() *rtreego.Rect {
	if e.rect == nil {
		r := e.edge.Src.Point.Rectangle()
		r = r.Extend(e.edge.Dst.Point)
		e.rect = RtreegoRect(r)
	}
	return e.rect
}

type Rtree struct {
	tree *rtreego.Rtree
}

func (rtree Rtree) Search(rect Rectangle) []*Edge {
	spatials := rtree.tree.SearchIntersect(RtreegoRect(rect))
	edges := make([]*Edge, len(spatials))
	for i := range spatials {
		edges[i] = spatials[i].(*edgeSpatial).edge
	}
	return edges
}

func (graph *Graph) Rtree() Rtree {
	rtree := rtreego.NewTree(2, 25, 50)
	for _, edge := range graph.Edges {
		rtree.Insert(&edgeSpatial{edge: edge})
	}
	return Rtree{rtree}
}
