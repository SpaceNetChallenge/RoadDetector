package common

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type Node struct {
	ID int
	Point Point
	In []*Edge
	Out []*Edge
}

func (node *Node) String() string {
	return fmt.Sprintf("Node(%v)", node.Point)
}

type Edge struct {
	ID int
	Src *Node
	Dst *Node
}

func (edge *Edge) Segment() Segment {
	return Segment{edge.Src.Point, edge.Dst.Point}
}

func (edge *Edge) Vector() Point {
	return edge.Segment().Vector()
}

func (edge *Edge) AngleTo(other *Edge) float64 {
	return edge.Segment().AngleTo(other.Segment())
}

func (edge *Edge) IsAdjacent(other *Edge) bool {
	for _, a := range []*Node{edge.Src, edge.Dst} {
		for _, b := range []*Node{other.Src, other.Dst} {
			if a == b {
				return true
			}
		}
	}
	return false
}

func (edge *Edge) ClosestPos(point Point) EdgePos {
	return EdgePos{
		Edge: edge,
		Position: edge.Segment().Project(point, false),
	}
}

func (edge *Edge) String() string {
	return fmt.Sprintf("Edge(%v -> %v)", edge.Src.Point, edge.Dst.Point)
}

type EdgePos struct {
	Edge *Edge
	Position float64
}

func (ep EdgePos) Point() Point {
	return ep.Edge.Segment().PointAtFactor(ep.Position, false)
}

type Graph struct {
	Nodes []*Node
	Edges []*Edge
}

func (graph *Graph) Bounds() Rectangle {
	r := EmptyRectangle
	for _, node := range graph.Nodes {
		r = r.Extend(node.Point)
	}
	return r
}

func (graph *Graph) LonLatToMeters(origin Point) {
	for _, node := range graph.Nodes {
		node.Point = node.Point.LonLatToMeters(origin)
	}
}

func (graph *Graph) MetersToLonLat(origin Point) {
	for _, node := range graph.Nodes {
		node.Point = node.Point.MetersToLonLat(origin)
	}
}

func (graph *Graph) AddNode(point Point) *Node {
	node := &Node{
		ID: len(graph.Nodes),
		Point: point,
	}
	graph.Nodes = append(graph.Nodes, node)
	return node
}

func (graph *Graph) AddEdge(src *Node, dst *Node) *Edge {
	for _, edge := range src.Out {
		if edge.Dst == dst {
			return edge
		}
	}
	edge := &Edge{
		ID: len(graph.Edges),
		Src: src,
		Dst: dst,
	}
	graph.Edges = append(graph.Edges, edge)
	edge.Src.Out = append(edge.Src.Out, edge)
	edge.Dst.In = append(edge.Dst.In, edge)
	return edge
}

func (graph *Graph) AddBidirectionalEdge(v1 *Node, v2 *Node) [2]*Edge {
	edge1 := graph.AddEdge(v1, v2)
	edge2 := graph.AddEdge(v2, v1)
	return [2]*Edge{edge1, edge2}
}

func (graph *Graph) GetSubgraphInRect(r Rectangle) *Graph {
	ngraph := &Graph{}
	nodeMap := make(map[int]*Node)
	for _, node := range graph.Nodes {
		if r.Contains(node.Point) {
			nodeMap[node.ID] = ngraph.AddNode(node.Point)
		}
	}
	for _, edge := range graph.Edges {
		if nodeMap[edge.Src.ID] != nil && nodeMap[edge.Dst.ID] != nil {
			ngraph.AddEdge(nodeMap[edge.Src.ID], nodeMap[edge.Dst.ID])
		}
	}
	return ngraph
}

func (graph *Graph) MakeBidirectional() {
	for _, edge := range graph.Edges {
		graph.AddEdge(edge.Dst, edge.Src)
	}
}

func (graph *Graph) Clone() *Graph {
	other := &Graph{}
	for _, node := range graph.Nodes {
		other.AddNode(node.Point)
	}
	for _, edge := range graph.Edges {
		other.AddEdge(other.Nodes[edge.Src.ID], other.Nodes[edge.Dst.ID])
	}
	return other
}

func ReadGraph(fname string) (*Graph, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	section := "vertices"
	var graph Graph

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return nil, err
			}
		}
		line = strings.TrimSpace(line)
		if section == "vertices" {
			if line == "" {
				section = "edges"
				continue
			}
			parts := strings.Split(line, " ")
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid vertex line: %s", line)
			}
			x, errx := strconv.ParseFloat(parts[0], 64)
			y, erry := strconv.ParseFloat(parts[1], 64)
			if errx != nil || erry != nil {
				return nil, fmt.Errorf("invalid vertex line: %s", line)
			}
			graph.AddNode(Point{x, y})
		} else if section == "edges" && line != "" {
			parts := strings.Split(line, " ")
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid edge line: %s", line)
			}
			src, errsrc := strconv.Atoi(parts[0])
			dst, errdst := strconv.Atoi(parts[1])
			if errsrc != nil || errdst != nil {
				return nil, fmt.Errorf("invalid edge line: %s", line)
			}
			graph.AddEdge(graph.Nodes[src], graph.Nodes[dst])
		}
	}

	return &graph, nil
}

func (graph *Graph) Write(fname string) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()

	// vertices
	for _, node := range graph.Nodes {
		if _, err := file.Write([]byte(fmt.Sprintf("%f %f\n", node.Point.X, node.Point.Y))); err != nil {
			return err
		}
	}
	file.Write([]byte("\n"))

	// edges
	for _, edge := range graph.Edges {
		if _, err := file.Write([]byte(fmt.Sprintf("%d %d\n", edge.Src.ID, edge.Dst.ID))); err != nil {
			return err
		}
	}

	return nil
}

func VisualizeGraphs(scale float64, fname string, graphs []*Graph, traces []*Trace) error {
	if len(graphs) == 0 {
		return fmt.Errorf("at least one graph must be provided")
	}
	var graphBoundables []Boundable
	for _, graph := range graphs[1:] {
		graphBoundables = append(graphBoundables, graph)
	}
	var options SVGOptions
	if scale > 0 {
		options.Scale = scale
	}
	var boundables [][]Boundable
	boundables = append(boundables, []Boundable{graphs[0]})
	if len(graphBoundables) > 0 {
		boundables = append(boundables, graphBoundables)
	}
	if len(traces) > 0 {
		boundables = append(boundables, []Boundable{Traces(traces)})
	}
	return CreateSVG(fname, boundables, options)
}
