package common

import (
	"math"
)

// returns shortest distances from the given source node to all vertices in the graph
func (graph *Graph) ShortestDistancesFromSource(src *Node) map[int]float64 {
	result := graph.ShortestPath(src, ShortestPathParams{})
	return result.Distances
}

type ShortestPathParams struct {
	// maximum distance to travel from src
	MaxDistance float64

	// terminate search once we reach any of these nodes
	StopNodes []*Node
}

func (params ShortestPathParams) IsStopNode(node *Node) bool {
	for _, other := range params.StopNodes {
		if other == node {
			return true
		}
	}
	return false
}

type ShortestPathResult struct {
	source *Node
	graph *Graph
	Distances map[int]float64
	Remaining map[int]bool
	Backpointers map[int]int
}

func (result ShortestPathResult) GetPathTo(node *Node) []*Node {
	if result.Remaining[node.ID] {
		return nil
	} else if _, ok := result.Backpointers[node.ID]; !ok {
		return nil
	}

	var reverseSeq []*Node
	curNode := node
	for curNode.ID != result.source.ID {
		reverseSeq = append(reverseSeq, curNode)
		curNode = result.graph.Nodes[result.Backpointers[curNode.ID]]
	}
	path := make([]*Node, len(reverseSeq))
	for i, node := range reverseSeq {
		path[len(path) - i - 1] = node
	}
	return path
}

func (graph *Graph) ShortestPath(src *Node, params ShortestPathParams) ShortestPathResult {
	// use Dijkstra's algorithm
	distances := make(map[int]float64)
	remaining := make(map[int]bool)
	backpointers := make(map[int]int)
	for _, node := range graph.Nodes {
		distances[node.ID] = math.Inf(1)
		remaining[node.ID] = true
	}

	distances[src.ID] = 0
	backpointers[src.ID] = src.ID
	for len(remaining) > 0 {
		var closestNode *Node
		var closestDistance float64
		for nodeID := range remaining {
			if !math.IsInf(distances[nodeID], 1) && (closestNode == nil || distances[nodeID] < closestDistance) {
				closestNode = graph.Nodes[nodeID]
				closestDistance = distances[nodeID]
			}
		}
		if closestNode == nil {
			break
		}
		delete(remaining, closestNode.ID)
		if (params.MaxDistance != 0 && closestDistance > params.MaxDistance) || params.IsStopNode(closestNode) {
			break
		}

		for _, edge := range closestNode.Out {
			d := closestDistance + edge.Segment().Length()
			if remaining[edge.Dst.ID] && d < distances[edge.Dst.ID] {
				distances[edge.Dst.ID] = d
				backpointers[edge.Dst.ID] = closestNode.ID
			}
		}
	}

	return ShortestPathResult{
		source: src,
		graph: graph,
		Distances: distances,
		Remaining: remaining,
		Backpointers: backpointers,
	}
}

type FollowParams struct {
	// Source, only one should be specified.
	SourceNodes []*Node
	SourcePos EdgePos

	// Distance to travel along graph from source.
	Distance float64

	// If true, don't search forwards.
	NoForwards bool

	// If true, search backwards (in addition to searching forwards).
	Backwards bool

	// If set, will be populated with nodes that we pass during following.
	SeenNodes map[int]bool

	// If set, we will stop immediately before these nodes rather than passing them.
	StopNodes map[int]bool
}

// Find locations after traveling along the graph from pos for distance.
func (graph *Graph) Follow(params FollowParams) []EdgePos {
	seenNodePairs := make(map[[2]int]bool)
	var positions []EdgePos

	var followNode func(node *Node, remaining float64, backwards bool)

	followForwards := func(pos EdgePos, remaining float64) {
		seenNodePairs[[2]int{pos.Edge.Src.ID, pos.Edge.Dst.ID}] = true

		if pos.Position + remaining <= pos.Edge.Segment().Length() {
			positions = append(positions, EdgePos{
				pos.Edge,
				pos.Position + remaining,
			})
		} else if params.StopNodes != nil && params.StopNodes[pos.Edge.Dst.ID] {
			positions = append(positions, EdgePos{
				pos.Edge,
				pos.Edge.Segment().Length(),
			})
		} else {
			followNode(pos.Edge.Dst, remaining - (pos.Edge.Segment().Length() - pos.Position), false)
		}
	}

	followBackwards := func(pos EdgePos, remaining float64) {
		seenNodePairs[[2]int{pos.Edge.Src.ID, pos.Edge.Dst.ID}] = true

		if remaining <= pos.Position {
			positions = append(positions, EdgePos{
				pos.Edge,
				pos.Position - remaining,
			})
		} else if params.StopNodes != nil && params.StopNodes[pos.Edge.Src.ID] {
			positions = append(positions, EdgePos{
				pos.Edge,
				0,
			})
		} else {
			followNode(pos.Edge.Src, remaining - pos.Position, true)
		}
	}

	followNode = func(node *Node, remaining float64, backwards bool) {
		if params.SeenNodes != nil {
			params.SeenNodes[node.ID] = true
		}
		var edges []*Edge
		if !backwards {
			edges = node.Out
		} else {
			edges = node.In
		}
		for _, edge := range edges {
			if seenNodePairs[[2]int{edge.Src.ID, edge.Dst.ID}] || seenNodePairs[[2]int{edge.Dst.ID, edge.Src.ID}] {
				continue
			}
			if !backwards {
				followForwards(EdgePos{
					edge,
					0,
				}, remaining)
			} else {
				followBackwards(EdgePos{
					edge,
					edge.Segment().Length(),
				}, remaining)
			}
		}
	}

	if len(params.SourceNodes) > 0 {
		for _, node := range params.SourceNodes {
			if !params.NoForwards {
				followNode(node, params.Distance, false)
			}
			if params.Backwards {
				followNode(node, params.Distance, true)
			}
		}
	} else {
		if !params.NoForwards {
			followForwards(params.SourcePos, params.Distance)
		}
		if params.Backwards {
			followBackwards(params.SourcePos, params.Distance)
		}
	}

	return positions
}
