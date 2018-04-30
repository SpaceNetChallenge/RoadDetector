package common

import (
	"fmt"
	"math"
)

// This is an improved version of Viterbi map matching, where we handle sparse traces by
//  applying multiple transitions based on the distance between observations. For example,
//  if two consecutive samples are k*VITERBI2_GRANULARITY apart, then we will apply (k-1)
//  transitions prior to a transition+emission pair.
// Additionally, Viterbi2 takes an edgeWeights map so that some edges being more likely
//  than other edges can be taken into account in the model. If edgeWeights is nil, then
//  the weights are determined based on the angle between the segments, similar to the
//  process in the old Viterbi function.

const VITERBI2_GRANULARITY = 50
const VITERBI2_SIGMA = 30
const VITERBI2_START_TOLERANCE = 100

// Map match each trace in traces to the road network specified by graph.
// edgeWeights: a map from edge IDs to weight of the ID. If nil, the transition probabilities
//  are weighted based on the angle difference between the source and destination edges.
// hitsOnly: only compute edgeHits, do not store the map-matched data.
// Returns edgeHits, a map from edge ID to the number of times the edge is passed by a trace.
func Viterbi2(traces []*Trace, graph *Graph, edgeWeights map[int]float64, hitsOnly bool) (edgeHits map[int]int) {
	// precompute transition probabilities
	transitionProbs := make([]map[int]float64, len(graph.Edges))
	for _, edge := range graph.Edges {
		probs := make(map[int]float64)
		transitionProbs[edge.ID] = probs

		var adjacentEdges []*Edge
		for _, other := range edge.Dst.Out {
			adjacentEdges = append(adjacentEdges, other)
		}

		// on all edges there is 0.5 self loop
		probs[edge.ID] = 0.5
		var totalProb float64 = 0.5

		// compute weights to adjacent edges if needed
		weights := edgeWeights
		if weights == nil {
			weights = make(map[int]float64)
			for _, other := range adjacentEdges {
				negAngle := math.Pi / 2 - edge.AngleTo(other)
				if negAngle < 0 {
					negAngle = 0
				}
				weights[other.ID] = negAngle * negAngle + 0.05
			}
		}

		// extract probabilities from the weights
		// we force the average probability to be at most 0.05
		// any additional probability mass is discarded (essentially it is directed to an
		//  impossible state)
		var totalWeight float64 = 0
		for _, other := range adjacentEdges {
			totalWeight += weights[other.ID]
		}
		averageWeight := totalWeight / float64(len(adjacentEdges))
		averageProb := 0.05
		if averageProb * float64(len(adjacentEdges)) + totalProb > 0.9 {
			averageProb = (0.9 - totalProb) / float64(len(adjacentEdges))
		}

		for _, other := range adjacentEdges {
			prob := averageProb * weights[other.ID] / averageWeight
			probs[other.ID] = prob
			totalProb += prob
		}
	}

	rtree := graph.Rtree()

	// get conditional emission probabilities
	emissionProbs := func(point Point, tolerance float64) map[int]float64 {
		candidates := rtree.Search(point.RectangleTol(tolerance))
		if len(candidates) == 0 {
			return nil
		}
		scores := make(map[int]float64)
		var totalScore float64 = 0
		for _, edge := range candidates {
			distance := edge.Segment().Distance(point)
			score := math.Exp(-0.5 * distance * distance / VITERBI2_SIGMA / VITERBI2_SIGMA)
			scores[edge.ID] = score
			totalScore += score
		}
		for i := range scores {
			scores[i] /= totalScore
		}
		return scores
	}

	// match a single trace
	matchTrace := func(trace *Trace, edgeHits map[int]int) {
		if len(trace.Observations) < 5 {
			fmt.Printf("viterbi: warning: too few observations, skipping trace (%d)", len(trace.Observations))
			return
		}
		// initial probability is uniform across candidates
		probs := make(map[int]float64)
		for _, edge := range rtree.Search(trace.Observations[0].Point.RectangleTol(VITERBI2_START_TOLERANCE)) {
			probs[edge.ID] = 0
		}
		backpointers := make([][]map[int]int, len(trace.Observations))
		for i := 1; i < len(trace.Observations); i++ {
			obs := trace.Observations[i]

			// apply extra transitions in case the vehicle traveled a large distance from the
			//  previous observation
			distance := obs.Point.Distance(trace.Observations[i - 1].Point)
			for distance > VITERBI2_GRANULARITY {
				nextProbs := make(map[int]float64)
				nextBackpointers := make(map[int]int)
				for prevEdgeID := range probs {
					transitions := transitionProbs[prevEdgeID]
					for nextEdgeID := range transitions {
						prob := probs[prevEdgeID] + math.Log(transitions[nextEdgeID])
						if curProb, ok := nextProbs[nextEdgeID]; !ok || prob > curProb {
							nextProbs[nextEdgeID] = prob
							nextBackpointers[nextEdgeID] = prevEdgeID
						}
					}
				}
				backpointers[i] = append(backpointers[i], nextBackpointers)
				probs = nextProbs
				distance -= VITERBI2_GRANULARITY
			}

			var nextProbs map[int]float64
			var nextBackpointers map[int]int

			// find the most likely to match the emission+transition
			// we use an increasing factor in case there are no edges within a reasonable distance
			//  from the observed point
			for factor := float64(1); len(nextProbs) < 2 && factor <= 4; factor *= 2 {
				nextProbs = make(map[int]float64)
				nextBackpointers = make(map[int]int)
				emissions := emissionProbs(obs.Point, VITERBI2_START_TOLERANCE * factor)
				if factor > 1 {
					//fmt.Printf("viterbi: warning: factor=%f at i=%d, point=%v\n", factor, i, obs.Point)
				}
				for prevEdgeID := range probs {
					transitions := transitionProbs[prevEdgeID]
					for nextEdgeID := range transitions {
						if emissions[nextEdgeID] == 0 {
							continue
						}
						prob := probs[prevEdgeID] + math.Log(transitions[nextEdgeID]) + math.Log(emissions[nextEdgeID])
						if curProb, ok := nextProbs[nextEdgeID]; !ok || prob > curProb {
							nextProbs[nextEdgeID] = prob
							nextBackpointers[nextEdgeID] = prevEdgeID
						}
					}
				}
			}
			backpointers[i] = append(backpointers[i], nextBackpointers)
			if len(nextProbs) == 0 {
				fmt.Printf("viterbi: warning: failed to find edge, skipping trace: i=%d, point=%v\n", i, obs.Point)
				return
			}
			probs = nextProbs
		}

		// collect state sequence and annotate trace with map matched data
		var bestEdgeID *int
		for edgeID := range probs {
			if bestEdgeID == nil || probs[edgeID] > probs[*bestEdgeID] {
				bestEdgeID = &edgeID
			}
		}
		curEdge := *bestEdgeID
		for i := len(trace.Observations) - 1; i >= 0; i-- {
			if !hitsOnly {
				edge := graph.Edges[curEdge]
				position := edge.Segment().Project(trace.Observations[i].Point, false)
				trace.Observations[i].SetMetadata("viterbi", EdgePos{edge, position})
			}

			for j := len(backpointers[i]) - 1; j >= 0; j-- {
				prevEdge := backpointers[i][j][curEdge]
				if prevEdge != curEdge {
					edgeHits[curEdge]++
					curEdge = prevEdge
				}
			}
		}
	}

	traceCh := make(chan *Trace)
	n := 48
	doneCh := make(chan map[int]int)
	for i := 0; i < n; i++ {
		go func() {
			edgeHits := make(map[int]int)
			for trace := range traceCh {
				matchTrace(trace, edgeHits)
			}
			doneCh <- edgeHits
		}()
	}
	for traceIdx, trace := range traces {
		if traceIdx % 100 == 0 {
			fmt.Printf("progress: %d/%d\n", traceIdx, len(traces))
		}
		traceCh <- trace
	}
	close(traceCh)
	edgeHits = make(map[int]int)
	for i := 0; i < n; i++ {
		threadEdgeHits := <- doneCh
		for edgeID, hits := range threadEdgeHits {
			edgeHits[edgeID] += hits
		}
	}
	return
}
