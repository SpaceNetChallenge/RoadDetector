package common

import (
	"fmt"
	"math"
	"time"
)

const VITERBI_NORMALIZE_DELTA time.Duration = time.Second
const VITERBI_SIGMA = 25

func NormalizeTraces(traces []*Trace) {
	for _, trace := range traces {
		previous := trace.Observations[0]
		newObs := []*Observation{previous}
		for _, obs := range trace.Observations[1:] {
			dt := obs.Time.Sub(previous.Time)
			n := int(dt / VITERBI_NORMALIZE_DELTA) + 1
			for i := 1; i < n; i++ {
				intermediateTime := previous.Time.Add(dt * time.Duration(i) / time.Duration(n))
				intermediatePoint := previous.Point.Add(obs.Point.Sub(previous.Point).Scale(float64(i) / float64(n)))
				newObs = append(newObs, &Observation{
					Time: intermediateTime,
					Point: intermediatePoint,
				})
			}
			newObs = append(newObs, obs)
			previous = obs
		}
		trace.Observations = newObs
	}
}

func Viterbi(graph *Graph, traces []*Trace, tolerance float64) {
	transitionProbs := make([]map[int]float64, len(graph.Edges))
	for _, edge := range graph.Edges {
		probs := make(map[int]float64)
		transitionProbs[edge.ID] = probs

		var adjacentEdges []*Edge
		for _, other := range edge.Dst.Out {
			adjacentEdges = append(adjacentEdges, other)
		}

		// set scores and then reweight so that sum is 1
		probs[edge.ID] = 30
		var totalScore float64 = 30
		for _, other := range adjacentEdges {
			negAngle := math.Pi / 2 - edge.AngleTo(other)
			if negAngle < 0 {
				negAngle = 0
			}
			score := negAngle * negAngle + 0.05
			totalScore += score
			probs[other.ID] = score
		}
		for id := range probs {
			probs[id] /= totalScore
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
			score := math.Exp(-0.5 * distance * distance / VITERBI_SIGMA / VITERBI_SIGMA)
			scores[edge.ID] = score
			totalScore += score
		}
		for i := range scores {
			scores[i] /= totalScore
		}
		return scores
	}

	for _, trace := range traces {
		// run viterbi
		probs := make(map[int]float64)
		for _, edge := range rtree.Search(trace.Observations[0].Point.RectangleTol(tolerance)) {
			probs[edge.ID] = 0
		}
		backpointers := make([]map[int]int, len(trace.Observations))
		failed := false
		for i := 1; i < len(trace.Observations); i++ {
			obs := trace.Observations[i]
			var nextProbs map[int]float64

			for factor := float64(1); len(nextProbs) < 2 && factor <= 4; factor *= 2 {
				backpointers[i] = make(map[int]int)
				emissions := emissionProbs(obs.Point, tolerance * factor)
				if factor > 1 {
					fmt.Printf("viterbi: warning: factor=%f at i=%d, point=%v\n", factor, i, obs.Point)
				}
				nextProbs = make(map[int]float64)
				for prevEdgeID := range probs {
					transitions := transitionProbs[prevEdgeID]
					for nextEdgeID := range transitions {
						if emissions[nextEdgeID] == 0 {
							continue
						}
						prob := probs[prevEdgeID] + math.Log(transitions[nextEdgeID]) + math.Log(emissions[nextEdgeID])
						if curProb, ok := nextProbs[nextEdgeID]; !ok || prob > curProb {
							nextProbs[nextEdgeID] = prob
							backpointers[i][nextEdgeID] = prevEdgeID
						}
					}
				}
			}
			probs = nextProbs
			//fmt.Printf("%d/%d %v %v\n", i, len(trace.Observations), obs.Point, probs)
			if len(probs) == 0 {
				fmt.Printf("viterbi: warning: failed to find edge, skipping trace: i=%d, point=%v\n", i, obs.Point)
				failed = true
				break
			}
		}
		if failed {
			continue
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
			edge := graph.Edges[curEdge]
			position := edge.Segment().Project(trace.Observations[i].Point, false)
			trace.Observations[i].SetMetadata("viterbi", EdgePos{edge, position})
			if i > 0 {
				curEdge = backpointers[i][curEdge]
			}
		}
	}
}
