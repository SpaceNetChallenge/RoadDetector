package common

import (
	"math"
)

func KDE(traces Traces, cellSize float64, sigma float64) [][]float64 {
	// compute histogram
	rect := traces.Bounds()
	numX := int((rect.Max.X - rect.Min.X) / cellSize + 1)
	numY := int((rect.Max.Y - rect.Min.Y) / cellSize + 1)
	histogram := make([][]int, numX)
	for i := range histogram {
		histogram[i] = make([]int, numY)
	}

	getHistogramIndices := func(p Point) (int, int) {
		return int((p.X - rect.Min.X) / cellSize), int((p.Y - rect.Min.Y) / cellSize)
	}

	for _, trace := range traces {
		var previousObs *Observation
		for _, obs := range trace.Observations {
			if previousObs != nil {
				startX, startY := getHistogramIndices(previousObs.Point)
				endX, endY := getHistogramIndices(obs.Point)
				cells := DrawLineOnCells(startX, startY, endX, endY, numX, numY)
				for _, cellidx := range cells {
					histogram[cellidx[0]][cellidx[1]]++
				}
			}
			previousObs = obs
		}
	}

	// create Gaussian kernel
	kernelRadius := int((2 * sigma) / cellSize + 1)
	kernelCells := 2 * kernelRadius + 1
	kernel := make([][]float64, kernelCells)
	for i := range kernel {
		kernel[i] = make([]float64, kernelCells)
		for j := range kernel[i] {
			dsq := (i - kernelRadius) * (i - kernelRadius) + (j - kernelRadius) * (j - kernelRadius)
			kernel[i][j] = math.Exp(float64(-dsq) / (2 * sigma * sigma)) / (math.Pi * 2 * sigma * sigma)
		}
	}

	// apply kernel
	out := make([][]float64, numX)
	for i := range out {
		out[i] = make([]float64, numY)
		for j := range out[i] {
			for dx := -kernelRadius; dx <= kernelRadius; dx++ {
				for dy := -kernelRadius; dy <= kernelRadius; dy++ {
					if i + dx >= 0 && i + dx < numX && j + dy >= 0 && j + dy < numY {
						out[i][j] += kernel[dx + kernelRadius][dy + kernelRadius] * float64(histogram[i + dx][j + dy])
					}
				}
			}
		}
	}

	return out
}
