package common

// Use Bresenham's algorithm to get indices of cells to draw a line.
func DrawLineOnCells(startX int, startY int, endX int, endY int, maxX int, maxY int) [][2]int {
	abs := func(x int) int {
		if x >= 0 {
			return x
		} else {
			return -x
		}
	}

	// followX indicates whether to move along x or y coordinates
	followX := abs(endY - startY) <= abs(endX - startX)
	var x0, x1, y0, y1 int
	if followX {
		x0 = startX
		x1 = endX
		y0 = startY
		y1 = endY
	} else {
		x0 = startY
		x1 = endY
		y0 = startX
		y1 = endX
	}

	deltaX := abs(x1 - x0)
	deltaY := abs(y1 - y0)
	var currentError int = 0

	var xstep, ystep int
	if x0 < x1 {
		xstep = 1
	} else {
		xstep = -1
	}
	if y0 < y1 {
		ystep = 1
	} else {
		ystep = -1
	}

	points := make([][2]int, 0, deltaX + 1)
	addPoint := func(x int, y int) {
		if x >= 0 && x < maxX && y >= 0 && y < maxY {
			points = append(points, [2]int{x, y})
		}
	}

	x := x0
	y := y0

	for x != x1 + xstep {
		if followX {
			addPoint(x, y)
		} else {
			addPoint(y, x)
		}

		x += xstep
		currentError += deltaY
		if currentError >= deltaX {
			y += ystep
			currentError -= deltaX
		}
	}

	return points
}
