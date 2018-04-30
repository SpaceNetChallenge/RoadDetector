package main

import (
	"./common"

	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

func main() {
	paths := os.Args[1:]
	for _, trainpath := range paths {
		if trainpath[len(trainpath) - 1] == '/' {
			trainpath = trainpath[:len(trainpath) - 1]
		}
		parts := strings.Split(trainpath, "/")
		d := parts[len(parts) - 1]
		dparts := strings.Split(d, "_")
		city := fmt.Sprintf("%s_%s_%s", dparts[0], dparts[1], dparts[2])

		graphs := make(map[string]*common.Graph)
		vertices := make(map[[2]string]*common.Node)

		bytes, err := ioutil.ReadFile(fmt.Sprintf("%s/summaryData/%s.csv", trainpath, d))
		if err != nil {
			panic(err)
		}
		for _, line := range strings.Split(string(bytes), "\n") {
			line = strings.TrimSpace(line)
			if !strings.Contains(line, city) {
				continue
			}
			parts := strings.SplitN(line, ",", 2)
			id := strings.Split(parts[0], "img")[1]

			if graphs[id] == nil {
				graphs[id] = &common.Graph{}
			}

			if strings.Contains(line, "LINESTRING EMPTY") {
				continue
			}

			pointsStr := strings.Split(strings.Split(strings.Split(parts[1], "(")[1], ")")[0], ", ")

			for _, pointStr := range pointsStr {
				if vertices[[2]string{id, pointStr}] == nil {
					pointParts := strings.Split(pointStr, " ")
					x, _ := strconv.ParseFloat(pointParts[0], 64)
					y, _ := strconv.ParseFloat(pointParts[1], 64)
					vertices[[2]string{id, pointStr}] = graphs[id].AddNode(common.Point{x, y})
				}
			}

			for i := 0; i < len(pointsStr) - 1; i++ {
				prev := pointsStr[i]
				next := pointsStr[i + 1]
				graphs[id].AddBidirectionalEdge(vertices[[2]string{id, prev}], vertices[[2]string{id, next}])
			}
		}

		for id, graph := range graphs {
			if err := graph.Write(fmt.Sprintf("/wdata/spacenet2017/favyen/graphs/%s.%s.%s.graph", d, city, id)); err != nil {
				panic(err)
			}
		}
	}
}
