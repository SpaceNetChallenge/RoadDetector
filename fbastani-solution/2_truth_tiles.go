package main

import (
	"./common"

	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"fmt"
	"math"
	"os"
	"strings"
)

func main() {
	fmt.Println("initializing tasks")
	type Task struct {
		Label string
		Graph *common.Graph
	}
	var tasks []Task

	graphDir := "/wdata/spacenet2017/favyen/graphs/"
	files, err := ioutil.ReadDir(graphDir)
	if err != nil {
		panic(err)
	}
	for _, file := range files {
		if !strings.HasSuffix(file.Name(), ".graph") {
			continue
		}
		graph, err := common.ReadGraph(graphDir + file.Name())
		if err != nil {
			panic(err)
		}
		tasks = append(tasks, Task{
			Label: strings.Split(file.Name(), ".graph")[0],
			Graph: graph,
		})
	}

	processTask := func(task Task, threadID int) {
		values := make([][]uint8, 650)
		for i := range values {
			values[i] = make([]uint8, 650)
		}
		for _, edge := range task.Graph.Edges {
			segment := edge.Segment()
			for _, pos := range common.DrawLineOnCells(int(segment.Start.X)/2, int(segment.Start.Y)/2, int(segment.End.X)/2, int(segment.End.Y)/2, 650, 650) {
				for i := -4; i <= 4; i++ {
					for j := -4; j <= 4; j++ {
						d := math.Sqrt(float64(i * i + j * j))
						if d > 4 {
							continue
						}
						x := pos[0] + i
						y := pos[1] + j
						if x >= 0 && x < 650 && y >= 0 && y < 650 {
							values[x][y] = 255
						}
					}
				}
			}
		}

		img := image.NewGray(image.Rect(0, 0, 650, 650))
		for i := 0; i < 650; i++ {
			for j := 0; j < 650; j++ {
				img.SetGray(i, j, color.Gray{values[i][j]})
			}
		}

		f, err := os.Create(fmt.Sprintf("/wdata/spacenet2017/favyen/truth/%s.png", task.Label))
		if err != nil {
			panic(err)
		}
		if err := png.Encode(f, img); err != nil {
			panic(err)
		}
		f.Close()
	}

	fmt.Println("launching workers")
	n := 8
	taskCh := make(chan Task)
	doneCh := make(chan bool)
	for threadID := 0; threadID < n; threadID++ {
		go func(threadID int) {
			for task := range taskCh {
				processTask(task, threadID)
			}
			doneCh <- true
		}(threadID)
	}
	fmt.Println("running tasks")
	for i, task := range tasks {
		if i % 10 == 0 {
			fmt.Printf("... task progress: %d/%d\n", i, len(tasks))
		}
		taskCh <- task
	}
	close(taskCh)
	for threadID := 0; threadID < n; threadID++ {
		<- doneCh
	}
}
