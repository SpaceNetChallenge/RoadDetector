package common

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func ReadChicagoMap(fname string) (*Graph, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	m := make(map[string]*Node)
	graph := &Graph{}

	addVertex := func(line string) bool {
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			return false
		}
		lat, errlat := strconv.ParseFloat(parts[0], 64)
		lon, errlon := strconv.ParseFloat(parts[1], 64)
		if errlat != nil || errlon != nil {
			return false
		}
		m[line] = graph.AddNode(Point{lon, lat})
		return true
	}

	for {
		line1, _ := reader.ReadString('\n')
		line2, _ := reader.ReadString('\n')
		_, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return nil, err
			}
		}
		line1 = strings.TrimSpace(line1)
		line2 = strings.TrimSpace(line2)
		if _, ok := m[line1]; !ok {
			if ok := addVertex(line1); !ok {
				return nil, fmt.Errorf("invalid line: %s", line1)
			}
		}
		if _, ok := m[line2]; !ok {
			if ok := addVertex(line2); !ok {
				return nil, fmt.Errorf("invalid line: %s", line2)
			}
		}
		graph.AddEdge(m[line1], m[line2])
	}

	return graph, nil
}

func ReadDaviesMap(verticesFname string, edgesFname string) (*Graph, error) {
	graph := &Graph{}
	vertexIDMap := make(map[string]*Node)

	readVertices := func() error {
		file, err := os.Open(verticesFname)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				} else {
					return err
				}
			}
			parts := strings.Split(line, " ")
			if len(parts) != 3 {
				return fmt.Errorf("invalid line: %s", line)
			}
			lat, errlat := strconv.ParseFloat(parts[1], 64)
			lon, errlon := strconv.ParseFloat(parts[2], 64)
			if errlat != nil || errlon != nil {
				return fmt.Errorf("invalid line: %s", line)
			}
			vertexIDMap[parts[0]] = graph.AddNode(Point{lon, lat})
		}
		return nil
	}

	readEdges := func() error {
		file, err := os.Open(edgesFname)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				} else {
					return err
				}
			}
			parts := strings.Split(line, " ")
			if len(parts) != 2 {
				return fmt.Errorf("invalid line: %s", line)
			} else if vertexIDMap[parts[0]] == nil || vertexIDMap[parts[1]] == nil {
				return fmt.Errorf("line has unknown node IDs: %s", line)
			}
			graph.AddBidirectionalEdge(vertexIDMap[parts[0]], vertexIDMap[parts[1]])
		}
		return nil
	}

	if err := readVertices(); err != nil {
		return nil, err
	}
	if err := readEdges(); err != nil {
		return nil, err
	}
	return graph, nil
}

func ReadAhmedMap(verticesFname string, edgesFname string) (*Graph, error) {
	graph := &Graph{}
	vertexIDMap := make(map[string]*Node)

	readVertices := func() error {
		file, err := os.Open(verticesFname)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				} else {
					return err
				}
			}
			parts := strings.Split(strings.TrimSpace(line), ",")
			if len(parts) != 4 {
				return fmt.Errorf("invalid line: %s", line)
			}
			x, errx := strconv.ParseFloat(parts[1], 64)
			y, erry := strconv.ParseFloat(parts[2], 64)
			if errx != nil || erry != nil {
				return fmt.Errorf("invalid line: %s", line)
			}
			vertexIDMap[parts[0]] = graph.AddNode(Point{x, y})
		}
		return nil
	}

	readEdges := func() error {
		file, err := os.Open(edgesFname)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				} else {
					return err
				}
			}
			parts := strings.Split(strings.TrimSpace(line), ",")
			if len(parts) != 3 {
				return fmt.Errorf("invalid line: %s", line)
			} else if vertexIDMap[parts[1]] == nil || vertexIDMap[parts[2]] == nil {
				return fmt.Errorf("line has unknown node IDs: %s", line)
			}
			graph.AddEdge(vertexIDMap[parts[1]], vertexIDMap[parts[2]])
		}
		return nil
	}

	if err := readVertices(); err != nil {
		return nil, err
	}
	if err := readEdges(); err != nil {
		return nil, err
	}
	return graph, nil
}

func ReadKharitaMap(fname string) (*Graph, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	m := make(map[[2]string]*Node)
	graph := &Graph{}

	getOrCreateVertex := func(lonstr string, latstr string) *Node {
		k := [2]string{lonstr, latstr}
		if m[k] != nil {
			return m[k]
		}
		lon, errlon := strconv.ParseFloat(lonstr, 64)
		lat, errlat := strconv.ParseFloat(latstr, 64)
		if errlat != nil || errlon != nil {
			return nil
		}
		node := graph.AddNode(Point{lon, lat})
		m[k] = node
		return node
	}

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
		parts := strings.Split(line, " ")
		node1 := getOrCreateVertex(parts[0], parts[1])
		node2 := getOrCreateVertex(parts[3], parts[4])
		if node1 == nil || node2 == nil {
			return nil, fmt.Errorf("invalid line: %s", line)
		}
		graph.AddEdge(node1, node2)
	}

	return graph, nil
}
