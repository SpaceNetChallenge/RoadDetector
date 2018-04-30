package common

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"

	"github.com/qedus/osmpbf"
)

var HIGHWAY_BLACKLIST []string = []string{
	"pedestrian",
	"footway",
	"bridleway",
	"steps",
	"path",
	"sidewalk",
	"cycleway",
	"proposed",
	"construction",
	"bus_stop",
	"crossing",
	"elevator",
	"emergency_access_point",
	"escape",
	"give_way",
}


func isBlacklisted(highway string) bool {
	for _, x := range HIGHWAY_BLACKLIST {
		if highway == x {
			return true
		}
	}
	return false
}

func LoadOSM(path string, bounds Rectangle) (*Graph, error) {
	graphs, err := LoadOSMMultiple(path, []Rectangle{bounds}, OSMOptions{})
	if err != nil {
		return nil, err
	} else {
		return graphs[0], nil
	}
}

type OSMOptions struct {
	Verbose bool
	EdgeWidths []map[int]float64
	NoParking bool
	NoTunnels bool
	LayerEdges []map[int]bool
	EdgeTags []map[int]map[string]string
	OneWay bool
	OnlyMotorways bool
	MotorwayEdges []map[int]bool
}

func LoadOSMMultiple(path string, regions []Rectangle, options OSMOptions) ([]*Graph, error) {
	graphs := make([]*Graph, len(regions))
	for i := range graphs {
		graphs[i] = &Graph{}
	}
	vertexIDMaps := make([]map[int64]*Node, len(regions))
	for i := range vertexIDMaps {
		vertexIDMaps[i] = make(map[int64]*Node)
	}
	vertexRegionMap := make(map[int64][]int)

	// do two passes through the OSM data:
	//  1) collect vertices in the bounds
	//  2) collect edges from the OSM ways
	process := func(f func(v interface{})) error {
		file, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("error opening %s: %v", path, err)
		}
		defer file.Close()

		d := osmpbf.NewDecoder(file)
		d.SetBufferSize(osmpbf.MaxBlobSize)
		d.Start(runtime.GOMAXPROCS(-1))
		for {
			if v, err := d.Decode(); err == io.EOF {
				break
			} else if err != nil {
				return fmt.Errorf("decode error: %v", err)
			} else {
				f(v)
			}
		}
		return nil
	}

	var count int64 = 0
	err := process(func(v interface{}) {
		switch v := v.(type) {
		case *osmpbf.Node:
			point := Point{v.Lon, v.Lat}
			for i := range regions {
				if regions[i].Contains(point) {
					vertexIDMaps[i][v.ID] = graphs[i].AddNode(point)
					vertexRegionMap[v.ID] = append(vertexRegionMap[v.ID], i)
				}
			}
			count++
			if options.Verbose && count % 10000000 == 0 {
				fmt.Printf("finished %dM vertices\n", count / 1000000)
			}
		}
	})
	if err != nil {
		return nil, err
	}

	count = 0
	err = process(func(v interface{}) {
		switch v := v.(type) {
		case *osmpbf.Way:
			highway, ok := v.Tags["highway"]
			if !ok || isBlacklisted(highway) {
				return
			} else if len(v.NodeIDs) < 2 {
				return
			}

			if options.NoParking {
				if v.Tags["amenity"] == "parking" || v.Tags["service"] == "parking_aisle" {
					return
				} else if v.Tags["service"] == "driveway" {
					return
				}
			}
			if options.NoTunnels {
				if len(v.Tags["layer"]) >= 2 && v.Tags["layer"][0] == '-' {
					return
				}
			}
			isMotorway := v.Tags["highway"] == "motorway" || v.Tags["highway"] == "trunk"
			if options.OnlyMotorways && !isMotorway {
				return
			}

			// determine oneway, 0 for no, 1 for forward, -1 for reverse
			oneway := 0
			if options.OneWay {
				// (1) if oneway tag is set, use that exclusively
				//     (note that this overrides (2) since some ways can have motorway but
				//      use tag set to "no" to disable oneway)
				// (2) based on other tags that are default oneway
				if v.Tags["oneway"] != "" {
					if v.Tags["oneway"] == "yes" || v.Tags["oneway"] == "1" {
						oneway = 1
					} else if v.Tags["oneway"] == "-1" {
						oneway = -1
					}
				} else if v.Tags["highway"] == "motorway" || v.Tags["junction"] == "roundabout" {
					oneway = 1
				}
			}

			type RegionEdge struct {
				Edge *Edge
				RegionID int
			}

			var wayEdges []RegionEdge
			var lastVertexID int64 = v.NodeIDs[0]
			for _, vertexID := range v.NodeIDs[1:] {
				for _, regionID := range vertexRegionMap[vertexID] {
					node1 := vertexIDMaps[regionID][lastVertexID]
					node2 := vertexIDMaps[regionID][vertexID]
					if node1 != nil && node2 != nil {
						if oneway == 0 {
							edge := graphs[regionID].AddBidirectionalEdge(node1, node2)
							wayEdges = append(
								wayEdges,
								RegionEdge{edge[0], regionID},
								RegionEdge{edge[1], regionID},
							)
						} else if oneway == 1 {
							edge := graphs[regionID].AddEdge(node1, node2)
							wayEdges = append(wayEdges, RegionEdge{edge, regionID})
						} else if oneway == -1 {
							edge := graphs[regionID].AddEdge(node2, node1)
							wayEdges = append(wayEdges, RegionEdge{edge, regionID})
						} else {
							panic(fmt.Errorf("invalid oneway %d", oneway))
						}
					}
				}
				lastVertexID = vertexID
			}

			if len(options.EdgeWidths) > 0 {
				var width float64
				if val, ok := v.Tags["lanes"]; ok {
					lanes, _ := strconv.ParseFloat(strings.Split(val, ";")[0], 64)
					if lanes == 1 {
						width = 6.6
					} else {
						width = lanes * 3.7
					}
				} else if val, ok := v.Tags["width"]; ok {
					width, _ = strconv.ParseFloat(strings.Fields(strings.Split(val, ";")[0])[0], 64)
				} else {
					width = 6.6
				}
				for _, redge := range wayEdges {
					options.EdgeWidths[redge.RegionID][redge.Edge.ID] = width
				}
			}

			if len(options.LayerEdges) > 0 && v.Tags["layer"] != "" {
				for _, redge := range wayEdges {
					options.LayerEdges[redge.RegionID][redge.Edge.ID] = true
				}
			}

			if len(options.EdgeTags) > 0 {
				for _, redge := range wayEdges {
					options.EdgeTags[redge.RegionID][redge.Edge.ID] = v.Tags
				}
			}

			if len(options.MotorwayEdges) > 0 && isMotorway {
				for _, redge := range wayEdges {
					options.MotorwayEdges[redge.RegionID][redge.Edge.ID] = true
				}
			}

			count++
			if options.Verbose && count % 100000 == 0 {
				fmt.Printf("finished %dK ways\n", count / 1000)
			}
		}
	})
	if err != nil {
		return nil, err
	}

	return graphs, nil
}
