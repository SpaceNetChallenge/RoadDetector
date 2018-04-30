package common

import (
	"time"
)

type Observation struct {
	Time time.Time
	Point Point
	Metadata map[string]interface{}
}

func (obs *Observation) SetMetadata(k string, val interface{}) {
	if obs.Metadata == nil {
		obs.Metadata = make(map[string]interface{})
	}
	obs.Metadata[k] = val
}

func (obs *Observation) GetMetadata(k string) interface{} {
	if obs.Metadata == nil {
		return nil
	} else {
		return obs.Metadata[k]
	}
}

type Trace struct {
	Name string
	Observations []*Observation
}

func (trace *Trace) LastObservation() *Observation {
	if len(trace.Observations) > 0 {
		return trace.Observations[len(trace.Observations) - 1]
	} else {
		return nil
	}
}

type Traces []*Trace

// Convert coordinate system from longitude/latitude to Cartesian meters.
// This assumes that the GPS sequences cover a small region so that curvature can be ignored.
func (traces Traces) LonLatToMeters(origin Point) {
	for _, trace := range traces {
		for i := range trace.Observations {
			trace.Observations[i].Point = trace.Observations[i].Point.LonLatToMeters(origin)
		}
	}
}

func (traces Traces) Bounds() Rectangle {
	r := EmptyRectangle
	for _, trace := range traces {
		for _, obs := range trace.Observations {
			r = r.Extend(obs.Point)
		}
	}
	return r
}
