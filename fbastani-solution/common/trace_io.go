package common

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
)

func LoadCartelTraces(tracePath string) (Traces, error) {
	file, err := os.Open(tracePath)
	if err != nil {
		return nil, fmt.Errorf("error opening %s: %v", tracePath, err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)

	type ActiveTrace struct {
		Trace *Trace
		Address string
	}

	var traces Traces
	var activeTrace ActiveTrace

	parseLine := func(line string) (address string, t time.Time, longitude float64, latitude float64, err error) {
		parts := strings.Split(line, ",")
		if len(parts) != 6 {
			err = fmt.Errorf("expected 6 comma-separated parts, but got %d", len(parts))
			return
		}
		address = parts[1]
		timeInt, perr := strconv.ParseInt(parts[0], 10, 64)
		if perr != nil {
			err = fmt.Errorf("bad time %s: %v", parts[0], perr)
			return
		}
		t = time.Unix(timeInt, 0)
		longitude, perr = strconv.ParseFloat(parts[3], 64)
		if perr != nil {
			err = fmt.Errorf("bad longitude %s: %v", parts[3], perr)
			return
		}
		latitude, perr = strconv.ParseFloat(parts[2], 64)
		if perr != nil {
			err = fmt.Errorf("bad longitude %s: %v", parts[2], perr)
			return
		}
		return
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
		if line == "" {
			continue
		}
		address, t, longitude, latitude, err := parseLine(line)
		if err != nil {
			return nil, fmt.Errorf("invalid line %s: %v", line, err)
		}
		if activeTrace.Trace == nil || activeTrace.Address != address || t.Sub(activeTrace.Trace.LastObservation().Time) > time.Minute {
			activeTrace.Trace = new(Trace)
			activeTrace.Address = address
			traces = append(traces, activeTrace.Trace)
		}
		activeTrace.Trace.Observations = append(activeTrace.Trace.Observations, &Observation{
			Time: t,
			Point: Point{
				X: longitude,
				Y: latitude,
			},
		})
	}

	return traces, nil
}

type CMTOptions struct {
	SetMetadata bool
	Limit int
}

func LoadCMTTraces(tracePath string, rect *Rectangle, options CMTOptions) (Traces, error) {
	file, err := os.Open(tracePath)
	if err != nil {
		return nil, fmt.Errorf("error opening %s: %v", tracePath, err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)

	var traces Traces
	var currentTrace *Trace
	var currentTripID int

	parseLine := func(line string) (tripID int, t time.Time, longitude float64, latitude float64, speed float64, heading float64, err error) {
		parts := strings.Split(line, ",")
		if len(parts) < 9 {
			err = fmt.Errorf("expected >=9 comma-separated parts, but got %d", len(parts))
			return
		}
		tripID, perr := strconv.Atoi(parts[8])
		if perr != nil {
			err = fmt.Errorf("bad trip ID %s: %v", parts[8], perr)
			return
		}
		timeInt, perr := strconv.ParseInt(parts[0], 10, 64)
		if perr != nil {
			err = fmt.Errorf("bad time %s: %v", parts[0], perr)
			return
		}
		t = time.Unix(timeInt, 0)
		longitude, perr = strconv.ParseFloat(parts[2], 64)
		if perr != nil {
			err = fmt.Errorf("bad longitude %s: %v", parts[2], perr)
			return
		}
		latitude, perr = strconv.ParseFloat(parts[1], 64)
		if perr != nil {
			err = fmt.Errorf("bad longitude %s: %v", parts[1], perr)
			return
		}
		speed, perr = strconv.ParseFloat(parts[3], 64)
		if perr != nil {
			err = fmt.Errorf("bad speed %s: %v", parts[3], perr)
			return
		}
		heading, perr = strconv.ParseFloat(parts[4], 64)
		if perr != nil {
			err = fmt.Errorf("bad heading %s: %v", parts[4], perr)
			return
		}
		return
	}

	count := 0
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
		if line == "" {
			continue
		}
		tripID, t, longitude, latitude, speed, heading, err := parseLine(line)
		if err != nil {
			return nil, fmt.Errorf("invalid line %s: %v", line, err)
		}
		count++
		if count % 100000 == 0 {
			fmt.Printf("count=%d ntraces=%d\n", count, len(traces))
		}
		point := Point{longitude, latitude}
		if rect != nil && !rect.Contains(point) {
			if currentTrace != nil {
				currentTrace = nil
			}
			continue
		}
		if currentTrace == nil || currentTripID != tripID {
			if options.Limit > 0 && len(traces) >= options.Limit {
				break
			}
			currentTrace = &Trace{Name: strconv.Itoa(tripID)}
			currentTripID = tripID
			traces = append(traces, currentTrace)
		}
		obs := Observation{
			Time: t,
			Point: point,
		}
		if options.SetMetadata {
			obs.Metadata = map[string]interface{}{
				"cmt_speed": speed,
				"cmt_heading": heading,
			}
		}
		currentTrace.Observations = append(currentTrace.Observations, &obs)
	}

	return traces, nil
}

/*func LoadTraces(tracePath string) ([]Trace, error) {
	var traces []Trace

	files, err := ioutil.ReadDir(tracePath)
	if err != nil {
		return nil, fmt.Errorf("error reading %s: %v", tracePath, err)
	}
	for _, fileInfo := range files {
		if fileInfo.IsDir() || !strings.HasSuffix(fileInfo.Name(), ".csv.gz") {
			continue
		}
		file, err := os.Open(path.Join(tracePath, fileInfo.Name()))
		if err != nil {
			return nil, fmt.Errorf("error reading %s: %v", fileInfo.Name(), err)
		}

	}
}*/

func LoadChicagoTraces(tracePath string) (Traces, error) {
	return loadTextTraces(tracePath, ",", 4, 2, 1, 3)
}

func LoadTraces(tracePath string) (Traces, error) {
	return loadTextTraces(tracePath, " ", 3, 0, 1, 2)
}

func loadTextTraces(tracePath string, delimiter string, expectedParts int, lonIdx int, latIdx int, tsIdx int) (Traces, error) {
	var traces []*Trace

	files, err := ioutil.ReadDir(tracePath)
	if err != nil {
		return nil, fmt.Errorf("error reading %s: %v", tracePath, err)
	}

	loadTrace := func(file *os.File, name string) error {
		defer file.Close()
		reader := bufio.NewReader(file)
		trace := Trace{Name: name}
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				} else {
					return err
				}
			}
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			parts := strings.Split(line, delimiter)
			if len(parts) < expectedParts {
				return fmt.Errorf("invalid line (bad parts): %s", line)
			}
			lon, lonerr := strconv.ParseFloat(parts[lonIdx], 64)
			lat, laterr := strconv.ParseFloat(parts[latIdx], 64)
			ts, tserr := strconv.ParseFloat(parts[tsIdx], 64)
			if lonerr != nil || laterr != nil || tserr != nil {
				return fmt.Errorf("invalid line (%v %v %v): %s", lonerr, laterr, tserr, line)
			}
			trace.Observations = append(trace.Observations, &Observation{
				Time: time.Unix(int64(ts), 0),
				Point: Point{lon, lat},
			})
		}
		traces = append(traces, &trace)
		return nil
	}

	for _, fileInfo := range files {
		if fileInfo.IsDir() || !strings.HasSuffix(fileInfo.Name(), ".txt") {
			continue
		}
		file, err := os.Open(path.Join(tracePath, fileInfo.Name()))
		if err != nil {
			return nil, fmt.Errorf("error reading %s: %v", fileInfo.Name(), err)
		}
		if err := loadTrace(file, strings.Split(fileInfo.Name(), ".txt")[0]); err != nil {
			return nil, err
		}
	}

	return traces, nil
}

func SaveTraces(tracePath string, traces Traces) error {
	saveTrace := func(fname string, trace *Trace) error {
		file, err := os.Create(fname)
		if err != nil {
			return err
		}
		defer file.Close()
		for _, obs := range trace.Observations {
			line := fmt.Sprintf("%f %f %d\n", obs.Point.X, obs.Point.Y, obs.Time.Unix())
			file.Write([]byte(line))
		}
		return nil
	}

	for i, trace := range traces {
		fname := path.Join(tracePath, fmt.Sprintf("%d.txt", i))
		if err := saveTrace(fname, trace); err != nil {
			return err
		}
	}

	return nil
}

func SaveChicagoTraces(tracePath string, traces Traces) error {
	saveTrace := func(fname string, trace *Trace) error {
		file, err := os.Create(fname)
		if err != nil {
			return err
		}
		defer file.Close()
		for i, obs := range trace.Observations {
			var prev, next string
			if i - 1 < 0 {
				prev = "None"
			} else {
				prev = strconv.Itoa(i - 1)
			}
			if i + 1 >= len(trace.Observations) {
				next = "None"
			} else {
				next = strconv.Itoa(i + 1)
			}
			line := fmt.Sprintf("%d,%f,%f,%d,0,%s,%s\n", i, obs.Point.Y, obs.Point.X, obs.Time.Unix(), prev, next)
			file.Write([]byte(line))
		}
		return nil
	}

	for i, trace := range traces {
		fname := path.Join(tracePath, fmt.Sprintf("trip_%d.txt", i))
		if err := saveTrace(fname, trace); err != nil {
			return err
		}
	}

	return nil
}

func SaveKharitaTraces(fname string, traces Traces) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()
	counter := 0
	for traceID, trace := range traces {
		for _, obs := range trace.Observations {
			/*
			var nextObs *Observation
			for _, futureObs := range trace.Observations[i+1:] {
				vector := futureObs.Point.LonLatToMeters(obs.Point).Sub(obs.Point.LonLatToMeters(obs.Point))
				if vector.Magnitude() < 15 {
					continue
				} else if vector.Magnitude() > 80 {
					break
				}
				nextObs = futureObs
				break
			}
			if nextObs == nil {
				continue
			}
			dtime := obs.Time.Sub((*nextObs).Time).Seconds()

			a := obs.Point.LonLatToMeters(obs.Point)
			b := (*nextObs).Point.LonLatToMeters(obs.Point)
			dspace := a.Distance(b)
			angle := -(Point{0, 1}.SignedAngle(b.Sub(a)))
			if angle < 0 {
				angle = 2 * math.Pi + angle
			}

			speed = dspace / dtime * 3.6
			heading = angle * 180 / math.Pi
			*/
			var heading, speed float64
			speed = 20

			/*if cmtSpeed := obs.Metadata["cmt_speed"]; cmtSpeed != nil {
				// convert m/s to km/h
				speed = cmtSpeed.(float64) / 3.6
			}*/
			if cmtHeading := obs.Metadata["cmt_heading"]; cmtHeading != nil {
				heading = cmtHeading.(float64)
			} else if posHeading := obs.Metadata["heading"]; posHeading != nil {
				heading = -(posHeading.(float64) * 180 / math.Pi)
				if heading < 0 {
					heading += 360
				}
			}
			if mSpeed := obs.Metadata["speed"]; mSpeed != nil {
				speed = (mSpeed.(float64)) * 3.6
			}

			/*
			line := fmt.Sprintf(
				"%d,%s+03,%f,%f,%f,%f\n",
				traceID,
				obs.Time.UTC().Format("2006-01-02 15:04:05"),
				obs.Point.X,
				obs.Point.Y,
				speed,
				heading,
			)
			*/
			line := fmt.Sprintf(
				"%f\t%f\t%d\t%d\t%d\t%f\t%s+03\t%f\t%f\t%f\n",
				obs.Point.X,
				obs.Point.Y,
				counter,
				counter,
				traceID,
				speed,
				obs.Time.UTC().Format("2006-01-02 15:04:05"),
				obs.Point.X,
				obs.Point.Y,
				heading,
			)
			file.Write([]byte(line))
			counter++
		}
	}
	return nil
}
