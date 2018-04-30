package common

import (
	"math"
)

type Boundable interface {
	Bounds() Rectangle
}

type Point struct {
	X float64
	Y float64
}

func (point Point) LonLatToMeters(origin Point) Point {
	return Point{
		X: 111111 * math.Cos(origin.Y * math.Pi / 180) * (point.X - origin.X),
		Y: 111111 * (point.Y - origin.Y),
	}
}

// Converts from meters back to longitude/latitude
// origin should be the same point passed to LonLatToMeters (origin should be longitude/latitude)
func (point Point) MetersToLonLat(origin Point) Point {
	return Point{
		X: point.X / 111111 / math.Cos(origin.Y * math.Pi / 180) + origin.X,
		Y: point.Y / 111111 + origin.Y,
	}
}

func (point Point) Rectangle() Rectangle {
	return point.RectangleTol(0)
}

func (point Point) Bounds() Rectangle {
	return point.Rectangle()
}

func (point Point) RectangleTol(tol float64) Rectangle {
	t := Point{tol, tol}
	return Rectangle{
		Min: point.Sub(t),
		Max: point.Add(t),
	}
}

func (point Point) Dot(other Point) float64 {
	return point.X * other.X + point.Y * other.Y
}

func (point Point) Magnitude() float64 {
	return math.Sqrt(point.X * point.X + point.Y * point.Y)
}

func (point Point) Distance(other Point) float64 {
	dx := point.X - other.X
	dy := point.Y - other.Y
	return math.Sqrt(dx * dx + dy * dy)
}

func (point Point) Add(other Point) Point {
	return Point{point.X + other.X, point.Y + other.Y}
}

func (point Point) Sub(other Point) Point {
	return Point{point.X - other.X, point.Y - other.Y}
}

func (point Point) Scale(f float64) Point {
	return Point{f * point.X, f * point.Y}
}

func (point Point) MulPairwise(other Point) Point {
	return Point{point.X * other.X, point.Y * other.Y}
}

func (point Point) AngleTo(other Point) float64 {
	s := point.Dot(other) / point.Magnitude() / other.Magnitude()
	s = math.Max(-1, math.Min(1, s))
	angle := math.Acos(s)
	if angle > math.Pi {
		return 2 * math.Pi - angle
	} else {
		return angle
	}
}

func (point Point) SignedAngle(other Point) float64 {
	return math.Atan2(other.Y, other.X) - math.Atan2(point.Y, point.X)
}

// computes the z-coordinate of the cross product, assuming that
// both points are on the z=0 plane
func (point Point) Cross(other Point) float64 {
	return point.X * other.Y - point.Y * other.X
}

type Segment struct {
	Start Point
	End Point
}

func (segment Segment) Length() float64 {
	return segment.Start.Distance(segment.End)
}

func (segment Segment) Project(point Point, normalized bool) float64 {
	l := segment.Length()
	if l == 0 {
		return 0
	}
	t := point.Sub(segment.Start).Dot(segment.End.Sub(segment.Start)) / l / l
	t = math.Max(0, math.Min(1, t))
	if !normalized {
		t *= l
	}
	return t
}

func (segment Segment) ProjectPoint(point Point) Point {
	t := segment.Project(point, true)
	return segment.PointAtFactor(t, true)
}

func (segment Segment) PointAtFactor(factor float64, normalized bool) Point {
	if segment.Length() == 0 {
		return segment.Start
	}

	if !normalized {
		factor = factor / segment.Length()
	}
	return segment.Start.Add(segment.End.Sub(segment.Start).Scale(factor))
}

func (segment Segment) ProjectWithWidth(point Point, width float64) Point {
	proj := segment.ProjectPoint(point)
	d := point.Sub(proj)
	if d.Magnitude() < 1 {
		return point
	} else {
		d = d.Scale(1 / d.Magnitude())
		return proj.Add(d)
	}
}

func (segment Segment) Distance(point Point) float64 {
	p := segment.ProjectPoint(point)
	return p.Distance(point)
}

func (segment Segment) Vector() Point {
	return segment.End.Sub(segment.Start)
}

func (segment Segment) AngleTo(other Segment) float64 {
	return segment.Vector().AngleTo(other.Vector())
}

func (segment Segment) Bounds() Rectangle {
	return segment.Start.Rectangle().Extend(segment.End)
}

// 2D implementation of "On fast computation of distance between line segments" (V. Lumelsky)
func (segment Segment) DistanceToSegment(other Segment) float64 {
	d1 := segment.Vector()
	d2 := other.Vector()
	d12 := other.Start.Sub(segment.Start)

	r := d1.Dot(d2)
	s1 := d1.Dot(d12)
	s2 := d2.Dot(d12)
	mag1 := d1.Dot(d1)
	mag2 := d2.Dot(d2)

	if mag1 == 0 && mag2 == 0 {
		return segment.Start.Distance(other.Start)
	} else if mag1 == 0 {
		return other.Distance(segment.Start)
	} else if mag2 == 0 {
		return segment.Distance(other.Start)
	}

	denominator := mag1 * mag2 - r * r
	var t, u float64
	if denominator != 0 {
		t = (s1 * mag2 - s2 * r) / denominator
		if t < 0 {
			t = 0
		} else if t > 1 {
			t = 1
		}
	}
	u = (t * r - s2) / mag2
	if u < 0 || u > 1 {
		if u < 0 {
			u = 0
		} else if u > 1 {
			u = 1
		}
		t = (u * r + s1) / mag1
		if t < 0 {
			t = 0
		} else if t > 1 {
			t = 1
		}
	}
	dx := d1.X * t - d2.X * u - d12.X
	dy := d1.Y * t - d2.Y * u - d12.Y
	return math.Sqrt(dx * dx + dy * dy)
}

func (segment Segment) Line() Line {
	return Line{segment.Start, segment.End}
}

// from https://github.com/paulmach/go.geo/blob/master/line.go
func (segment Segment) Intersection(other Segment) *Point {
	d1 := segment.Vector()
	d2 := other.Vector()
	d12 := other.Start.Sub(segment.Start)

	den := d1.Y * d2.X - d1.X * d2.Y
	u1 := d1.X * d12.Y - d1.Y * d12.X
	u2 := d2.X * d12.Y - d2.Y * d12.X

	if den == 0 {
		// collinear
		if u1 == 0 && u2 == 0 {
			return &segment.Start
		} else {
			return nil
		}
	}

	if u1 / den < 0 || u1 / den > 1 || u2 / den < 0 || u2 / den > 1 {
		return nil
	}

	p := segment.PointAtFactor(u2 / den, true)
	return &p
}

type Line struct {
	A Point
	B Point
}

func (line Line) ProjectPoint(point Point) Point {
	vector := line.B.Sub(line.A)
	t := point.Sub(line.A).Dot(vector) / vector.Magnitude() / vector.Magnitude()
	return line.A.Add(vector.Scale(t))
}

type Rectangle struct {
	Min Point
	Max Point
}

var EmptyRectangle Rectangle = Rectangle{
	Min: Point{math.Inf(1), math.Inf(1)},
	Max: Point{math.Inf(-1), math.Inf(-1)},
}

func (rect Rectangle) Extend(point Point) Rectangle {
	return Rectangle{
		Min: Point{
			X: math.Min(rect.Min.X, point.X),
			Y: math.Min(rect.Min.Y, point.Y),
		},
		Max: Point{
			X: math.Max(rect.Max.X, point.X),
			Y: math.Max(rect.Max.Y, point.Y),
		},
	}
}

func (rect Rectangle) ExtendRect(other Rectangle) Rectangle {
	return rect.Extend(other.Min).Extend(other.Max)
}

func (rect Rectangle) Contains(point Point) bool {
	return point.X >= rect.Min.X && point.X <= rect.Max.X && point.Y >= rect.Min.Y && point.Y <= rect.Max.Y
}

func (rect Rectangle) Lengths() Point {
	return rect.Max.Sub(rect.Min)
}

func (rect Rectangle) AddTol(tol float64) Rectangle {
	return Rectangle{
		Min: Point{
			X: rect.Min.X - tol,
			Y: rect.Min.Y - tol,
		},
		Max: Point{
			X: rect.Max.X + tol,
			Y: rect.Max.Y + tol,
		},
	}
}

func (rect Rectangle) Bounds() Rectangle {
	return rect
}

func (rect Rectangle) Intersects(other Rectangle) bool {
	return rect.Max.Y >= other.Min.Y && other.Max.Y >= rect.Min.Y && rect.Max.X >= other.Min.X && other.Max.X >= rect.Min.X
}
