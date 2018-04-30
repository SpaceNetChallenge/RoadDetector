package common

import (
	"github.com/ajstarks/svgo"

	"fmt"
	"math"
	"math/rand"
	"os"
)

type WeightedBoundable struct {
	Boundable Boundable
	Weight float64
}

func (wb WeightedBoundable) Bounds() Rectangle {
	return wb.Boundable.Bounds()
}

type WidthBoundable struct {
	Boundable Boundable
	Width float64
}

func (b WidthBoundable) Bounds() Rectangle {
	return b.Boundable.Bounds()
}

type ColoredBoundable struct {
	Boundable Boundable
	Color string
}

func (b ColoredBoundable) Bounds() Rectangle {
	return b.Boundable.Bounds()
}

type SvgLabel struct {
	Point Point
	Text string
}

func (b SvgLabel) Bounds() Rectangle {
	return b.Point.Bounds()
}

type EmbeddedImage struct {
	Src Point
	Dst Point
	Image string
}

func (img EmbeddedImage) Bounds() Rectangle {
	return EmptyRectangle.Extend(img.Src).Extend(img.Dst)
}

type WrappingBoundable interface {
	Unwrap() Boundable
}

type SVGOptions struct {
	Scale float64
	Zoom float64
	ScaleX float64
	ScaleY float64
	Sparse float64
	Bounds Rectangle
	Blur float64
	StrokeWidth float64
	Color string
	Unflip bool
}

func CreateSVG(fname string, elements [][]Boundable, options SVGOptions) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()

	// get bounds over all elements
	var r Rectangle
	if options.Bounds.Max.X > options.Bounds.Min.X {
		r = options.Bounds
	} else {
		r = EmptyRectangle
		for _, l := range elements {
			for _, element := range l {
				r = r.ExtendRect(element.Bounds())
			}
		}
	}

	// autoscale if requested
	scaleFactor := options.Scale
	if scaleFactor == 0 {
		l := math.Max(r.Lengths().X, r.Lengths().Y)
		scaleFactor = 1000 / l
	}
	scale := Point{scaleFactor, scaleFactor}
	if options.ScaleX != 0 && options.ScaleY != 0 {
		scale.X = options.ScaleX
		scale.Y = options.ScaleY
	}
	if options.Zoom != 0 {
		scale = scale.Scale(options.Zoom)
	}

	strokeWidth := options.StrokeWidth
	if strokeWidth == 0 {
		strokeWidth = 2
	}

	canvas := svg.New(file)
	length := r.Lengths().MulPairwise(scale)
	canvas.Start(int(length.X) + 1, int(length.Y) + 1)
	canvas.Rect(0, 0, int(length.X) + 1, int(length.Y) + 1, "fill:white")
	if options.Blur > 0 {
		canvas.Filter("blur")
		canvas.FeGaussianBlur(svg.Filterspec{In: "SourceGraphic"}, options.Blur, options.Blur)
		canvas.Fend()
		canvas.Group("filter=\"url(#blur)\"")
		canvas.Rect(0, 0, int(length.X) + 1, int(length.Y) + 1, "fill:white")
	}
	transform := func(point Point) (int, int) {
		p := point.Sub(r.Min).MulPairwise(scale)
		if options.Unflip {
			return int(p.X), int(p.Y)
		} else {
			return int(p.X), int(length.Y - p.Y)
		}
	}
	var drawElement func(element Boundable, color string, weight float64, width float64) error
	drawElement = func(element Boundable, color string, weight float64, width float64) error {
		switch element := element.(type) {
		case Point:
			if !r.Contains(element) {
				return nil
			}
			x, y := transform(element)
			style := fmt.Sprintf("fill:%s", color)
			if weight != 1 {
				style += fmt.Sprintf(";opacity:%f", weight)
			}
			canvas.Circle(x, y, int(width), style)
		case Segment:
			if !r.Intersects(element.Bounds()) {
				return nil
			}
			srcX, srcY := transform(element.Start)
			dstX, dstY := transform(element.End)
			style := fmt.Sprintf("stroke:%s;stroke-width:%f", color, width)
			if weight != 1 {
				style += fmt.Sprintf(";opacity:%f", weight)
			}
			canvas.Line(srcX, srcY, dstX, dstY, style)
		case *Graph:
			for _, edge := range element.Edges {
				if err := drawElement(edge.Segment(), color, weight, width); err != nil {
					return err
				}
			}
		case Traces:
			for _, trace := range element {
				for j := 1; j < len(trace.Observations); j++ {
					if err := drawElement(Segment{trace.Observations[j - 1].Point, trace.Observations[j].Point}, color, weight, width); err != nil {
						return err
					}
				}
			}
		case SvgLabel:
			style := fmt.Sprintf("fill:%s;font-size:%f", color, width)
			x, y := transform(element.Point)
			canvas.Text(x, y, element.Text, style)
		case WeightedBoundable:
			return drawElement(element.Boundable, color, element.Weight, width)
		case WidthBoundable:
			return drawElement(element.Boundable, color, weight, element.Width)
		case ColoredBoundable:
			return drawElement(element.Boundable, element.Color, weight, width)
		case EmbeddedImage:
			srcX, srcY := transform(element.Src)
			dstX, dstY := transform(element.Dst)
			canvas.Image(srcX, srcY, dstX - srcX, dstY - srcY, element.Image)
		case WrappingBoundable:
			return drawElement(element.Unwrap(), color, weight, width)
		default:
			return fmt.Errorf("failed to process an element: %v", element)
		}
		return nil
	}
	var colors []string
	if options.Color != "" {
		colors = []string{options.Color}
	} else {
		colors = []string{"red", "blue", "green", "purple", "olive", "gray"}
	}
	for i, l := range elements {
		color := colors[i % len(colors)]
		for _, element := range l {
			if options.Sparse > 0 && rand.Float64() >= options.Sparse {
				continue
			}
			if err := drawElement(element, color, 1, strokeWidth); err != nil {
				return err
			}
		}
	}
	if options.Blur > 0 {
		canvas.Gend()
	}
	canvas.End()
	return nil
}
