"""Serum 2 curve data, shared by LFO shapes and modulation remap curves.

Everything here was measured on Serum 2.1.5 by reading an LFO or a macro through oscillator
pitch. Points are stored as xVals / yVals with y = 0 at the top, the
same orientation Vital uses, plus one curve value per segment.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math

NO_LOOPBACK = 481            # curveData.loopbackPointNum in every stock LFO without a loop point
SMOOTH_SUBDIVISIONS = 8
REMAP_POINTS = 33
ENVELOPE_TAIL_CONSTANTS = 12.0
ENVELOPE_END_MARGIN = 0.002


def power_scale(t: float, power: float) -> float:
    """Vital's futils::powerScale: the bend both synths draw between two points."""
    if abs(power) < 0.01:
        return t
    return (math.exp(power*t)-1.0)/(math.exp(power)-1.0)


def curve_value(power: float) -> float:
    """Serum's curve value for a segment that follows Vital's powerScale(t, power).

    Serum draws the same exponential. Its stored value c (0..1, 0.5 = straight) puts the level
    reached half-way along the segment at 4 (1 - c)^3 for c >= 0.5, mirrored below 0.5.
    """
    if abs(power) < 0.01:
        return 0.5
    half = 1.0/(1.0+math.exp(abs(power)/2.0))
    c = 1.0-(half/4.0)**(1.0/3.0)
    return c if power > 0 else 1.0-c


def lfo_curve(points: list, powers: list, smooth: bool) -> dict:
    """Vital LFO points -> Serum LFO curveData.

    Serum lists its points plus a closing entry at x = 1 whose level it takes from the first
    point; numPoints is the entry count minus one. A shape that ends at a different level than
    it starts gets an extra first point holding the end level, so the curve jumps at x = 0 the
    way Vital wraps around.
    """
    n = len(points)
    if smooth and n*SMOOTH_SUBDIVISIONS <= 512:
        # Serum has no smoothed segments; sample Vital's eased segment into straight pieces.
        fine = []
        for i in range(n-1):
            (x0, y0), (x1, y1) = points[i], points[i+1]
            for j in range(SMOOTH_SUBDIVISIONS):
                t = j/SMOOTH_SUBDIVISIONS
                eased = power_scale(0.5*math.sin((t-0.5)*math.pi)+0.5, powers[i])
                fine.append((x0+(x1-x0)*t, y0+(y1-y0)*min(1.0, max(0.0, eased))))
        fine.append(points[-1])
        points = fine; powers = [0.0]*len(points); n = len(points)
    curves = [curve_value(p) for p in powers]
    first, last = points[0][1], points[-1][1]
    if abs(first-last) < 1e-9:
        xs = [x for x, _ in points]; ys = [y for _, y in points]; cv = curves[:n-1]+[0.5]
    else:
        xs = [0.0]+[x for x, _ in points[:-1]]+[1.0]
        ys = [last]+[y for _, y in points[:-1]]+[last]
        cv = [0.5]+curves[:n-1]+[0.5]
    return dict(xVals=xs, yVals=ys, curveVals=cv, numPoints=len(xs)-1, loopbackPointNum=NO_LOOPBACK)


def envelope_tail_curve(points: list, powers: list, smooth: bool, hz: float, tau: float) -> tuple:
    """Keep an Envelope's real endpoint long enough for Serum's smoothing to settle.

    Serum holds a phase near .999, including its smoothing state. Scale both the original
    points and rate by q so their times stay the same, then hold the endpoint for twelve time
    constants. Duplicate x=1 points close Vital's loop; the value before that jump is held.
    """
    smooth=smooth and len(points)*SMOOTH_SUBDIVISIONS<=512
    points=list(points)
    while len(points)>2 and abs(points[-1][0]-points[-2][0])<1e-9:
        points.pop()
    q=(1.0-ENVELOPE_END_MARGIN)/(1.0+ENVELOPE_TAIL_CONSTANTS*tau*hz)
    tail=1.0/(hz*q)-1.0/hz
    curve=lfo_curve(points,list(powers[:len(points)-1])+[0.0],smooth)
    curve['xVals']=[x*q for x in curve['xVals']]+[1.0]
    curve['yVals'].append(points[-1][1])
    curve['curveVals'].append(0.5)
    curve['numPoints']+=1
    return curve,q,tail


def remap_curve(points: list, powers: list | None = None) -> dict:
    """A modulation slot's source curve through (input, output) pairs, both 0..1.

    Unlike an LFO shape this curve does not loop: every entry is a real point.
    """
    powers = list(powers or [])+[0.0]*len(points)
    return dict(xVals=[float(x) for x, _ in points], yVals=[1.0-float(y) for _, y in points],
                curveVals=[curve_value(p) for p in powers[:len(points)-1]]+[0.5], numPoints=len(points)-1)


def sampled_curve(function, count: int = REMAP_POINTS) -> dict:
    """Straight-line remap curve through `function` (0..1 -> 0..1)."""
    return remap_curve([(i/(count-1), min(1.0, max(0.0, function(i/(count-1))))) for i in range(count)])


def vital_power_curve(power: float, bipolar: bool) -> dict | None:
    """Remap curve for Vital's modulation "power" (measured on Vital 1.0.7).

    Vital bends the source with powerScale(x, -power); a bipolar routing bends each half away
    from the centre. Serum applies its curve before centring a bipolar source, so the same
    picture works there.
    """
    if abs(power) < 0.01:
        return None
    if bipolar:
        return remap_curve([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)], [power, -power])
    return remap_curve([(0.0, 0.0), (1.0, 1.0)], [-power])


def vital_remap(value: float, power: float, bipolar: bool) -> float:
    """Value of a Vital modulation source after its power remap, still 0..1."""
    if abs(power) < 0.01:
        return value
    if not bipolar:
        return power_scale(value, -power)
    side = 2.0*value-1.0
    return 0.5+0.5*math.copysign(power_scale(abs(side), -power), side)


def line_value(points: list, powers: list, x: float) -> float:
    """Value (0..1) of a Vital line (points as (x, y) with y = 0 at the top) at `x`.

    Each segment bends with its own power; a step is two points on the same x, and the later one
    holds from there. Outside the first and last point the end values hold.
    """
    if x <= points[0][0]:
        return 1.0-points[0][1]
    for i in range(len(points)-1, 0, -1):
        (x0, y0), (x1, y1) = points[i-1], points[i]
        if x >= x1:
            return 1.0-y1
        if x >= x0 and x1 > x0:
            t = (x-x0)/(x1-x0)
            return 1.0-(y0+(y1-y0)*power_scale(t, powers[i-1] if i-1 < len(powers) else 0.0))
    return 1.0-points[0][1]


def line_curve(points: list, powers: list) -> dict:
    """A modulation slot's source curve following a Vital line mapping (its remap editor)."""
    points = [(min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))) for x, y in points]
    powers = [float(p) for p in powers[:len(points)-1]]+[0.0]*max(0, len(points)-1-len(powers))
    if points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1])); powers.insert(0, 0.0)
    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1])); powers.append(0.0)
    return remap_curve([(x, 1.0-y) for x, y in points], powers)
