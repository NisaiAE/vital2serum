"""Unison: voice detuning, centre/side balance and stereo width in both synths.

Measured from the spectral lines of a sine oscillator (Vital 1.0.7 and Serum 2.1.5). Levels are powers relative to one voice with unison off.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math
from .curves import power_scale

# --- Vital -------------------------------------------------------------------------------------
# Level of one detuned voice over the centre voice (dB) against unison_blend. Vital keeps the
# total power of the stack equal to that of a single voice.
VITAL_SIDE_DB = [(0.05, -27.42), (0.1, -21.35), (0.2, -15.23), (0.3, -11.59), (0.4, -8.95), (0.5, -6.86),
                 (0.6, -5.10), (0.7, -3.55), (0.8, -2.14), (0.9, -0.80), (1.0, 0.50)]

# --- Serum -------------------------------------------------------------------------------------
# kParamDetuneWid is the unison blend. Level of a detuned voice against the blend (dB relative
# to blend 100), in steps of 5. The centre voice follows 1 - 0.555 (blend / 100)^2 in amplitude.
SERUM_SIDE_DB = [-22.49, -21.88, -20.29, -18.18, -15.92, -13.72, -11.68, -9.83, -8.16, -6.67, -5.35,
                 -4.18, -3.15, -2.26, -1.51, -0.89, -0.40, -0.06, 0.14, 0.17, 0.0]
SERUM_CENTRE_LOSS = 0.555
# Every voice loses 0.25 dB per unison voice. Levels at blend 0 (centre) and blend 100 (side):
SERUM_PER_VOICE_DB = 0.25
SERUM_CENTRE_ODD_DB = -6.41       # the single centre voice of an odd count, both channels
SERUM_CENTRE_EVEN_DB = -1.39      # each of the two centre voices of an even count
SERUM_SIDE_BASE_DB = -5.95
SERUM_PAIR_DB = -4.88             # each voice of a two-voice stack, whatever the blend
# kParamUnisonStereo in steps of 10: level of a detuned voice in its far channel over its near
# channel (amplitude), and the drop of its total power (dB). Serum's pan is not constant-power.
SERUM_WIDTH_RATIO = [1.0, 0.7766, 0.6005, 0.4603, 0.3475, 0.2564, 0.1823, 0.1219, 0.0727, 0.0326, 0.0]
SERUM_WIDTH_DROP_DB = [-5.02, -4.93, -4.67, -4.27, -3.75, -3.17, -2.55, -1.90, -1.26, -0.62, 0.0]

DETUNE_MODES = {
    None: lambda t: t,                                   # Linear (Serum's default)
    'kDetuneSuper': lambda t: math.tan(t)/math.tan(1.0),
    'kDetuneExp': lambda t: t*t,
    'kDetuneInv': lambda t: math.sin(math.pi*t/2.0),
}


def _interp(x: float, xs: list, ys: list) -> float:
    if x <= xs[0]: return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1: return y0+(y1-y0)*(x-x0)/(x1-x0)
    return ys[-1]


def voice_positions(voices: int) -> list:
    """Distance of each detuned pair from the centre, as a fraction of the outermost pair."""
    if voices % 2:
        pairs = (voices-1)//2
        return [k/pairs for k in range(1, pairs+1)]
    return [(2*k-1)/(voices-1) for k in range(1, voices//2+1)]


def detune_mode(voices: int, power: float) -> str | None:
    """Serum detune mode whose voice spacing is closest to Vital's detune_power.

    Vital places a voice at powerScale(t, detune_power) of the full detune; Serum offers four
    fixed spacings. The outermost voices coincide in every mode.
    """
    inner = [t for t in voice_positions(voices) if t < 1.0-1e-9]
    if not inner:
        return None
    def worst(mode):
        return max(abs(DETUNE_MODES[mode](t)-power_scale(t, power)) for t in inner)
    return min(DETUNE_MODES, key=lambda mode: (worst(mode), mode is not None))


def serum_range(vital_range: float) -> int:
    """Native range is an integer number of semitones, including zero."""
    return math.ceil(min(48.0,max(0.0,vital_range)))


def _range_ratio(vital_range: float) -> float:
    native=serum_range(vital_range)
    return min(48.0,max(0.0,vital_range))/native if native else 0.0


def serum_detune(vital_detune: float, vital_range: float=2.0) -> float:
    """kParamDetune for Vital's unison_detune (0..10) at the same detune range.

    Vital spreads the outer voices by (detune / 10)^2 of the range; Serum by detune^1.25.
    """
    return (max(0.0, vital_detune)/10.0)**1.6*_range_ratio(vital_range)**0.8


def detune_position(vital_detune: float, vital_range: float=2.0) -> float:
    """Position of Serum's detune control on its modulation scale (the stored value is its square)."""
    return (max(0.0, vital_detune)/10.0)**0.8*_range_ratio(vital_range)**0.4


def vital_side_db(blend: float) -> float:
    if blend <= 0.0:
        return -120.0
    first = VITAL_SIDE_DB[0]
    if blend < first[0]:                       # a detuned voice's amplitude is proportional to blend here
        return first[1]+20.0*math.log10(blend/first[0])
    return _interp(blend, [b for b, _ in VITAL_SIDE_DB], [d for _, d in VITAL_SIDE_DB])


def centre_gain(spread: float) -> float:
    """Power gain of Vital's centre voice from stereo_spread.

    The centre voice runs once per channel. Narrowing the spread pans the two copies towards
    each other, where they add in phase: up to +3 dB at spread 0. A lone voice does the same.
    """
    return 1.0+math.sin((1.0-min(1.0, max(0.0, spread)))*math.pi/2.0)


def serum_width(spread: float) -> float:
    """kParamUnisonStereo that leaves a detuned voice as far to one side as Vital's stereo_spread."""
    target = math.tan((1.0-min(1.0, max(0.0, spread)))*math.pi/4.0)       # far / near channel in Vital
    widths = [10.0*i for i in range(11)]
    return _interp(-target, [-x for x in SERUM_WIDTH_RATIO], widths)


def _serum_side(blend: float) -> float:
    return _interp(blend, [5.0*i for i in range(21)], SERUM_SIDE_DB)


def _serum_centre(blend: float) -> float:
    return 20.0*math.log10(1.0-SERUM_CENTRE_LOSS*(blend/100.0)**2)


def match(voices: int, blend: float, spread: float) -> tuple:
    """(kParamDetuneWid, kParamUnisonStereo, gain in dB) reproducing a Vital unison stack.

    The blend is chosen so a detuned voice sits at Vital's level relative to the centre voice,
    the width so it sits as far to the side, and the gain makes up the difference in total power.
    Either setting is None where Serum's control does nothing for that voice count.
    """
    boost = centre_gain(spread)
    if voices <= 1:
        return None, None, 10.0*math.log10(boost)
    width = serum_width(spread)
    drop = _interp(width, [10.0*i for i in range(11)], SERUM_WIDTH_DROP_DB)
    if voices == 2:
        return None, width, -(10.0*math.log10(2.0)+SERUM_PAIR_DB+drop)
    ratio = 10.0**(vital_side_db(blend)/10.0)              # power of a detuned voice over the centre
    loss = SERUM_PER_VOICE_DB*voices
    if voices % 2:
        centre = 1.0/(1.0+(voices-1)*ratio)
        vital_total = centre*boost+(voices-1)*ratio*centre
        target = 10.0*math.log10(ratio/boost)
        def serum_ratio(b): return SERUM_SIDE_BASE_DB+_serum_side(b)+drop-SERUM_CENTRE_ODD_DB-_serum_centre(b)
    else:
        vital_total = 1.0
        target = 10.0*math.log10(2.0*ratio)                # each centre voice carries half the centre power
        def serum_ratio(b): return SERUM_SIDE_BASE_DB+_serum_side(b)-SERUM_CENTRE_EVEN_DB-_serum_centre(b)
    best = min((0.5*i for i in range(201)), key=lambda b: abs(serum_ratio(b)-target))
    side = 10.0**((SERUM_SIDE_BASE_DB-loss+_serum_side(best)+drop)/10.0)
    if voices % 2:
        serum_total = 10.0**((SERUM_CENTRE_ODD_DB-loss+_serum_centre(best))/10.0)+(voices-1)*side
    else:
        serum_total = 2.0*10.0**((SERUM_CENTRE_EVEN_DB-loss+drop+_serum_centre(best))/10.0)+(voices-2)*side
    return best, width, 10.0*math.log10(vital_total/serum_total)
