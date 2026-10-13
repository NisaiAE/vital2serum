"""Oscillator-to-oscillator FM and RM warps in both synths.

Measured on Vital 1.0.7 and Serum 2.1.5 from the spectral lines and the waveform of a sine
carrier under a sine modulator. theta is the carrier's phase, m the modulator signal it reads.

Vital (distortion types 7/8 = FM, 10/11 = RM; amount a = 0..1):
    FM   sin(theta + 12 pi a^2 m)             a phase modulation of +-6 a^2 cycles
    RM   sin(theta) ((1 - a) + a m)
  m is the other oscillator's output before its level, pan and routing and after its own warp,
  spectral morph and unison stack: a full-scale table read by one voice peaks at 1. The whole
  stack is summed, so every carrier voice hears the same chorused modulator. A modulator that
  is switched off gives m = 0 (FM does nothing, RM leaves (1 - a) of the carrier).
  'A' (types 7, 10) is the lower-numbered of the two other oscillators, 'B' (8, 11) the higher.

Serum (kParamWarp w = 0..1; values above 1 act as 1):
    kPD_OSC   sin(theta + 16 pi w^4 (1 + 1.0031 m))   one-sided: 0..16 w^4 cycles, so the carrier
              also runs 16 pi w^4 rad ahead; the index is 50.42 w^4
    kFM_OSC   a frequency deviation of 88.4 kHz w^4 m: its index falls as the modulator rises
    kRM_OSC   sin(theta) ((1 - w) + 2.006 w m)
    kAM_OSC   sin(theta) (1 + 2.006 w m)
  ..._OSC reads the lower-numbered other oscillator and ..._OSC2 the higher, as in Vital. m is
  the modulator before its volume, pan and routing and after its own warp. Unison is paired
  voice by voice: a carrier voice reads one modulator voice (a lone carrier voice the lowest
  detuned one), never the sum. A disabled modulator switches the warp off altogether.
  A modulation amount of 100 moves kParamWarp across its whole range, linearly. The second warp
  slot (kParamWarpMenu2, kParamWarp2) follows the same laws; two kPD_OSC slots add their indices.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math
from .unison import centre_gain, vital_side_db

VITAL_FM_INDEX = 12.0*math.pi     # rad at amount 1 for a full-scale modulator
SERUM_PD_INDEX = 50.42            # rad at kParamWarp 1 for a full-scale table
SERUM_PD_LEAD = 16.0*math.pi      # rad the carrier runs ahead at kParamWarp 1
SERUM_FM_HZ = 88.4e3              # peak deviation of kFM_OSC at kParamWarp 1 (44.1 kHz)
SERUM_RM_GAIN = 2.0064            # sidebands of kRM_OSC / kAM_OSC over the Vital form
MENUS = {7: 'kPD_OSC', 8: 'kPD_OSC2', 10: 'kRM_OSC', 11: 'kRM_OSC2'}


def modulator(carrier: int, kind: int) -> int | None:
    """Oscillator (1..3) that Vital's distortion type `kind` reads on oscillator `carrier`.

    None for the types that read the sample (9, 12) or no oscillator at all. Serum's menu from
    MENUS reads the same oscillator from the same slot.
    """
    if kind not in MENUS:
        return None
    low, high = [i for i in (1, 2, 3) if i != carrier]
    return low if kind in (7, 10) else high


def stack_gain(voices: int, blend: float, spread: float) -> float:
    """Amplitude (rms) of a Vital modulator's unison stack over one voice at full stereo spread.

    The carrier reads one channel of the stack. Vital keeps the power of the stack constant, but
    a centre voice gains up to 3 dB as stereo_spread narrows (see unison.centre_gain).
    """
    boost = centre_gain(spread)
    if voices <= 1:
        return math.sqrt(boost)
    if voices % 2 == 0:
        return 1.0
    side = 10.0**(vital_side_db(blend)/10.0)
    centre = 1.0/(1.0+(voices-1)*side)
    return math.sqrt(centre*boost+(voices-1)*side*centre)


def vital_index(amount: float, peak: float = 1.0) -> float:
    """Phase deviation (rad) of Vital's FM for a modulator peaking at `peak`."""
    return VITAL_FM_INDEX*amount*amount*peak


def serum_index(warp: float, peak: float = 1.0) -> float:
    """Phase deviation (rad) of Serum's kPD_OSC for a modulator table peaking at `peak`."""
    w = min(1.0, max(0.0, warp))
    return SERUM_PD_INDEX*w**4*peak


def pd_warp(amount: float, ratio: float = 1.0) -> float:
    """kParamWarp of kPD_OSC with the index of Vital's FM amount.

    `ratio` is the Vital modulator's amplitude over the Serum modulator's: the gain the table
    was divided by on conversion times stack_gain(). The result is also the position of the
    control for a modulated amount. Serum stops at pd_reach(ratio).
    """
    a = min(1.0, max(0.0, amount))
    return min(1.0, (VITAL_FM_INDEX*ratio/SERUM_PD_INDEX)**0.25*math.sqrt(a))


def pd_reach(ratio: float = 1.0) -> float:
    """Largest Vital FM amount one kPD_OSC slot can follow (above 1: the whole range)."""
    return math.sqrt(SERUM_PD_INDEX/(VITAL_FM_INDEX*ratio))


def pd_warps(amount: float, ratio: float = 1.0) -> tuple:
    """(kParamWarp, kParamWarp2) with the index of Vital's FM amount.

    Two slots set to the same menu add their indices (measured up to 100.8 rad), so the second
    takes what the first cannot reach. It stays 0.0 (leave the slot alone) within pd_reach().
    """
    a = min(1.0, max(0.0, amount))
    rest = VITAL_FM_INDEX*a*a*ratio-SERUM_PD_INDEX
    if rest <= 0.0:
        return pd_warp(a, ratio), 0.0
    return 1.0, min(1.0, (rest/SERUM_PD_INDEX)**0.25)


def pd_lead(amount: float, ratio: float = 1.0) -> float:
    """Degrees the Serum carrier runs ahead under kPD_OSC at pd_warps(amount, ratio).

    Subtract it from the carrier's kParamInitialPhase: the waveform is then Vital's, which
    matters when modulator and carrier keep a fixed phase (octaves, no random phase).
    """
    first, second = pd_warps(amount, ratio)
    return math.degrees(SERUM_PD_LEAD*(first**4+second**4)) % 360.0


def rm_warp(amount: float, ratio: float = 1.0) -> float:
    """kParamWarp of kRM_OSC with Vital's balance of sidebands to carrier (also the position)."""
    d = min(1.0, max(0.0, amount))
    g = SERUM_RM_GAIN/ratio
    return d/(d+g*(1.0-d)) if d > 0.0 else 0.0


def rm_gain(amount: float, ratio: float | None = 1.0) -> float:
    """Amplitude gain for the carrier's level that brings kRM_OSC at rm_warp() to Vital's level.

    With ratio None the modulator is switched off: write no warp, Vital leaves (1 - amount).
    """
    d = min(1.0, max(0.0, amount))
    return (1.0-d)+(d*ratio/SERUM_RM_GAIN if ratio else 0.0)
