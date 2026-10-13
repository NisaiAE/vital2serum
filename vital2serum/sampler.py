"""Vital's sample source on Serum 2's Noise oscillator: pitch, start position, loop and pan.

Measured by playing the same clips (a 440 Hz sine, a linear chirp) through Vital 1.0.7 and
Serum 2.1.5 and reading the output frequency, the position in the clip and the channel levels.
Pure functions; mapping._sample embeds the PCM itself.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math

# --- Vital -------------------------------------------------------------------------------------
# With keytrack the clip runs at its own speed on C4 and follows the keyboard; without it every
# key plays the same speed. sample_transpose is rounded to whole semitones (halves away from
# zero) and limited to +-48, sample_tune to +-1. voice_transpose and voice_tune move the sample
# only with keytrack. The clip's sample rate and the host's are both compensated.
VITAL_ROOT = 60
VITAL_MAX_TRANSPOSE = 48
GLOBAL_SNAP = 1 << 12             # sample_transpose_quantize: snap the played note, not the transposition

# --- Serum -------------------------------------------------------------------------------------
# Oscillator3.kParamPitchTrack = 1: the clip runs at its own speed on C5 and follows the keyboard.
# kParamColor then moves it in whole semitones, round(128 (colour - 0.5)), and kParamFine by up
# to +-1 semitone. The oscillator's own octave, semitone, fine and coarse pitch do nothing.
SERUM_ROOT = 72
TRACKED_RANGE = 128.0
TRACKED_LIMIT = 64
# The tracked speed is proportional to the field detuneFactor. Serum stores sample rate / 2^16.5
# there (0.5179013309837353 for its own 48 kHz files); the field sampleRate is not read, and a
# missing detuneFactor means 44.1 kHz.
DETUNE_FACTOR_44100 = 0.4758218478413068
# Without pitch tracking kParamColor is continuous: 96 semitones per unit above 0.5 and 384 per
# unit below. Every clip is then played as if it were 44.1 kHz (a 48 kHz clip sounds 1.47
# semitones low at host rates of 44.1, 48 and 96 kHz), and kParamFine moves the pitch by an
# amount that depends on the colour (0.96 semitones at 0.5, 0.6 at 0.75). The tuning and the
# clip's rate therefore go into the colour.
UNTRACKED_UP = 96.0
UNTRACKED_DOWN = 384.0
UNTRACKED_CLIP_RATE = 44100.0
# The start position is kParamInitialPhase (% of the clip) plus a random part spread evenly over
# (kParamRandomPhase / 100)^2 of the clip; both default to 0. Vital's random phase covers the
# whole clip. A one-shot started inside the clip stops at the clip's end in both synths.
RANDOM_WHOLE_CLIP = 100.0
RANDOM_FIRST_HALF = 100.0*math.sqrt(0.5)
# Channel gain against u = +-kParamPan / 50, relative to the centre (11 measured points fit
# within 0.002). Vital pans with sqrt(2) sin / cos, so Serum is 1.37 dB louder at full pan.
PAN_CURVE = 3.0-2.0*math.sqrt(2.0)
MIRROR_LIMIT = 5_000_000          # frames; a mirrored clip stays within the converter's clip limit


def _round(x: float) -> int:
    return int(math.copysign(math.floor(abs(x)+0.5), x))


def snap(transpose: int, quantize: int) -> int:
    """Vital's transpose quantize: the nearest enabled pitch class (bit 0 = no transposition).

    A transposition half-way between two enabled notes goes to the lower one.
    """
    allowed = [pc for pc in range(12) if quantize >> pc & 1]
    if not allowed:
        return transpose
    octave, pc = divmod(transpose, 12)
    candidates = [a+12*k for a in allowed for k in (-1, 0, 1)]
    return octave*12+min(candidates, key=lambda c: (abs(c-pc), c))


def vital_transpose(transpose: float, quantize: int = 0) -> int:
    """Whole semitones Vital adds for sample_transpose."""
    limited = max(-VITAL_MAX_TRANSPOSE, min(VITAL_MAX_TRANSPOSE, _round(transpose)))
    return snap(limited, int(quantize))


def colour(semitones: float, tracked: bool) -> float:
    """kParamColor that moves the clip by `semitones` (0.5 = the clip's own speed), unclamped."""
    if tracked:
        return 0.5+semitones/TRACKED_RANGE
    return 0.5+semitones/(UNTRACKED_UP if semitones >= 0 else UNTRACKED_DOWN)


def detune_factor(sample_rate: float) -> float:
    """NoiseOsc3.detuneFactor for a clip of this sample rate."""
    return DETUNE_FACTOR_44100*sample_rate/44100.0


def _pan_gain(u: float) -> float:
    return (1.0+u)*(1.0-PAN_CURVE*u)


def pan(vital_pan: float) -> tuple:
    """(kParamPan, amplitude gain) placing the sample where Vital's sample_pan puts it.

    Both synths treat a stereo clip as a balance control (the far channel fades out). The pan
    is chosen for the same right / left ratio and the gain makes up the level difference.
    """
    p = max(-1.0, min(1.0, vital_pan))
    if abs(p) < 1e-9:
        return 0.0, 1.0
    theta = (abs(p)+1.0)*math.pi/4.0
    near, far = math.sqrt(2.0)*math.sin(theta), math.sqrt(2.0)*math.cos(theta)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        u = (lo+hi)/2.0
        if _pan_gain(-u)*near > _pan_gain(u)*far: lo = u
        else: hi = u
    u = (lo+hi)/2.0
    return math.copysign(50.0*u, p), near/_pan_gain(u)


def mirror(channels: list) -> list:
    """Each channel followed by its reverse: looped, this is Vital's bounce."""
    return [list(ch)+list(ch)[::-1] for ch in channels]


def noise_settings(keytrack: bool, transpose: float, tune: float, random_phase: bool, loop: bool,
                   bounce: bool, sample_rate: float, frames: int, *, transpose_quantize: int = 0,
                   voice_transpose: float = 0.0, voice_tune: float = 0.0) -> dict:
    """Serum Noise oscillator settings for Vital's sample playback.

    Returns `oscillator` (plain params to add to Oscillator3), `noise` (the plain params of
    NoiseOsc3), `fields` (NoiseOsc3 fields), `mirror` (embed mirror(pcm) instead of the clip;
    fields['numFrames'] already counts it), `transpose_law` (sample_transpose -> position of
    kParamColor, for a modulation destination) and `notes` as (kind, feature, message).
    """
    notes = []
    quantize = int(transpose_quantize)
    tune = max(-1.0, min(1.0, tune))
    tracked = bool(keytrack)
    if tracked:
        if quantize & GLOBAL_SNAP and quantize & (GLOBAL_SNAP-1):
            notes.append(('unsupported', 'sample_transpose_quantize',
                          "Snapping the played note to a scale has no counterpart on Serum's Noise oscillator; "
                          'only the transposition is snapped.'))
        # The colour moves in whole semitones and kParamFine stops at +-1.
        offset, fine = float(SERUM_ROOT-VITAL_ROOT)+voice_transpose, tune+voice_tune
        if offset != int(offset) or abs(fine) > 1.0:
            total = offset+fine
            offset = float(_round(total)); fine = total-offset
        lowest, highest = -float(TRACKED_LIMIT), float(TRACKED_LIMIT)
    else:
        # Vital ignores the voice transposition here; Serum needs the clip's rate made up.
        offset, fine = tune+12.0*math.log2(sample_rate/UNTRACKED_CLIP_RATE), 0.0
        lowest, highest = -UNTRACKED_DOWN/2.0, UNTRACKED_UP/2.0
    semitones = vital_transpose(transpose, quantize)+offset
    if not lowest <= semitones <= highest:
        clamped = max(lowest, min(highest, semitones))
        notes.append(('clipped', 'sample_transpose',
                      f"Sample pitch of {semitones:+.2f} semitones is outside the Noise oscillator's range; "
                      f'{clamped:+.2f} is used.'))
        semitones = clamped
    noise = {'kParamOneShot': 0.0 if loop else 1.0}
    if colour(semitones, tracked) != 0.5: noise['kParamColor'] = colour(semitones, tracked)
    if abs(fine) > 1e-9: noise['kParamFine'] = fine
    mirrored = bool(bounce)
    if mirrored and frames > MIRROR_LIMIT:
        mirrored = False
        notes.append(('unsupported', 'sample_bounce',
                      'Clip too long to embed with its reverse; it loops forwards only.'))
    if random_phase:
        # Vital starts anywhere on the forward pass, which is the first half of a mirrored clip.
        noise['kParamRandomPhase'] = RANDOM_FIRST_HALF if mirrored else RANDOM_WHOLE_CLIP
    def transpose_law(value: float) -> float:
        value = max(-VITAL_MAX_TRANSPOSE, min(VITAL_MAX_TRANSPOSE, value))
        return colour(max(lowest, min(highest, value+offset)), tracked)
    return dict(oscillator={'kParamPitchTrack': 1.0} if tracked else {}, noise=noise,
                fields={'detuneFactor': detune_factor(sample_rate), 'numFrames': int(frames)*(2 if mirrored else 1)},
                mirror=mirrored, transpose_law=transpose_law, notes=notes)
