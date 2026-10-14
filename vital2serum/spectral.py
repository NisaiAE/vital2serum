"""Full-table spectral morphs with Vital's phase and amplitude laws.

Independent NumPy formulation of mtytel/vital commit
636ca0ef517a4db087a6a08a6a8a5e704e21f836 (Matt Tytel),
spectral_morph.h, wavetable.cpp and futils.h.
Copyright 2013-2019 Matt Tytel
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .wavetable import FRAME_SIZE, NUM_POSITIONS, _MT19937
from .modulation import active_modulations, amount_modulators

SIZE = FRAME_SIZE
HARMONICS = SIZE // 2 + 1
STATIC_MORPHS = {2: 'formant scale', 5: 'smear', 6: 'random amplitudes',
                 9: 'phase disperse', 10: 'static Shepard blend', 11: 'wavetable skew'}
TRAJECTORY_MORPHS = {k:v for k,v in STATIC_MORPHS.items() if k != 10}
TYPES = STATIC_MORPHS


def static_eligible(v, prefix: str) -> bool:
    """Retain the old path when the fixed spectrum or Shepard pitch cannot be represented."""
    used=set(v.used)
    try:
        kind=v.integer(prefix+'spectral_morph_type',0,16)
        if kind not in STATIC_MORPHS: return False
        mods=active_modulations(v);into=amount_modulators(mods)
        active={i:m for i,m in mods.items() if i not in v.folded and (abs(m.amount)>1e-12 or into.get(i))}
        moved={m.target for m in active.values()}
        if any(prefix+key in moved for key in ('spectral_morph_amount','spectral_morph_type','spectral_morph_phase')):
            return False
        if abs(v.get(prefix+'spectral_morph_phase')-.5)>1e-6: return False
        if (v.get(prefix+'unison_voices')>1 or prefix+'unison_voices' in moved) and any(
                abs(v.get(prefix+key))>1e-9 or prefix+key in moved
                for key in ('frame_spread','spectral_morph_spread')):
            return False
        if kind == 10:
            original_moved={m.target for i,m in mods.items() if abs(m.amount)>1e-12 or into.get(i)}
            if any(prefix+key in original_moved for key in ('spectral_morph_amount','spectral_morph_type',
                    'spectral_morph_phase','wave_frame','transpose','tune','unison_voices')):
                return False
            if (v.get(prefix+'wave_frame')!=0 or v.get(prefix+'distortion_type')!=0 or
                    v.get(prefix+'spectral_morph_spread')!=0 or v.get(prefix+'frame_spread')!=0 or
                    any(prefix+key in moved for key in ('wave_frame','transpose','tune','unison_voices'))):
                return False
        return True
    finally:
        v.used.clear();v.used.update(used)


def selected_positions(frames: np.ndarray) -> np.ndarray:
    """Map the fully processed Vital table to the existing Serum frame layout."""
    return frames if len(frames)==1 else frames[list(range(NUM_POSITIONS-2))+[NUM_POSITIONS-1]]


def spectral_frames(frames: np.ndarray, kind: int, amount: float) -> np.ndarray:
    """Morph all original frames before choosing the exported positions."""
    spectrum=TableSpectrum.from_frames(frames)
    return np.stack([spectrum.frame(kind,amount,index) for index in range(len(frames))])

def sin_approx(cycles):
    p = 0.5 - np.mod(cycles, 1.0)
    q = p * (8.0 - 16.0 * np.abs(p))
    return q * (0.776 + 0.224 * np.abs(q))


def seeded_random_values():
    # Match libc++'s one-word float canonical, as the WaveSource renderer does.
    gen = _MT19937(4)
    count = 4 * (((16 + 1) * (HARMONICS + 1)) // 4)
    out = np.empty(count, dtype=np.float32)
    for i in range(count):
        u = np.float32(np.float32(gen()) / np.float32(4294967296.0))
        out[i] = np.float32(np.float32(2.0) * u - np.float32(1.0))
    return out.astype(np.float64)


RANDOM_VALUES = seeded_random_values()


@dataclass
class TableSpectrum:
    amplitudes: np.ndarray
    normalized: np.ndarray
    phases: np.ndarray
    time_frames: np.ndarray

    @classmethod
    def from_frames(cls, frames, phase_defaults=True):
        frames = np.asarray(frames, dtype=np.float64).reshape(-1, SIZE)
        coeff = np.fft.rfft(frames, axis=-1)
        amplitudes = np.abs(coeff)
        phases = np.angle(coeff)
        normalized = np.exp(1j * phases)
        if phase_defaults:
            # Interpolate Cartesian normalized components; do not normalize again.
            for k in range(HARMONICS):
                good = np.flatnonzero(amplitudes[:, k] > 0.1)
                last, z = -1, 1j
                for w in good:
                    if last < 0:
                        last, z = 0, normalized[w, k]
                    if w > last + 1:
                        t = np.arange(1, w - last) / (w - last)
                        normalized[last + 1:w, k] = z + t * (normalized[w, k] - z)
                    last, z = int(w), normalized[w, k]
                normalized[last + 1:, k] = z
        return cls(amplitudes, normalized, phases, frames)

    def coefficients(self, kind, amount, frame_index=0, last_harmonic=1023):
        if kind not in TYPES:
            raise ValueError('unsupported spectral type')
        if not 0 <= amount <= 1:
            raise ValueError('amount must be bounded')
        j = min(max(int(frame_index), 0), len(self.amplitudes) - 1)
        a, z = self.amplitudes[j], self.normalized[j]
        h = min(max(int(last_harmonic), 0), HARMONICS - 1)
        out = np.zeros(HARMONICS, dtype=np.complex128)
        k = np.arange(1, h + 1)
        if kind == 2:
            shift = 2.0 ** (-4.0 * (amount - 0.5))
            last = min(h, int(SIZE / (2.0 * shift)))
            k = np.arange(1, last + 1)
            x = np.maximum(1.0, k * shift)
            l = np.floor(x).astype(int)
            l -= (k + l) % 2
            t = (x - l) * 0.5
            coeff = np.pad(a * z, (0, 4))
            out[k] = shift * ((1.0 - t) * coeff[l] + t * coeff[l + 2])
            out[0] = a[0] * z[0]
        elif kind == 5:
            s = 1.0 - (1.0 - amount) ** 3
            b = np.zeros(HARMONICS)
            b[:2] = a[:2] * (1.0 - s)
            carry = b[:2].copy()
            # Four float lanes contain two duplicated harmonic amplitudes.
            for g in range(1, h // 2 + 1):
                stop = min(2 * g + 2, HARMONICS)
                take = stop - 2 * g
                b[2 * g:stop] = (1.0 - s) * a[2 * g:stop] + s * carry[:take]
                carry = b[2 * g:stop] * (g + 0.25) / g
            out = b * z
        elif kind == 6:
            s = 15.0 * amount
            stage = min(int(s), 14)
            t = s - stage
            # Integer division is in SIMD groups, not one stage per harmonic array.
            offset1 = 4 * ((stage * HARMONICS) // 4)
            offset2 = 4 * (((stage + 1) * HARMONICS) // 4)
            bins = np.arange(HARMONICS)
            r1 = RANDOM_VALUES[offset1 + 2 * bins]
            r2 = RANDOM_VALUES[offset2 + 2 * bins]
            g1 = (1.0 + s) * np.maximum(1.0 - s - s * r1, 0.0)
            g2 = (1.0 + s) * np.maximum(1.0 - s - s * r2, 0.0)
            out = np.minimum(((1.0 - t) * g1 + t * g2) * a, 1024.0) * z
        elif kind == 9:
            d = -(2.0 * amount - 1.0) * 0.05
            bins = np.arange(HARMONICS)
            theta = ((bins - 24.0) ** 2 - 23.0 ** 2) * d / (2.0 * np.pi)
            rotate = sin_approx(theta + 0.25) + 1j * sin_approx(theta)
            out = a * z * rotate
        elif kind == 10:
            s = 1.0 - amount
            out = a * z * (1.0 - s)
            k = np.arange(0, h + 1, 2)
            half = k // 2
            amp = (1.0 - s) * a[k] + s * a[half]
            ratio = (a[k] + 0.001) / (a[half] + 0.001)
            n = (1.0 - s) * z[k] + s * z[half]
            close = (ratio > 0.5) & (ratio < 2.0)
            # C++ casts and signed division truncate toward zero.
            phase = self.phases[j]
            delta = (phase[half] - phase[k]) / (2.0 * np.pi)
            wraps = np.trunc((np.trunc(delta) + 1.0) / 2.0)
            cycles = phase[k] / (2.0 * np.pi) + (delta - 2.0 * wraps) * s
            n[close] = sin_approx(cycles[close] + 0.25) + 1j * sin_approx(cycles[close])
            out[k] = amp * n
        else:
            if len(self.amplitudes) == 1:
                out = a * z
            else:
                s = 16.0 * amount ** 2
                q = j / 256.0 + s * np.log2(k) / 11.0
                pos = 256.0 * (1.0 - np.abs(1.0 - 2.0 * np.mod(q / 2.0, 1.0)))
                frm = np.minimum(pos.astype(int), len(self.amplitudes) - 2)
                t = np.minimum(1.0, pos - frm)
                amp = self.amplitudes[frm, k] + t * (self.amplitudes[frm + 1, k] - self.amplitudes[frm, k])
                norm = self.normalized[frm, k] + t * (self.normalized[frm + 1, k] - self.normalized[frm, k])
                out[k] = amp * norm
                out[0] = a[0] * z[0]
        # Grouped morphs process both harmonics in the final SIMD group.
        grouped = kind in (5, 6, 9, 10)
        end = min(HARMONICS, (h // 2 + 1) * 2) if grouped else h + 1
        out[end:] = 0.0
        return out

    def frame(self, kind, amount, frame_index=0, last_harmonic=1023):
        return np.fft.irfft(self.coefficients(kind, amount, frame_index, last_harmonic), n=SIZE)


