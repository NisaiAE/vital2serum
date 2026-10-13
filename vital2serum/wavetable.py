"""Render Vital wavetable JSON the way Vital's WavetableCreator does when it loads a preset.

Port of mtytel/vital (Copyright 2013-2019 Matt Tytel, GPL-3.0-or-later), made and changed in 2026,
from src/common/wavetable/*: wavetable_creator.cpp,
wavetable_group.cpp, wavetable_component.cpp, wavetable_keyframe.cpp, wave_source.cpp,
wave_line_source.cpp (+ src/common/line_generator.cpp), file_source.cpp, wave_warp_modifier.cpp,
wave_fold_modifier.cpp, frequency_filter_modifier.cpp, phase_modifier.cpp, wave_window_modifier.cpp.
Shepard Tone Source and Slew Limiter raise ConversionError, so nothing is silently dropped.

A frame is kept as 2048 time-domain samples. Every Vital component leaves its frame with matching
time and frequency data, so the spectrum is recomputed with an unscaled real FFT when needed
(WaveFrame::toFrequencyDomain == numpy.fft.rfft; toTimeDomain == numpy.fft.irfft, which like Vital
ignores the imaginary parts of the DC and Nyquist bins).

Vital renders kNumOscillatorWaveFrames = 257 positions (0..256) and its oscillator plays frame
int(wave_frame) with no morphing between frames, holding the last keyframe beyond it. Serum 2 plays
frame floor((table_pos - 1) / 255 * num_frames) of a table (measured on 2.1.5), also without morphing.
A 256-frame table holding Vital positions 0..254 and 256 therefore reproduces every integer
wave_frame except 255 (which plays position 256); see serum_table_position().

Vital plays frames from their spectra, so remove_all_dc (which zeroes the DC bin) is emulated by
subtracting each frame's mean. full_normalize's span is measured on the time data before that,
exactly as WavetableCreator::render(int) does.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import base64
import binascii
import math
import numpy as np
from .errors import ConversionError

FRAME_SIZE = 2048
NUM_POSITIONS = 257          # vital::kNumOscillatorWaveFrames
OUTPUT_FRAMES = 256          # Serum maximum
MAX_KEYFRAMES = 257
MAX_FILE_SAMPLES = 8 * 176400  # Vital saves at most ~176400 samples; allow headroom
KNOWN_TYPES = ('Wave Source', 'Line Source', 'Audio File Source', 'Shepard Tone Source', 'Phase Shift',
               'Wave Window', 'Frequency Filter', 'Slew Limiter', 'Wave Folder', 'Wave Warp')
_BINS = np.arange(FRAME_SIZE // 2 + 1)
_INDEX = np.arange(FRAME_SIZE)


# --------------------------------------------------------------------------- helpers
def _num(value, name: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConversionError(f'{name} must be a finite number')
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ConversionError(f'{name} out of range: {value}')
    return float(value)


def _flag(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ConversionError(f'{name} must be a boolean')


def _enum(value, name: str, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or not 0 <= value < count:
        raise ConversionError(f'{name} must be an integer from 0 to {count - 1}')
    return int(value)


def _lerp(a, b, t):
    return a + t * (b - a)


def _cubic_tween(prev, frm, to, nxt, range_prev: float, rng: float, range_next: float, t: float):
    """WavetableKeyframe::cubicTween (works on numpy arrays and scalars)."""
    slope_from = (to - prev) / (1.0 + range_prev / rng) if range_prev > 0.0 else 0.0
    slope_to = (nxt - frm) / (1.0 + range_next / rng) if range_next > 0.0 else 0.0
    delta = to - frm
    movement = _lerp(frm, to, t)
    smooth = t * (1.0 - t) * ((1.0 - t) * (slope_from - delta) + t * (delta - slope_to))
    return movement + smooth


def _power_scale(value: float, power: float) -> float:
    """futils::powerScale (value >= 0 callers)."""
    if abs(power) < 0.01:
        return value
    return (math.exp(power * value) - 1.0) / (math.exp(power) - 1.0)


def _signed_power_scale(values, power: float):
    """highResPowerScale / frequency_filter powerScale: odd-symmetric power curve."""
    values = np.asarray(values, dtype=np.float64)
    if abs(power) < 0.01:
        return values
    magnitude = (np.exp(power * np.abs(values)) - 1.0) / (math.exp(power) - 1.0)
    return np.where(values >= 0.0, magnitude, -magnitude)


def _smooth_transition(t: float) -> float:
    return 0.5 * math.sin((t - 0.5) * math.pi) + 0.5


def _spectrum(frame):
    return np.fft.rfft(frame)


def _time(spectrum):
    return np.fft.irfft(spectrum, n=FRAME_SIZE)


def _frequency_lerp(fa, fb, t: float):
    """WaveSourceKeyframe::linearFrequencyInterpolate on two spectra; returns a time frame."""
    amp_a, amp_b = np.sqrt(np.abs(fa)), np.sqrt(np.abs(fb))
    amp = _lerp(amp_a, amp_b, t) ** 2
    phase = np.angle(fa) + t * np.angle(np.conj(fa) * fb)
    phase = np.where(amp_a == 0, np.angle(fb), phase)
    out = amp * np.exp(1j * phase)
    out[0] = _lerp(fa[0].real, fb[0].real, t)
    out[-1] = _lerp(fa[-1].real, fb[-1].real, t)
    return _time(out)


def _decode_b64(text, limit: int, feature: str) -> bytes:
    if not isinstance(text, str) or len(text) > limit * 2:
        raise ConversionError(f'{feature}: invalid/excessive base64 data')
    try:
        return base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ConversionError(f'{feature}: malformed base64 data') from exc


class _MT19937:
    """std::mt19937 (used by vital::utils::RandomGenerator)."""
    def __init__(self, seed: int):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xffffffff
        for i in range(1, 624):
            prev = self.mt[i - 1]
            self.mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xffffffff
        self.index = 624

    def __call__(self) -> int:
        if self.index >= 624:
            mt = self.mt
            for i in range(624):
                y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff)
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908b0df if y & 1 else 0)
            self.index = 0
        y = self.mt[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        return y


def _random_phases(seed: int, count: int) -> np.ndarray:
    """uniform_real_distribution<float>(-kPi, kPi) over std::mt19937 as built with libc++ (macOS)."""
    engine = _MT19937(seed)
    low, high = np.float32(-math.pi), np.float32(math.pi)
    span = np.float32(high - low)
    two32 = np.float32(4294967296.0)
    out = np.empty(count, dtype=np.float32)
    for i in range(count):
        canonical = np.float32(np.float32(engine()) / two32)
        out[i] = np.float32(span * canonical + low)
    return out.astype(np.float64)


# --------------------------------------------------------------------------- components
class _Component:
    """Keyframe bookkeeping shared by all components (WavetableComponent::interpolate)."""
    is_source = False

    def __init__(self, data: dict, feature: str, notes: list[str]):
        self.feature = feature
        self.notes = notes
        style = data.get('interpolation_style', 1)
        self.style = _enum(style, f'{feature}: interpolation_style', 3)
        keys = data.get('keyframes')
        if not isinstance(keys, list) or not 1 <= len(keys) <= MAX_KEYFRAMES:
            raise ConversionError(f'{feature}: expected 1..{MAX_KEYFRAMES} wavetable keyframes')
        parsed = []
        for key in keys:
            if not isinstance(key, dict):
                raise ConversionError(f'{feature}: malformed wavetable keyframe')
            pos = key.get('position', 0)
            if isinstance(pos, bool) or not isinstance(pos, (int, float)) or not math.isfinite(pos) \
                    or not 0 <= pos <= NUM_POSITIONS - 1 or pos != int(pos):
                raise ConversionError(f'{feature}: keyframe position must be an integer in 0..{NUM_POSITIONS - 1}')
            parsed.append((int(pos), self.parse_keyframe(key)))
        parsed.sort(key=lambda item: item[0])      # Vital inserts by position; ties keep file order
        self.positions = [p for p, _ in parsed]
        self.keyframes = [k for _, k in parsed]
        if self.style == 2 and not isinstance(self, WaveSource) and len(self.keyframes) > 1:
            notes.append(f'{feature}: cubic keyframe interpolation is only defined for Wave Source; rendered as linear.')

    def parse_keyframe(self, key: dict):
        raise NotImplementedError

    def locate(self, position: float):
        n = len(self.keyframes)
        index = sum(1 for p in self.positions if int(position) >= p) - 1   # getIndexFromPosition - 1
        if index < 0 or index >= n - 1 or self.style == 0:
            return ('copy', min(max(index, 0), n - 1))
        from_pos, to_pos = self.positions[index], self.positions[index + 1]
        t = (position - from_pos) / (to_pos - from_pos)
        if self.style == 1:
            return ('linear', index, t)
        next_index = index + 2 if index + 2 < n else index
        prev_index = index - 1 if index - 1 >= 0 else index + 1
        return ('cubic', index, t, prev_index, next_index,
                float(from_pos - self.positions[prev_index]), float(to_pos - from_pos),
                float(self.positions[next_index] - to_pos))

    def params(self, position: float) -> dict:
        """Keyframe parameters (dicts of floats) interpolated linearly, as the modifiers do."""
        where = self.locate(position)
        if where[0] == 'copy':
            return self.keyframes[where[1]]
        a, b, t = self.keyframes[where[1]], self.keyframes[where[1] + 1], where[2]
        return {k: _lerp(a[k], b[k], t) for k in a}

    def render(self, frame: np.ndarray, position: float) -> np.ndarray:
        raise NotImplementedError


class WaveSource(_Component):
    is_source = True

    def __init__(self, data, feature, notes):
        self.frequency_mode = _enum(data.get('interpolation', 0), f'{feature}: interpolation', 2) == 1
        super().__init__(data, feature, notes)

    def parse_keyframe(self, key):
        raw = _decode_b64(key.get('wave_data'), FRAME_SIZE * 4, self.feature)
        if len(raw) != FRAME_SIZE * 4:
            raise ConversionError(f'{self.feature}: a Wave Source frame must contain {FRAME_SIZE} float32 samples')
        values = np.frombuffer(raw, dtype='<f4').astype(np.float64)
        if not np.all(np.isfinite(values)) or np.abs(values).max() >= 1e6:
            raise ConversionError(f'{self.feature}: invalid/non-finite wavetable samples')
        return values

    def render(self, frame, position):
        where = self.locate(position)
        if where[0] == 'copy':
            return self.keyframes[where[1]]
        index, t = where[1], where[2]
        a, b = self.keyframes[index], self.keyframes[index + 1]
        if where[0] == 'linear':
            if not self.frequency_mode:
                return _lerp(a, b, t)
            return _frequency_lerp(_spectrum(a), _spectrum(b), t)
        prev, nxt = self.keyframes[where[3]], self.keyframes[where[4]]
        rp, rng, rn = where[5], where[6], where[7]
        if not self.frequency_mode:
            return _cubic_tween(prev, a, b, nxt, rp, rng, rn, t)
        fp, ff, ft, fn = (_spectrum(x) for x in (prev, a, b, nxt))       # cubicFrequencyInterpolate
        ap, af, at, an = (np.sqrt(np.abs(x)) for x in (fp, ff, ft, fn))
        amp = _cubic_tween(ap, af, at, an, rp, rng, rn, t) ** 2
        phase_prev = np.angle(fp)
        phase_from = phase_prev + np.where(af != 0, np.angle(np.conj(fp) * ff), 0.0)
        phase_to = phase_from + np.where(at != 0, np.angle(np.conj(ff) * ft), 0.0)
        phase_next = phase_to + np.where(an != 0, np.angle(np.conj(ft) * fn), 0.0)
        phase = _cubic_tween(phase_prev, phase_from, phase_to, phase_next, rp, rng, rn, t)
        out = amp * np.exp(1j * phase)
        out[0] = _cubic_tween(fp[0].real, ff[0].real, ft[0].real, fn[0].real, rp, rng, rn, t)
        out[-1] = _cubic_tween(fp[-1].real, ff[-1].real, ft[-1].real, fn[-1].real, rp, rng, rn, t)
        return _time(out)


class LineGenerator:
    """LineGenerator::render for a non-looping line (WaveLineSourceKeyframe never loops)."""
    MAX_POINTS = 100

    def __init__(self, data: dict, feature: str):
        if not isinstance(data, dict):
            raise ConversionError(f'{feature}: Line Source keyframe has no line')
        n = data.get('num_points')
        points = data.get('points', [])
        powers = data.get('powers', [])
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= self.MAX_POINTS:
            raise ConversionError(f'{feature}: line num_points must be 1..{self.MAX_POINTS}')
        if not isinstance(points, list) or not isinstance(powers, list) or len(powers) < n or len(points) < 2 * n:
            raise ConversionError(f'{feature}: malformed line points/powers')
        self.points = [(_num(points[2 * i], 'line x'), _num(points[2 * i + 1], 'line y')) for i in range(n)]
        self.powers = [_num(powers[i], 'line power') for i in range(n)]
        self.smooth = bool(data.get('smooth', False))

    @classmethod
    def blend(cls, a: 'LineGenerator', b: 'LineGenerator', t: float) -> 'LineGenerator':
        gen = cls.__new__(cls)
        n = min(len(a.points), len(b.points))
        gen.points = [(_lerp(a.points[i][0], b.points[i][0], t), _lerp(a.points[i][1], b.points[i][1], t))
                      for i in range(n)]
        gen.powers = [_lerp(a.powers[i], b.powers[i], t) for i in range(n)]
        gen.smooth = a.smooth
        return gen

    def render(self, resolution: int = FRAME_SIZE) -> np.ndarray:
        n = len(self.points)
        out = np.empty(resolution)
        point_index = 0
        last = self.points[0]
        current = self.points[0]
        current_power = 0.0
        for i in range(resolution):
            x = i / (resolution - 1.0)
            t = 1.0
            if current[0] > last[0]:
                t = (x - last[0]) / (current[0] - last[0])
            if self.smooth:
                t = _smooth_transition(t)
            t = min(1.0, max(0.0, _power_scale(t, current_power)))
            y = last[1] + t * (current[1] - last[1])
            out[i] = 1.0 - y
            while x > current[0] and point_index < n:
                current_power = self.powers[point_index % n]
                point_index += 1
                last = current
                current = self.points[point_index % n]
                if point_index >= n:
                    current = (current[0] + 1.0, current[1])
                    break
        return out


class LineSource(_Component):
    is_source = True

    def parse_keyframe(self, key):
        return (LineGenerator(key.get('line'), self.feature), _num(key.get('pull_power', 0.0), 'pull_power'))

    def render(self, frame, position):
        where = self.locate(position)
        if where[0] == 'copy':
            line = self.keyframes[where[1]][0]
        else:
            (la, pa), (lb, pb) = self.keyframes[where[1]], self.keyframes[where[1] + 1]
            line = LineGenerator.blend(la, lb, _power_scale(where[2], pa - pb))
        return line.render() * 2.0 - 1.0                    # WaveLineSourceKeyframe::render


class FileSource(_Component):
    """'Audio File Source' (file_source.cpp)."""
    is_source = True

    def __init__(self, data, feature, notes):
        self.window_size = _num(data.get('window_size'), f'{feature}: window_size', 1e-3, 1e7)
        self.fade_style = _enum(data.get('fade_style', 0), f'{feature}: fade_style', 4)
        self.phase_style = _enum(data.get('phase_style', 0), f'{feature}: phase_style', 3)
        self.normalize_gain = _flag(data.get('normalize_gain', False), f'{feature}: normalize_gain')
        self.normalize_mult = _flag(data.get('normalize_mult', True), f'{feature}: normalize_mult')
        raw = _decode_b64(data.get('audio_file', ''), MAX_FILE_SAMPLES * 2, feature)
        pcm = np.frombuffer(raw[:len(raw) // 2 * 2], dtype='<i2')
        audio = pcm.astype(np.float32) * np.float32(1.0 / 32767.0)
        self.size = int(audio.size)
        # loadBuffer: one leading and three trailing copies for Catmull-Rom reads
        if self.size:
            self.buffer = np.concatenate(([audio[0]], audio, [audio[-1]] * 3)).astype(np.float64)
        self.phases = None
        if self.phase_style == 1:        # kClear
            self.phases = np.where(_BINS % 2 == 0, -0.5 * math.pi, 0.5 * math.pi)
        elif self.phase_style == 2:      # kVocode
            seed = data.get('random_seed', 0)
            if isinstance(seed, bool) or not isinstance(seed, (int, float)):
                raise ConversionError(f'{feature}: random_seed must be a number')
            self.phases = _random_phases(int(seed), FRAME_SIZE)[:_BINS.size]
        super().__init__(data, feature, notes)

    def parse_keyframe(self, key):
        return {'start_position': _num(key.get('start_position', 0.0), 'start_position'),
                'window_fade': _num(key.get('window_fade', 1.0), 'window_fade')}

    def _sample(self, positions):
        """getScaledInterpolatedSample: Catmull-Rom over the padded buffer."""
        pos = np.clip(positions, 0.0, self.size - 1)
        start = pos.astype(np.int64)
        t = pos - start
        half_t = t * 0.5
        half_t2 = t * half_t
        half_t3 = half_t2 * t
        half_three_t3 = half_t3 * 3.0
        b = self.buffer
        return ((half_t2 * 2.0 - half_t3 - half_t) * b[start]
                + (half_three_t3 - half_t2 * 5.0 + 1.0) * b[start + 1]
                + (half_t + half_t2 * 4.0 - half_three_t3) * b[start + 2]
                + (half_t3 - half_t2) * b[start + 3])

    def render(self, frame, position):
        if not self.size:
            return np.zeros(FRAME_SIZE)
        p = self.params(position)
        start_position, window_fade = p['start_position'], p['window_fade']
        ws = self.window_size
        t = _INDEX / FRAME_SIZE
        if self.fade_style == 0:                               # kWaveBlend
            ratio = ws / FRAME_SIZE
            start_index = int(start_position / ratio + ws / 2.0 + FRAME_SIZE // 2) % FRAME_SIZE
            out = np.empty(FRAME_SIZE)
            out[(start_index + _INDEX) % FRAME_SIZE] = self._sample(start_position + t * ws)
            fade_samples = int(window_fade * FRAME_SIZE)
            if fade_samples > 0:
                j = np.arange(fade_samples)
                tt = j / (fade_samples - 1.0) if fade_samples > 1 else np.zeros(1)
                fade = 0.5 + 0.5 * np.cos(math.pi * tt)
                values = self._sample(start_position + ws + tt * (fade_samples * ratio))
                write = (start_index + j) % FRAME_SIZE
                if fade_samples <= FRAME_SIZE:
                    out[write] = _lerp(out[write], values, fade)
                else:
                    for w, v, f in zip(write, values, fade):
                        out[w] = _lerp(out[w], v, f)
        else:
            cycles_in = start_position / ws
            from_cycle = int(cycles_in)
            transition = cycles_in - from_cycle
            first = self._sample(from_cycle * ws + t * ws)
            if self.fade_style == 1:                           # kNoInterpolate
                out = first
            else:
                second = self._sample((from_cycle + 1) * ws + t * ws)
                if self.fade_style == 2:                       # kTimeInterpolate
                    out = _lerp(first, second, transition)
                else:                                          # kFreqInterpolate
                    out = _frequency_lerp(_spectrum(first), _spectrum(second), transition)
        if self.phases is not None:
            out = _time(np.abs(_spectrum(out)) * np.exp(1j * self.phases))
        if self.normalize_mult:
            peak = float(np.abs(out).max())
            out = out * (1.0 / max(1e-7 if self.normalize_gain else 1.0, peak))
        return out


class WaveWarp(_Component):
    def __init__(self, data, feature, notes):
        self.h_asym = _flag(data.get('horizontal_asymmetric', False), f'{feature}: horizontal_asymmetric')
        self.v_asym = _flag(data.get('vertical_asymmetric', False), f'{feature}: vertical_asymmetric')
        super().__init__(data, feature, notes)

    def parse_keyframe(self, key):
        return {'h': _num(key.get('horizontal_power', 0.0), 'horizontal_power'),
                'v': _num(key.get('vertical_power', 0.0), 'vertical_power')}

    def render(self, frame, position):
        p = self.params(position)
        horizontal = _INDEX / (FRAME_SIZE - 1.0)
        if self.h_asym:
            warped = _signed_power_scale(horizontal, p['h'])
        else:
            warped = 0.5 * _signed_power_scale(2.0 * horizontal - 1.0, p['h']) + 0.5
        float_index = (FRAME_SIZE - 1) * warped
        index = np.clip(np.trunc(float_index).astype(np.int64), 0, FRAME_SIZE - 2)
        vertical = np.clip(_lerp(frame[index], frame[index + 1], float_index - index), -1.0, 1.0)
        if self.v_asym:
            return 2.0 * _signed_power_scale(0.5 * vertical + 0.5, p['v']) - 1.0
        return _signed_power_scale(vertical, p['v'])


class WaveFold(_Component):
    def parse_keyframe(self, key):
        return {'boost': _num(key.get('fold_boost', 1.0), 'fold_boost')}

    def render(self, frame, position):
        boost = self.params(position)['boost']
        max_value = max(1.0, float(np.abs(frame).max()))
        value = np.clip(frame / max_value, -1.0, 1.0)
        return np.sin(max_value * boost * np.arcsin(value))


class FrequencyFilter(_Component):
    def __init__(self, data, feature, notes):
        self.filter_style = _enum(data.get('style', 0), f'{feature}: style', 4)
        self.normalize = _flag(data.get('normalize', True), f'{feature}: normalize')
        super().__init__(data, feature, notes)

    def parse_keyframe(self, key):
        return {'cutoff': _num(key.get('cutoff', 4.0), 'cutoff'), 'shape': _num(key.get('shape', 0.5), 'shape')}

    def render(self, frame, position):
        p = self.params(position)
        cutoff_index = 2.0 ** p['cutoff']
        delta = _BINS - cutoff_index
        slope = 1.0 / _lerp(1.0, 128.0, p['shape'] * p['shape'])
        if self.filter_style == 0:
            mult = np.clip(1.0 - slope * delta, 0.0, 1.0)
        elif self.filter_style == 1:
            mult = np.clip(1.0 - np.abs(slope * delta), 0.0, 1.0)
        elif self.filter_style == 2:
            mult = np.clip(1.0 + slope * delta, 0.0, 1.0)
        else:
            phase = _BINS / (cutoff_index * 2.0)
            mult = 2.0 * _signed_power_scale(1.0 - np.abs(2.0 * (phase - np.floor(phase)) - 1.0),
                                             _lerp(-9.0, 9.0, p['shape']))
        out = _time(_spectrum(frame) * mult)
        if self.normalize:
            out = out * (1.0 / max(1e-7, float(np.abs(out).max())))
        return out


class PhaseShift(_Component):
    def __init__(self, data, feature, notes):
        self.phase_style = _enum(data.get('style', 0), f'{feature}: style', 5)
        super().__init__(data, feature, notes)

    def parse_keyframe(self, key):
        return {'phase': _num(key.get('phase', 0.0), 'phase'), 'mix': _num(key.get('mix', 1.0), 'mix')}

    def render(self, frame, position):
        p = self.params(position)
        spec = _spectrum(frame)
        mix = p['mix']
        if self.phase_style == 4:                              # kClear
            return _time(np.abs(spec).astype(np.complex128))
        shift = -p['phase']
        if self.phase_style == 0:                              # kNormal
            angle = shift * _BINS
        elif self.phase_style == 1:                            # kEvenOdd
            angle = np.where(_BINS % 2 == 0, shift * _BINS, -shift * _BINS)
        elif self.phase_style == 2:                            # kHarmonic
            angle = np.full(_BINS.size, shift)
        else:                                                  # kHarmonicEvenOdd
            angle = np.where(_BINS % 2 == 0, shift, -shift)
        rotated = spec * np.exp(1j * angle)
        return _time(mix * rotated + (1.0 - mix) * spec)


class WaveWindow(_Component):
    def __init__(self, data, feature, notes):
        self.shape = _enum(data.get('window_shape', 0), f'{feature}: window_shape', 5)
        super().__init__(data, feature, notes)

    def parse_keyframe(self, key):
        return {'left': _num(key.get('left_position', 0.25), 'left_position'),
                'right': _num(key.get('right_position', 0.75), 'right_position')}

    def _window(self, t):
        if self.shape == 0:
            return 0.5 - 0.5 * np.cos(math.pi * t)
        if self.shape == 1:
            return np.sin(math.pi * t / 2.0)
        if self.shape == 3:
            return np.where(t < 1.0, 0.0, 1.0)
        if self.shape == 4:
            return t * np.cos(math.pi * (t * 1.5 + 0.5))
        return t

    def render(self, frame, position):
        p = self.params(position)
        out = frame.copy()
        t = _INDEX / (FRAME_SIZE - 1.0)
        left = t < p['left']
        if left.any():
            out[left] *= self._window(t[left] / p['left'])
        right = t > p['right']
        if right.any():
            out[right] *= self._window((1.0 - t[right]) / (1.0 - p['right']))
        return out


_FACTORY = {'Wave Source': WaveSource, 'Line Source': LineSource, 'Audio File Source': FileSource,
            'Wave Warp': WaveWarp, 'Wave Folder': WaveFold, 'Frequency Filter': FrequencyFilter,
            'Phase Shift': PhaseShift, 'Wave Window': WaveWindow}


def _component(data: dict, feature: str, notes: list[str]) -> _Component:
    if not isinstance(data, dict):
        raise ConversionError(f'{feature}: malformed wavetable component')
    kind = data.get('type')
    if kind in _FACTORY:
        return _FACTORY[kind](data, feature, notes)
    if kind in KNOWN_TYPES:
        raise ConversionError(f"{feature}: wavetable component '{kind}' is not implemented; "
                              "export the rendered table from Vital (or use --allow-placeholder)")
    raise ConversionError(f'{feature}: unknown wavetable component type {kind!r}')


def _version_tuple(text: str) -> tuple:
    parts = []
    for piece in text.split('.'):
        digits = ''.join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def serum_table_position(wave_frame: float, frames: int) -> float:
    """Serum kParamTablePos (1..256) that plays the frame rendered for Vital's int(wave_frame)."""
    if frames <= 1:
        return 1.0
    index = min(int(max(wave_frame, 0.0)), OUTPUT_FRAMES - 1)
    # Serum frame = floor((pos - 1) / 255 * 256); aim at the middle of the frame's span.
    return min(256.0, 1.0 + 255.0 * (index + 0.5) / OUTPUT_FRAMES)


def render_wavetable(table: dict, feature: str = 'wavetable', *, full: bool = False) -> tuple[np.ndarray, list[str]]:
    """Return (frames[OUTPUT_FRAMES or 1, 2048] float32, notes) for a Vital wavetable JSON object."""
    if not isinstance(table, dict):
        raise ConversionError(f'{feature}: no embedded wavetable data')
    notes: list[str] = []
    if all(k in table for k in ('num_points', 'points', 'powers')):
        # A bare LineGenerator used as a wavetable: initFromLineGenerator() after clear().
        table = {'groups': [{'components': [{'type': 'Line Source', 'interpolation_style': 1,
                                              'keyframes': [{'position': 0, 'line': table}]}]}],
                 'remove_all_dc': True, 'full_normalize': True, 'version': '1.0.0'}
    version = table.get('version', '')
    if isinstance(version, str) and version and _version_tuple(version) < (0, 7, 7):
        raise ConversionError(f'{feature}: wavetable saved by Vital {version} uses a pre-0.7.7 format; re-save the preset in Vital 1.0.x first')
    groups = table.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ConversionError(f'{feature}: wavetable has no groups')
    parsed_groups = []
    for g, group in enumerate(groups):
        if not isinstance(group, dict) or not isinstance(group.get('components'), list) or not group['components']:
            raise ConversionError(f'{feature}: wavetable group {g} has no components')
        comps = [_component(c, f'{feature} group {g} component {i}', notes) for i, c in enumerate(group['components'])]
        if not comps[0].is_source:
            notes.append(f'{feature}: group {g} starts with a modifier; it is applied to silence.')
        parsed_groups.append(comps)
    # WavetableCreator::clear() sets both flags; jsonToState keeps remove_all_dc unless given
    # and turns full_normalize off when the key is missing.
    remove_dc = bool(table.get('remove_all_dc', True))
    normalize = bool(table.get('full_normalize', False))
    positions = list(range(NUM_POSITIONS)) if full else list(range(OUTPUT_FRAMES - 1)) + [NUM_POSITIONS - 1]
    frames = np.zeros((len(positions), FRAME_SIZE))
    for k, position in enumerate(positions):
        combined = np.zeros(FRAME_SIZE)
        for comps in parsed_groups:
            frame = np.zeros(FRAME_SIZE)
            for comp in comps:
                frame = comp.render(frame, position)
            combined += frame
        if len(parsed_groups) > 1:
            combined /= len(parsed_groups)
        frames[k] = combined
    if not np.all(np.isfinite(frames)):
        raise ConversionError(f'{feature}: rendering produced non-finite samples')
    spans = np.maximum(frames.max(axis=1), 0.0) - np.minimum(frames.min(axis=1), 0.0)   # before DC removal
    if remove_dc:
        frames -= frames.mean(axis=1, keepdims=True)
    if normalize:
        max_span = float(spans.max())
        if max_span > 0.0:
            frames *= 2.0 / max_span
    frames = frames.astype(np.float32)
    if np.all(frames == frames[0]):
        frames = frames[:1]
    return frames, list(dict.fromkeys(notes))

# --- spectral morph ------------------------------------------------------------------------------
# Vital reshapes the harmonics of the playing frame before its time-domain warps. With a fixed
# amount that is a fixed table, so it can be written into the frames Serum plays. The laws below
# were read from Vital 1.0.7 with probe tables; no level change is applied by
# any of them.
MORPH_HARMONIC_STRETCH, MORPH_INHARMONIC_STRETCH, MORPH_LOW_PASS, MORPH_HIGH_PASS = 3, 4, 7, 8
MORPH_PHASE_DISPERSE = 9          # moves phases only; the harmonic levels stay as they are
BAKED_MORPHS = {MORPH_HARMONIC_STRETCH: 'harmonic stretch', MORPH_INHARMONIC_STRETCH: 'inharmonic stretch',
                MORPH_LOW_PASS: 'low pass', MORPH_HIGH_PASS: 'high pass'}


def spectral_morph(frames: np.ndarray, kind: int, amount: float) -> np.ndarray:
    """Apply Vital's spectral morph `kind` at a fixed `amount` (0..1) to time-domain frames.

    low pass      harmonics up to 2^(10 a) pass, the next one in proportion to the fraction
    high pass     harmonics below 2^(10 a) * 1025/1024 are removed, the one at the edge in part
    harmonic      harmonic k moves to 1 + (k - 1) * 2^(8 (a - 0.5))
    inharmonic    harmonic k moves to 1 + (k - 1) * k^(2.4 (a - 0.5))
    A moved harmonic is shared between the two bins it lands between, in proportion.
    """
    if kind not in BAKED_MORPHS:
        raise ConversionError(f'spectral morph {kind} cannot be rendered')
    spectrum = np.fft.rfft(np.asarray(frames, dtype=np.float64), axis=-1)
    harmonic = np.arange(spectrum.shape[-1], dtype=np.float64)
    if kind == MORPH_LOW_PASS:
        spectrum[..., 1:] *= np.clip(2.0 ** (10.0 * amount) + 1.0 - harmonic[1:], 0.0, 1.0)
    elif kind == MORPH_HIGH_PASS:
        spectrum[..., 1:] *= np.clip(harmonic[1:] + 1.0 - 2.0 ** (10.0 * amount) * 1025.0 / 1024.0, 0.0, 1.0)
    else:
        k = harmonic[1:]
        if kind == MORPH_HARMONIC_STRETCH:
            target = 1.0 + (k - 1.0) * 2.0 ** (8.0 * (amount - 0.5))
        else:
            target = 1.0 + (k - 1.0) * k ** (2.4 * (amount - 0.5))
        low = np.floor(target).astype(np.int64)
        share = target - low
        size = spectrum.shape[-1]
        moved = np.zeros_like(spectrum)
        moved[..., 0] = spectrum[..., 0]
        flat_in = spectrum.reshape(-1, size)
        flat_out = moved.reshape(-1, size)
        for row_in, row_out in zip(flat_in, flat_out):
            for index, weight in ((low, 1.0 - share), (low + 1, share)):
                inside = (index >= 1) & (index < size)
                np.add.at(row_out, index[inside], row_in[1:][inside] * weight[inside])
        spectrum = moved
    return np.fft.irfft(spectrum, FRAME_SIZE, axis=-1)


WARP_SYNC, WARP_FORMANT, WARP_QUANTIZE, WARP_BEND, WARP_SQUEEZE, WARP_PULSE_WIDTH = 1, 2, 3, 4, 5, 6
PHASE_WARPS = {WARP_SYNC: 'sync', WARP_FORMANT: 'formant', WARP_QUANTIZE: 'quantize', WARP_BEND: 'bend',
               WARP_SQUEEZE: 'squeeze', WARP_PULSE_WIDTH: 'pulse width'}
WARP_OVERSAMPLING = 8             # the warped wave is evaluated this much finer, then band-limited
SWEEP_FRAMES = OUTPUT_FRAMES


def warp_phase(kind: int, amount: float, phase: float = 0.5, size: int = FRAME_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """(read position 0..1, gain) along one cycle for Vital's oscillator distortion `kind`.

    Vital reads the table at a bent phase; `phase` is its distortion_phase (0.5 = none). Taken
    from Vital's source and checked against one-cycle renders of a ramp table (Vital 1.0.7).
    With x the cycle position and s the same position counted from -0.5 to 0.5 around the table start:
      sync         (x - 0.5) * 2^(8 a - 4) + phase
      formant      the same read, times sin(pi x)
      quantize     round(s d + phase - 0.5) / d, with d = 2^(27.2 (1 - a)^3 + 1) steps per cycle;
                   a step landing half a cycle or more away reads the middle of the table
      bend         a cubic through the cycle whose end slopes are 3a -+ 2(a - a^2); a = 0.5 is straight
      squeeze      the quarter either side of the start is spread over 1/d of it, d = 1.9 a + 0.05
      pulse width  s / (1 - a); silent once that passes half a cycle
    """
    if kind not in PHASE_WARPS:
        raise ConversionError(f'oscillator distortion {kind} is not a phase warp')
    x = np.arange(size, dtype=np.float64) / size
    signed = np.where(x < 0.5, x, x - 1.0)
    shift = phase - 0.5
    gain = np.ones(size)
    if kind in (WARP_SYNC, WARP_FORMANT):
        read = (x - 0.5) * 2.0 ** (8.0 * amount - 4.0) + phase
        if kind == WARP_FORMANT:
            gain = np.sin(np.pi * x)
    elif kind == WARP_QUANTIZE:
        steps = 2.0 ** (32.0 * 0.85 * (1.0 - amount) ** 3 + 1.0)
        read = (np.round(signed * steps + shift) - shift) / steps
        read = np.where(np.abs(read) >= 0.5, -0.5, read)        # Vital's integer phase saturates there
    elif kind == WARP_BEND:
        t = np.mod(signed - shift + 0.5, 1.0)
        offset = 2.0 * (amount - amount * amount)
        end, start = 3.0 * amount + offset, 3.0 * amount - offset
        read = t ** 3 + end * (t ** 2 - t ** 3) + start * (t - 2.0 * t ** 2 + t ** 3) - 0.5 + shift
    elif kind == WARP_SQUEEZE:
        t = signed - shift
        t = t - np.round(t)
        distance = np.abs(t)
        d = 1.9 * amount + 0.05
        read = np.sign(t) * np.where(distance <= 0.25 * d, distance / d, 0.5 - (0.5 - distance) / (2.0 - d)) + shift
    else:
        read = signed / max(1.0 - amount, 2.0 ** -32)
        gain = (np.abs(read) < 0.5).astype(np.float64)
        read = np.clip(read, -0.5, 0.5) + shift
    return np.mod(read, 1.0), gain


def _oversampled(frames: np.ndarray) -> np.ndarray:
    """Frames on a WARP_OVERSAMPLING times finer grid, by band-limited interpolation."""
    spectrum = np.fft.rfft(np.asarray(frames, dtype=np.float64), axis=-1)
    spectrum[..., -1] = 0.0
    size = FRAME_SIZE * WARP_OVERSAMPLING
    return np.fft.irfft(spectrum, size, axis=-1) * WARP_OVERSAMPLING


def _band_limited(fine: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(fine, axis=-1)[..., :FRAME_SIZE // 2 + 1]
    spectrum[..., -1] = 0.0
    return np.fft.irfft(spectrum, FRAME_SIZE, axis=-1) / WARP_OVERSAMPLING


def _read_warped(fine: np.ndarray, kind: int, amount: float, phase: float) -> np.ndarray:
    size = fine.shape[-1]
    read, gain = warp_phase(kind, amount, phase, size)
    position = read * size
    index = np.floor(position).astype(np.int64) % size
    share = position - np.floor(position)
    return (fine[..., index] * (1.0 - share) + fine[..., (index + 1) % size] * share) * gain


def phase_warp(frames: np.ndarray, kind: int, amount: float, phase: float = 0.5) -> np.ndarray:
    """Vital's oscillator distortion `kind` at a fixed `amount`, rendered into each frame."""
    return _band_limited(_read_warped(_oversampled(frames), kind, amount, phase))


def sweep_amounts(count: int = SWEEP_FRAMES) -> np.ndarray:
    """Amount held by each frame of a sweep table: the middle of the span Serum plays it for."""
    return (np.arange(count, dtype=np.float64) + 0.5) / count


def warp_sweep(frame: np.ndarray, kind: int, phase: float = 0.5, count: int = SWEEP_FRAMES) -> np.ndarray:
    """A table whose position is the distortion amount: frame j holds `frame` warped by (j + 0.5) / count."""
    fine = _oversampled(np.asarray(frame, dtype=np.float64).reshape(1, FRAME_SIZE))
    return _band_limited(np.concatenate([_read_warped(fine, kind, float(a), phase) for a in sweep_amounts(count)]))


def morph_sweep(frame: np.ndarray, kind: int, count: int = SWEEP_FRAMES) -> np.ndarray:
    """A table whose position is the spectral morph amount."""
    frame = np.asarray(frame, dtype=np.float64).reshape(1, FRAME_SIZE)
    return np.concatenate([spectral_morph(frame, kind, float(a)) for a in sweep_amounts(count)])


def sweep_position(amount: float, count: int = SWEEP_FRAMES) -> float:
    """Serum kParamTablePos (1..256) playing the sweep frame for `amount`."""
    return 1.0 + 255.0 * min(1.0, max(0.0, amount)) if count > 1 else 1.0
