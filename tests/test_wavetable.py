import base64
import math
import struct
import numpy as np
import pytest
from vital2serum.errors import ConversionError
from vital2serum.wavetable import (render_wavetable, spectral_morph, warp_phase, phase_warp, warp_sweep, sweep_position,
                                   _MT19937, FRAME_SIZE)

N = FRAME_SIZE
X = np.arange(N) / N


def b64_frame(values):
    return base64.b64encode(struct.pack(f'<{N}f', *values)).decode()


def table(components, **flags):
    return {'version': '1.0.7', 'name': 't', 'groups': [{'components': components}], **flags}


def wave_source(frames, interpolation=0, style=1):
    return {'type': 'Wave Source', 'interpolation': interpolation, 'interpolation_style': style,
            'keyframes': [{'position': p, 'wave_data': b64_frame(v)} for p, v in frames]}


def test_dc_removed_by_default_and_normalize_off():
    frames, _ = render_wavetable(table([wave_source([(0, 0.25 + 0.5 * np.sin(2 * np.pi * X))])]))
    assert frames.shape == (1, N)
    assert abs(float(frames.mean())) < 1e-6
    assert float(frames.max()) == pytest.approx(0.5, abs=1e-5)       # no normalisation when the key is absent


def test_full_normalize_uses_span_before_dc_removal():
    frames, _ = render_wavetable(table([wave_source([(0, 0.5 + 0.25 * np.sin(2 * np.pi * X))])],
                                       remove_all_dc=True, full_normalize=True))
    # samples span 0.25..0.75; Vital's span starts from 0 -> 0.75, measured before the DC bin is
    # dropped -> scale 2/0.75 (a post-DC span of 0.5 would give 4 and a peak of 1.0)
    assert float(frames.max()) == pytest.approx(0.25 * 2 / 0.75, abs=1e-4)


def test_frequency_interpolation_midpoint():
    one, two = np.sin(2 * np.pi * X), np.sin(4 * np.pi * X)
    frames, _ = render_wavetable(table([wave_source([(0, one), (256, two)], interpolation=1)], remove_all_dc=False))
    mid = np.fft.rfft(frames[0].astype(np.float64))
    assert abs(mid[1]) / (N / 2) == pytest.approx(1.0, abs=1e-5)       # frame 0 is the first keyframe
    k = 128                                                            # frame 128 = Vital position 128
    t = k / 256
    spec = np.fft.rfft(frames[k].astype(np.float64)) / (N / 2)
    assert abs(spec[1]) == pytest.approx((1 - t) ** 2, abs=1e-4)       # sqrt-amplitudes are interpolated
    assert abs(spec[2]) == pytest.approx(t ** 2, abs=1e-4)


def test_line_source_ramp():
    line = {'num_points': 2, 'points': [0.0, 1.0, 1.0, 0.0], 'powers': [0.0, 0.0], 'smooth': False}
    comp = {'type': 'Line Source', 'interpolation_style': 1, 'num_points': 2,
            'keyframes': [{'position': 0, 'pull_power': 0.0, 'line': line}]}
    frames, _ = render_wavetable(table([comp], remove_all_dc=False))
    assert frames[0][0] == pytest.approx(-1.0) and frames[0][-1] == pytest.approx(1.0)
    assert frames[0][N // 2] == pytest.approx(2 * (N // 2) / (N - 1) - 1, abs=1e-6)


def file_source(audio, fade_style=1, window=N, **extra):
    pcm = np.round(np.asarray(audio) * 32767).astype('<i2').tobytes()
    comp = {'type': 'Audio File Source', 'interpolation_style': 1, 'window_size': window, 'fade_style': fade_style,
            'phase_style': 0, 'normalize_gain': False, 'normalize_mult': True, 'random_seed': 1,
            'audio_sample_rate': 44100, 'audio_file': base64.b64encode(pcm).decode(),
            'keyframes': [{'position': 0, 'start_position': 0.0, 'window_fade': 1.0, 'window_size': window}]}
    comp.update(extra)
    return comp


def test_file_source_reads_cycles():
    audio = np.concatenate([0.5 * np.sin(2 * np.pi * X), 0.5 * np.sin(4 * np.pi * X)])
    comp = file_source(audio)
    comp['keyframes'].append({'position': 256, 'start_position': float(N), 'window_fade': 1.0, 'window_size': N})
    frames, _ = render_wavetable(table([comp], remove_all_dc=False))
    assert np.allclose(frames[0], 0.5 * np.sin(2 * np.pi * X), atol=1e-4)
    assert np.allclose(frames[-1], 0.5 * np.sin(4 * np.pi * X), atol=1e-4)


def test_file_source_clear_phase_removes_dc_and_keeps_magnitudes():
    audio = 0.5 * np.sin(2 * np.pi * X) + 0.2
    frames, _ = render_wavetable(table([file_source(audio, phase_style=1)], remove_all_dc=False))
    spec = np.fft.rfft(frames[0].astype(np.float64))
    assert abs(spec[0]) < 1e-3
    assert abs(spec[1]) / (N / 2) == pytest.approx(0.5, abs=1e-3)


def test_mt19937_reference_value():
    assert _MT19937(5489)() == 3499211612          # first output of std::mt19937 default seed


def test_modifiers_render():
    sine = np.sin(2 * np.pi * X)
    fold = {'type': 'Wave Folder', 'interpolation_style': 1, 'keyframes': [{'position': 0, 'fold_boost': 1.0}]}
    frames, _ = render_wavetable(table([wave_source([(0, sine)]), fold], remove_all_dc=False))
    assert np.allclose(frames[0], sine, atol=1e-5)                    # boost 1 folds nothing
    clear = {'type': 'Phase Shift', 'style': 4, 'interpolation_style': 1, 'keyframes': [{'position': 0, 'phase': 0, 'mix': 1}]}
    frames, _ = render_wavetable(table([wave_source([(0, sine)]), clear], remove_all_dc=False))
    assert np.allclose(frames[0], np.cos(2 * np.pi * X), atol=1e-5)  # all phases zeroed -> cosine
    window = {'type': 'Wave Window', 'window_shape': 2, 'interpolation_style': 1,
              'keyframes': [{'position': 0, 'left_position': 0.5, 'right_position': 1.0}]}
    frames, _ = render_wavetable(table([wave_source([(0, np.ones(N))]), window], remove_all_dc=False))
    assert frames[0][0] == 0 and frames[0][-1] == 1 and frames[0][N // 4] == pytest.approx((N // 4) / (N - 1) / 0.5, abs=1e-5)
    lowpass = {'type': 'Frequency Filter', 'style': 0, 'normalize': False, 'interpolation_style': 1,
               'keyframes': [{'position': 0, 'cutoff': 1.0, 'shape': 0.0}]}
    two = np.sin(2 * np.pi * X) + np.sin(2 * np.pi * 8 * X)
    frames, _ = render_wavetable(table([wave_source([(0, two)]), lowpass], remove_all_dc=False))
    spec = np.abs(np.fft.rfft(frames[0].astype(np.float64))) / (N / 2)
    assert spec[1] == pytest.approx(1.0, abs=1e-4) and spec[8] < 1e-4


def test_serum_table_position_selects_rendered_frame():
    from vital2serum.wavetable import serum_table_position
    for w in (0, 0.9, 1, 96, 96.99, 128, 254, 254.5, 255, 256):
        pos = serum_table_position(w, 256)
        frame = min(int((pos - 1) / 255 * 256), 255)                  # Serum 2.1.5 frame selection
        assert frame == min(int(w), 255)
    assert serum_table_position(200, 1) == 1.0


@pytest.mark.parametrize('bad', [{'type': 'Slew Limiter', 'keyframes': [{'position': 0}]},
                                 {'type': 'Nope', 'keyframes': [{'position': 0}]}])
def test_unimplemented_components_raise(bad):
    with pytest.raises(ConversionError):
        render_wavetable(table([bad]))


def harmonics(frame):
    return np.abs(np.fft.rfft(np.asarray(frame, dtype=np.float64))) / (N / 2)


def test_spectral_low_and_high_pass_edges():
    # Measured on Vital 1.0.7 with a saw: amount 0.05 leaves harmonic 2 at 0.414 (low pass) and
    # harmonic 1 at 0.584 (high pass); amount 0.9 cuts above 512 / leaves 512 at 0.5.
    flat = sum(np.sin(2 * np.pi * k * X) for k in (1, 2, 3, 4, 511, 512, 513))
    low = harmonics(spectral_morph(flat, 7, 0.05))
    assert low[1] == pytest.approx(1) and low[2] == pytest.approx(2 ** 0.5 - 1, abs=1e-6) and low[3] == pytest.approx(0, abs=1e-9)
    high = harmonics(spectral_morph(flat, 8, 0.05))
    assert high[1] == pytest.approx(0.584, abs=1e-3) and high[2] == pytest.approx(1)
    assert harmonics(spectral_morph(flat, 7, 0.9))[[512, 513]] == pytest.approx([1, 0], abs=1e-9)
    assert harmonics(spectral_morph(flat, 8, 0.9))[[511, 512, 513]] == pytest.approx([0, 0.5, 1], abs=1e-9)
    assert harmonics(spectral_morph(flat, 7, 1.0))[[1, 512, 513]] == pytest.approx([1, 1, 1])


def test_spectral_stretch_moves_harmonics():
    # Measured: harmonic stretch at 0.75 sends k to 1 + 4 (k - 1); inharmonic stretch at 0.75
    # sends k to 1 + (k - 1) k^0.6, shared between the two neighbouring harmonics.
    probe = sum(np.sin(2 * np.pi * k * X) for k in (1, 2, 3, 5))
    stretched = harmonics(spectral_morph(probe, 3, 0.75))
    assert sorted(np.where(stretched > 0.5)[0]) == [1, 5, 9, 17]
    assert harmonics(spectral_morph(probe, 3, 0.5))[[1, 2, 3, 5]] == pytest.approx([1, 1, 1, 1])
    inharmonic = harmonics(spectral_morph(np.sin(2 * np.pi * 5 * X), 4, 0.75))
    target = 1 + 4 * 5 ** 0.6
    assert inharmonic[11] == pytest.approx(12 - target, abs=1e-6) and inharmonic[12] == pytest.approx(target - 11, abs=1e-6)
    frames = spectral_morph(np.stack([probe, probe]), 4, 0.3)
    assert frames.shape == (2, N) and np.allclose(frames[0], frames[1])
    with pytest.raises(ConversionError):
        spectral_morph(probe, 6, 0.5)


def test_warp_phase_follows_vital():
    # read positions measured on Vital 1.0.7 with a ramp table (one cycle at C0), at cycle position 1/64 + k/16
    at = lambda kind, amount, phase=0.5: warp_phase(kind, amount, phase)[0][32::128]
    assert at(1, 0.75)[:4] == pytest.approx([0.5625, 0.8125, 0.0625, 0.3125])            # sync: 4 cycles, centred
    assert at(1, 0.0)[[0, 15]] == pytest.approx([0.4697, 0.5283], abs=1e-4)              # a sixteenth of the table
    assert at(1, 0.25, 0.8)[0] == pytest.approx(0.679, abs=1e-3)
    read, gain = warp_phase(2, 0.5)
    assert read[32::128] == pytest.approx(at(1, 0.5)) and gain[[0, 512, 1024]] == pytest.approx([0, 2 ** -0.5, 1])
    assert at(3, 0.75)[[0, 3, 8, 13]] == pytest.approx([0, 0.3724, 0.6276, 0], abs=1e-3)   # quantize: 2.685 steps
    assert at(3, 0.25) == pytest.approx(np.arange(16) / 16 + 1 / 64, abs=1e-4)
    assert at(3, 0.75, 0.25)[[0, 5, 8, 9]] == pytest.approx([0.0931, 0.4655, 0.5, 0.7207], abs=1e-3)
    assert at(4, 0.5) == pytest.approx(np.arange(16) / 16 + 1 / 64)                       # bend: straight in the middle
    assert at(4, 0.0)[[0, 8, 15]] == pytest.approx([0.637, 0.5, 0.593], abs=1e-3)
    assert at(5, 1.0)[[0, 7, 8]] == pytest.approx([0.008, 0.232, 0.752], abs=1e-3)        # squeeze
    assert at(5, 0.5) == pytest.approx(np.arange(16) / 16 + 1 / 64)
    read, gain = warp_phase(6, 0.5)                                                       # pulse width: half the cycle
    assert read[32::128][[1, 3, 14]] == pytest.approx([0.156, 0.406, 0.781], abs=1e-3)
    assert list(gain[32::128]) == [1, 1, 1, 1] + [0] * 8 + [1, 1, 1, 1]


def test_phase_warp_tables():
    sine = np.sin(2 * np.pi * X)
    assert harmonics(phase_warp(sine, 1, 0.75))[[1, 4, 5]] == pytest.approx([0, 1, 0], abs=1e-6)
    assert harmonics(phase_warp(sine, 4, 0.5))[[1, 2]] == pytest.approx([1, 0], abs=1e-6)
    table = warp_sweep(sine, 1)
    assert table.shape == (256, FRAME_SIZE)
    assert harmonics(table[127])[1] > 0.99 and harmonics(table[224])[8] > 0.8             # amounts 0.498 and 0.877
    assert sweep_position(0.0) == 1 and sweep_position(1.0) == 256 and sweep_position(0.5, 1) == 1
    with pytest.raises(Exception):
        warp_phase(7, 0.5)
