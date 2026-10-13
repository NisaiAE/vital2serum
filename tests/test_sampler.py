"""Sample -> Noise oscillator, pinned to measurements of Vital 1.0.7 and Serum 2.1.5.

Both synths played a 440 Hz sine clip and a linear chirp; the comments give what was read.
"""
import math
import pytest
from vital2serum import sampler
from vital2serum.sampler import noise_settings

C_MAJOR = sum(1 << pc for pc in (0, 2, 4, 5, 7, 9, 11))


def settings(keytrack=1, transpose=0.0, tune=0.0, random_phase=0, loop=1, bounce=0, sample_rate=44100,
             frames=88200, **extra):
    return noise_settings(keytrack, transpose, tune, random_phase, loop, bounce, sample_rate, frames, **extra)


# --- pitch --------------------------------------------------------------------------------------
def test_keytracked_sample_is_rooted_an_octave_apart():
    # Vital: 440.000 Hz on C4, 220 on C3. Serum with pitch tracking: 440.001 Hz on C5, 220 on C4.
    # The colour makes up the octave in whole semitones (0.5 + 12 / 128).
    one_shot = settings(loop=0, frames=66150)
    assert one_shot['oscillator'] == {'kParamPitchTrack': 1.0}
    assert one_shot['noise'] == {'kParamOneShot': 1.0, 'kParamColor': 0.59375}
    assert one_shot['notes'] == [] and not one_shot['mirror']
    looped = settings(transpose=36.0, frames=364432)
    assert looped['noise'] == {'kParamOneShot': 0.0, 'kParamColor': 0.5+48/128}


def test_vital_rounds_and_limits_the_transposition():
    # 0.5 -> +1.00, 2.3 -> +2.00, -0.5 -> -1.00, -2.5 -> -3.00, 60 -> +48.00, -60 -> -48.00 semitones
    assert [sampler.vital_transpose(t) for t in (0.49, 0.5, 2.3, -0.49, -0.5, -2.5, 60, -60)] == [0, 1, 2, 0, -1, -3, 48, -48]


def test_tune_goes_to_the_fine_control_when_tracking():
    # Serum's kParamFine: +-1 -> +-1.000 semitone with pitch tracking, and it stops there.
    assert settings(transpose=12.0, tune=0.48)['noise'] == {'kParamOneShot': 0.0, 'kParamColor': 0.5+24/128, 'kParamFine': 0.48}
    # Vital limits sample_tune to +-1 (2 -> +1.000).
    assert settings(tune=2.0)['noise']['kParamFine'] == 1.0
    # 14 + 0.9 + voice 0.6 semitones: Vital 27.5 above the key; beyond +-1 the rest moves to the colour.
    carried = settings(transpose=14.0, tune=0.9, voice_tune=0.6)['noise']
    assert carried['kParamColor'] == 0.5+28/128 and carried['kParamFine'] == pytest.approx(-0.5)


def test_voice_transposition_moves_the_sample_only_with_keytrack():
    # Vital, voice_transpose 12 on C4: 880 Hz with keytrack, 440 Hz without.
    assert 'kParamColor' not in settings(voice_transpose=-12.0)['noise']
    assert settings(voice_transpose=-24.0, voice_tune=0.5)['noise'] == {'kParamOneShot': 0.0, 'kParamColor': 0.5-12/128, 'kParamFine': 0.5}
    assert settings(keytrack=0, voice_transpose=12.0, voice_tune=0.5)['noise'] == {'kParamOneShot': 0.0}


def test_untracked_colour_is_continuous_with_two_slopes():
    # Serum without tracking: colour 0.75 -> +24.000, 1.0 -> +48.000, 0.45 -> -19.200, 0.375 -> -48.000
    # semitones on every key. kParamFine is avoided there (0.96 semitones at 0.5, 0.6 at 0.75).
    assert sampler.colour(24.0, False) == 0.75 and sampler.colour(48.0, False) == 1.0
    assert sampler.colour(-19.2, False) == pytest.approx(0.45) and sampler.colour(-48.0, False) == 0.375
    up = settings(keytrack=0, transpose=12.0, tune=0.5)
    assert up['oscillator'] == {} and up['noise'] == {'kParamOneShot': 0.0, 'kParamColor': 0.5+12.5/96}
    assert settings(keytrack=0, transpose=-18.0)['noise']['kParamColor'] == 0.5-18/384


def test_clip_sample_rate():
    # Tracking: the speed follows detuneFactor (doubling it -> +12.000 semitones); Serum's own 48 kHz
    # files carry 0.5179013309837353. The field sampleRate changes nothing.
    assert sampler.detune_factor(44100) == pytest.approx(0.4758218478413068, rel=1e-12)
    assert sampler.detune_factor(48000) == pytest.approx(0.5179013309837353, rel=1e-12)
    assert settings(sample_rate=48000)['fields']['detuneFactor'] == pytest.approx(0.5179013309837353, rel=1e-12)
    # Not tracking: a 48 kHz clip of 440 Hz came out at 404.25 Hz (host 44.1, 48 and 96 kHz), so the
    # colour carries 12 log2(48000 / 44100) = 1.467 semitones. Vital plays 440 Hz in every case.
    assert settings(keytrack=0, sample_rate=48000)['noise']['kParamColor'] == pytest.approx(0.5+1.4671/96, abs=1e-6)
    eight_bit = settings(keytrack=0, transpose=-3.0, tune=0.48, sample_rate=48000)['noise']
    assert eight_bit['kParamColor'] == pytest.approx(0.5+(-3+0.48+1.4671)/384, abs=1e-6) and 'kParamFine' not in eight_bit


def test_pitch_outside_the_noise_oscillator_is_reported():
    # Tracking: the colour reaches +-64 semitones (1.0 -> +64.000). 48 + 12 + voice 12 = 72.
    high = settings(transpose=48.0, voice_transpose=12.0)
    assert high['noise']['kParamColor'] == 1.0 and [n[:2] for n in high['notes']] == [('clipped', 'sample_transpose')]
    # Not tracking: +48 is the top. A 96 kHz clip needs 13.47 on top of the transposition.
    fast = settings(keytrack=0, transpose=40.0, sample_rate=96000)
    assert fast['noise']['kParamColor'] == 1.0 and [n[0] for n in fast['notes']] == ['clipped']
    assert settings(transpose=48.0)['notes'] == [] and settings(keytrack=0, transpose=-48.0, tune=-1.0)['notes'] == []


def test_transpose_quantize_snaps_the_static_transposition():
    # Vital, C major: 1 -> 0, 3 -> 2, 6 -> 5, 8 -> 7, 10 -> 9, -1 stays. Only C: 6 -> 0, 8 -> 12, -1 -> 0.
    assert [sampler.snap(t, C_MAJOR) for t in (0, 1, 3, 6, 8, 10, -1)] == [0, 0, 2, 5, 7, 9, -1]
    assert [sampler.snap(t, 1) for t in (1, 3, 6, 8, 10, -1)] == [0, 0, 0, 12, 12, 0]
    assert [sampler.snap(t, 1 | 1 << 7) for t in (3, 6, 8, 10, -1)] == [0, 7, 7, 12, 0]       # C and G
    assert settings(transpose=6.0, transpose_quantize=C_MAJOR)['noise']['kParamColor'] == 0.5+17/128
    # The global switch snaps the played note (C#4 sounded as C4); without keytrack it changes nothing.
    snapped = settings(transpose_quantize=C_MAJOR | sampler.GLOBAL_SNAP)
    assert [n[:2] for n in snapped['notes']] == [('unsupported', 'sample_transpose_quantize')]
    assert settings(keytrack=0, transpose=8.0, transpose_quantize=1 | sampler.GLOBAL_SNAP)['notes'] == []


def test_transposition_as_a_modulation_destination():
    # Serum, macro at 100 -> kParamColor: 25 % moved +32.00 semitones tracking (1 % -> +1.28, so
    # modulation is not snapped) and +24.00 not tracking; -12.5 % -> -48.00 not tracking.
    law = settings(transpose=12.0)['transpose_law']
    assert law(12.0) == 0.5+24/128 and law(36.0)-law(12.0) == pytest.approx(24/128)
    free = settings(keytrack=0)['transpose_law']
    assert free(24.0) == 0.75 and free(-48.0) == 0.375 and free(60.0) == 1.0


# --- start, loop, bounce ------------------------------------------------------------------------
def test_start_position_and_loop():
    # Both start at the clip's first sample by default and loop without a crossfade.
    assert settings()['noise'] == {'kParamOneShot': 0.0, 'kParamColor': 0.59375}
    # Vital's random phase: 24 starts between 0.009 and 0.996 of the clip. Serum spreads the start
    # over (kParamRandomPhase / 100)^2 of it: 50 -> up to 0.249, 100 -> up to 0.967.
    assert settings(random_phase=1)['noise']['kParamRandomPhase'] == 100.0
    # A one-shot of 0.5 s sounded 0.505 s in Vital and 0.503 s in Serum.
    assert settings(loop=0)['noise']['kParamOneShot'] == 1.0


def test_bounce_is_the_clip_followed_by_its_reverse():
    # Vital's bounce repeats every 2 x 22050 samples and stops after one round trip without loop
    # (1.002 s for a 0.5 s clip); Serum played the mirrored clip for 1.002 s.
    out = settings(keytrack=0, bounce=1, frames=22050)
    assert out['mirror'] and out['fields']['numFrames'] == 44100 and out['notes'] == []
    assert sampler.mirror([[1, 2, 3], [4, 5, 6]]) == [[1, 2, 3, 3, 2, 1], [4, 5, 6, 6, 5, 4]]
    # Vital starts a bouncing clip on the forward pass only: the first half of the mirrored clip.
    assert (settings(bounce=1, random_phase=1)['noise']['kParamRandomPhase']/100)**2 == pytest.approx(0.5)
    long = settings(bounce=1, frames=sampler.MIRROR_LIMIT+1)
    assert not long['mirror'] and long['fields']['numFrames'] == sampler.MIRROR_LIMIT+1
    assert [n[:2] for n in long['notes']] == [('unsupported', 'sample_bounce')]


# --- pan ----------------------------------------------------------------------------------------
def test_pan_law():
    # Serum's noise oscillator, channel gain re centre at kParamPan -50 / -25: left 1.6557 / 1.3709,
    # right 0 / 0.5430. Vital at pan -1 / -0.5: left 1.412 / 1.306, right 0 / 0.541.
    assert sampler._pan_gain(1.0) == pytest.approx(1.6557, abs=2e-3) and sampler._pan_gain(0.5) == pytest.approx(1.3709, abs=2e-3)
    assert sampler._pan_gain(-0.5) == pytest.approx(0.5430, abs=2e-3) and sampler._pan_gain(-0.9) == pytest.approx(0.1155, abs=2e-3)
    assert sampler.pan(0.0) == (0.0, 1.0)
    value, gain = sampler.pan(-0.5)
    near, far = sampler._pan_gain(-value/50)*gain, sampler._pan_gain(value/50)*gain
    assert near == pytest.approx(math.sqrt(2)*math.cos(math.pi/8), abs=1e-6) and far == pytest.approx(math.sqrt(2)*math.sin(math.pi/8), abs=1e-6)
    value, gain = sampler.pan(1.0)
    assert value == pytest.approx(50.0) and gain == pytest.approx(math.sqrt(2)/1.6569, abs=1e-3)
