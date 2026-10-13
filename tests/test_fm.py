"""FM and RM warps pinned to measurements of Vital 1.0.7 and Serum 2.1.5 (sine carrier, sine modulator)."""
import math
import pytest
from vital2serum import fm

# Index (rad) fitted to the Bessel sidebands of the carrier.
VITAL_FM = {0.2: 1.508, 0.3: 3.41, 0.5: 9.44, 0.7: 18.46, 0.9: 30.54, 1.0: 37.70}       # amount
SERUM_PD = {0.2: 0.08, 0.3: 0.41, 0.4: 1.29, 0.5: 3.15, 0.6: 6.53, 0.7: 12.11, 0.8: 20.65,
            0.85: 26.32, 0.9: 33.08, 0.95: 41.07, 1.0: 50.42}                           # kParamWarp
# Carrier line and one sideband over the plain carrier, modulator at full scale.
VITAL_RM = {0.25: (0.75, 0.125), 0.5: (0.5, 0.25), 0.75: (0.25, 0.375), 1.0: (0.0, 0.5)}
SERUM_RM = {0.25: (0.75, 0.2508), 0.5: (0.5, 0.5016), 0.75: (0.25, 0.7524), 1.0: (0.0, 1.0032)}
# rms of the modulator a Vital carrier reads over one voice at full spread: (voices, blend, spread).
VITAL_STACK = {(1, .8, 1.0): 1.0, (1, .8, .5): 1.307, (1, .8, 0.0): 1.416, (2, .8, 1.0): 1.0, (2, .8, 0.0): 1.002,
               (3, .8, 1.0): 1.0, (3, .8, .5): 1.142, (3, .8, 0.0): 1.193, (7, .8, 1.0): 1.004, (7, .2, 1.0): 1.001,
               (7, .8, .5): 1.077, (7, .8, 0.0): 1.107, (4, .2, 1.0): 1.005, (16, .8, 1.0): 0.982}


def test_modulator_is_the_same_oscillator_in_both_synths():
    assert [fm.modulator(c, 7) for c in (1, 2, 3)] == [2, 1, 1]
    assert [fm.modulator(c, 8) for c in (1, 2, 3)] == [3, 3, 2]
    assert [fm.modulator(c, 10) for c in (1, 2, 3)] == [2, 1, 1]
    assert [fm.modulator(c, 11) for c in (1, 2, 3)] == [3, 3, 2]
    assert fm.modulator(1, 9) is None and fm.modulator(2, 12) is None and fm.modulator(1, 0) is None
    assert fm.MENUS == {7: 'kPD_OSC', 8: 'kPD_OSC2', 10: 'kRM_OSC', 11: 'kRM_OSC2'}


def test_index_laws_follow_the_measured_sidebands():
    for amount, index in VITAL_FM.items():
        assert fm.vital_index(amount) == pytest.approx(index, abs=0.03)
    for warp, index in SERUM_PD.items():
        assert fm.serum_index(warp) == pytest.approx(index, abs=0.02)
    assert fm.serum_index(50.0) == fm.serum_index(1.0)          # the control stops at 1, it is not a percentage
    assert fm.vital_index(0.5, 0.5) == pytest.approx(4.71, abs=0.03)        # half-scale table: 4.68 measured
    assert fm.serum_index(0.5, 0.5) == pytest.approx(1.58, abs=0.01)


def test_pd_warp_gives_vitals_index():
    assert fm.pd_warp(1.0) == pytest.approx(0.9299, abs=1e-4)
    assert fm.pd_warp(0.5) == pytest.approx(0.6575, abs=1e-4)
    for amount in (0.05, 0.2, 0.5, 0.9, 1.0):
        for ratio in (0.5, 1.0, 1.3):
            assert fm.serum_index(fm.pd_warp(amount, ratio))/ratio == pytest.approx(fm.vital_index(amount))
    assert fm.pd_warp(0.0) == 0.0 and fm.pd_warp(1.0, 2.0) == 1.0
    assert fm.pd_reach() == pytest.approx(1.1565, abs=1e-4)
    assert fm.pd_reach(1.751) == pytest.approx(0.8740, abs=1e-4)


def test_second_slot_takes_the_index_one_slot_cannot_reach():
    assert fm.pd_warps(0.6, 1.6) == (fm.pd_warp(0.6, 1.6), 0.0)
    first, second = fm.pd_warps(0.95, 1.6)
    assert first == 1.0 and second == pytest.approx(0.5313, abs=1e-4)
    assert (fm.serum_index(first)+fm.serum_index(second))/1.6 == pytest.approx(fm.vital_index(0.95))
    assert fm.pd_warps(1.0, 10.0) == (1.0, 1.0)


def test_pd_lead_is_the_measured_carrier_offset():
    full = fm.pd_warp(1.0)
    for warp, degrees in ((0.4, 74.0), (0.5, 180.0), (0.6, 14.0)):          # fitted to the waveform, +-2
        assert fm.pd_lead((warp/full)**2) == pytest.approx(degrees, abs=1.0)
    first, second = fm.pd_warps(0.95, 1.6)
    assert fm.pd_lead(0.95, 1.6) == pytest.approx(math.degrees(16*math.pi*(first**4+second**4)) % 360)


def test_rm_matches_carrier_and_sidebands():
    for warp, (carrier, side) in SERUM_RM.items():
        assert 1-warp == pytest.approx(carrier) and fm.SERUM_RM_GAIN*warp/2 == pytest.approx(side, abs=1e-4)
    assert fm.rm_warp(0.5) == pytest.approx(0.3326, abs=1e-4)
    assert fm.rm_gain(0.5) == pytest.approx(0.7492, abs=1e-4)
    for amount, (carrier, side) in VITAL_RM.items():
        for ratio in (1.0, 1.6):
            warp, gain = fm.rm_warp(amount, ratio), fm.rm_gain(amount, ratio)
            assert gain*(1-warp) == pytest.approx(carrier, abs=1e-9)
            assert gain*fm.SERUM_RM_GAIN*warp/ratio/2 == pytest.approx(side, abs=1e-9)
    assert fm.rm_warp(0.0) == 0.0 and fm.rm_warp(1.0) == 1.0
    assert fm.rm_gain(0.3, None) == pytest.approx(0.7)                      # modulator off: (1 - amount) remains


def test_stack_gain_follows_the_measured_modulator_level():
    for (voices, blend, spread), level in VITAL_STACK.items():
        assert fm.stack_gain(voices, blend, spread) == pytest.approx(level, abs=0.02)
