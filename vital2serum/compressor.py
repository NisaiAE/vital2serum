"""Compressor: Vital's multiband upward/downward compressor onto Serum 2's FXComp.

Measured on Vital 1.0.7 and Serum 2.1.5: gain against the level of a steady sine in each band,
the band splits from the harmonics of a low saw, level steps for the time constants, and saws,
supersaws and complete patches for what each detector reads. Levels are
dB of the mean square in Vital's internal units, where one full-level oscillator peaks at 1.0.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math

BANDS = ('low', 'band', 'high')
# Vital settings read by convert(), without their `compressor_` prefix.
KEYS = ('enabled_bands', 'attack', 'release', 'mix') + tuple(
    f'{band}_{name}' for band in BANDS
    for name in ('upper_threshold', 'upper_ratio', 'lower_threshold', 'lower_ratio', 'gain'))

# --- Vital -------------------------------------------------------------------------------------
# Fourth-order Linkwitz-Riley splits at 120 Hz and 2.5 kHz. compressor_enabled_bands chooses the
# bands in use; with fewer than three the middle band takes over the range of the missing one.
VITAL_ENABLED = {0: ('low', 'band', 'high'), 1: ('low', 'band'), 2: ('band', 'high'), 3: ('band',)}
# Each band follows its squared signal with a one-pole filter: (attack, release) in ms at the
# 0.5 settings. A setting v scales the time by e^(8v - 4); the floor is five samples at the
# doubled internal rate.
VITAL_TIME_MS = {'low': (2.8, 40.0), 'band': (1.4, 28.0), 'high': (0.7, 15.0)}
VITAL_TIME_FLOOR_MS = 5.0/88.2
VITAL_MAX_BOOST_DB = 20.0*math.log10(32.0)
VITAL_GAIN_DB = 30.0
VITAL_LOWER_OFF_DB = -78.9           # a lower threshold down here only reaches the noise floor
QUIET_DB = -70.0                     # band level under which a difference is taken to be inaudible
# With its fast attack and slow release the follower rides the peaks of the waveform, so it reads
# above the mean square by an amount that depends on the sound. Rows: that excess in dB against
# d = (release - attack) setting. 'low' is a sinusoid (what the low band holds), the others are
# medians over a set of patches at C3 and C4 ('wide' = one band over the whole range). Half of
# the sounds lie within about 1 dB of the median at d = 0, and 80 % within 2 dB.
DETECTOR_D = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75)
DETECTOR_DB = {
    'low':  (-6.9, -2.4, 0.7, 2.2, 2.7, 2.9, 2.9),
    'band': (-6.6, -2.0, 1.8, 4.5, 6.1, 7.8, 8.7),
    'high': (-7.7, -2.5, 2.3, 6.3, 8.9, 11.1, 12.2),
    'wide': (-6.6, -1.9, 1.7, 4.3, 5.8, 6.7, 7.7),
}

# --- Serum (multiband mode) ----------------------------------------------------------------------
# Bands 0, 1, 2 are low, mid, high, split by fourth-order filters at kParamXoverLow / kParamXoverHi.
# The detector reads the mean square (within about 0.5 dB for every sound tried) and the upper
# threshold of band i sits at
#   SERUM_UNIT_DB - (2 Thresh)^2 * (SERUM_ABOVE_DB[i] - SERUM_UD_DB * (ThreshUD_i - 100))
# with the lower one a deadband of 3..24 dB under it. Above: gain falls by Ratio_i (1 - 1/Ratio)
# per dB. Below: it rises by RatioBelow (2 RatioBelow_i - 1) per dB, so 0.5 is neutral and less
# than that expands downwards. Band gains and makeup come after the gain computer.
SERUM_UNIT_DB = 7.30                 # level of a sine that reads 0 dB
SERUM_ABOVE_DB = (33.8, 30.2, 35.5)
SERUM_UD_DB = 0.24
SERUM_UD_RANGE = (0.0, 200.0)
SERUM_DEADBAND_DB = (3.0, 24.0)
SERUM_BAND_GAIN_DB = 24.0
SERUM_MAX_BOOST_DB = 36.0
SERUM_SPLIT_HZ = (20.0, 20000.0)     # where the split frequencies stop
SERUM_LIMIT_RATIO = 1e6
# kParamMakeup m gives an amplitude of 1 + (m - 1) * 31/30, or * 63/30 once Ratio is 100 or more.
SERUM_MAKEUP_SLOPE = 63.0/30.0
SERUM_TIME_MS = (0.1, 1000.0)
# After a drop the detector level falls by 8.69 dB per kParamRelease in the low and mid bands and
# 2.14 times as fast in the high band. kParamAttack does nothing in multiband mode: the gain
# follows a rising level at once.
SERUM_RELEASE_SCALE = (1.0, 1.0, 0.468)


def _interp(x: float, xs: tuple, ys: tuple) -> float:
    if x <= xs[0]: return ys[0]
    for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
        if x <= x1: return y0+(y1-y0)*(x-x0)/(x1-x0)
    return ys[-1]


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def vital_times(band: str, attack: float, release: float) -> tuple:
    """(attack, release) time constants in ms of one band's follower of the squared signal."""
    scale = lambda v: math.exp(8.0*_clamp(v, 0.0, 1.0)-4.0)
    a, r = VITAL_TIME_MS[band]
    return max(a*scale(attack), VITAL_TIME_FLOOR_MS), max(r*scale(release), VITAL_TIME_FLOOR_MS)


def detector_db(kind: str, band: str, attack: float, release: float) -> float:
    """How far above the mean square Vital's detector reads for typical material (dB)."""
    ta, tr = vital_times(band, attack, release)
    a, r = VITAL_TIME_MS[band]
    return _interp(math.log((tr/ta)/(r/a))/8.0, DETECTOR_D, DETECTOR_DB[kind])


def vital_gain_db(reading_db: float, upper_threshold: float, upper_ratio: float,
                  lower_threshold: float, lower_ratio: float, gain: float = 0.0) -> float:
    """Gain of one Vital band whose detector reads `reading_db`.

    The ratios are the part of the excess that is taken away: 1 limits, 0 does nothing, and a
    negative lower ratio expands downwards. Boost stops at 32 times; the band gain comes last.
    """
    change = _clamp(upper_ratio, 0.0, 1.0)*min(0.0, upper_threshold-reading_db) \
        + _clamp(lower_ratio, -1.0, 1.0)*max(0.0, lower_threshold-reading_db)
    return min(change, VITAL_MAX_BOOST_DB)+_clamp(gain, -VITAL_GAIN_DB, VITAL_GAIN_DB)


def serum_knee(thresh: float, ud: float, band: int) -> float:
    """Level of the upper threshold of a Serum band."""
    return SERUM_UNIT_DB-(2.0*thresh)**2*(SERUM_ABOVE_DB[band]-SERUM_UD_DB*(_clamp(ud, *SERUM_UD_RANGE)-100.0))


def serum_thresholds(knees: dict) -> tuple:
    """(kParamThresh, {band: ThreshUD}) placing the upper thresholds of the given bands.

    kParamThresh scales the thresholds of all bands together and ThreshUD moves one band by up
    to 24 dB of the unscaled picture, so the scale is chosen to bring every band within reach
    (1, i.e. Thresh 0.5, when that works). Bands further apart than that are met half-way.
    """
    if not knees:
        return 0.5, {}
    need = {b: max(0.1, SERUM_UNIT_DB-k) for b, k in knees.items()}         # dB under Serum's 0 dB
    lowest = max(need, key=lambda b: need[b]/(SERUM_ABOVE_DB[b]+24.0))
    highest = min(need, key=lambda b: need[b]/(SERUM_ABOVE_DB[b]-24.0))
    lo = need[lowest]/(SERUM_ABOVE_DB[lowest]+24.0)
    hi = need[highest]/(SERUM_ABOVE_DB[highest]-24.0)
    if lo <= hi:
        scale = min(max(1.0, lo), hi)
    else:
        scale = (need[lowest]+need[highest])/(SERUM_ABOVE_DB[lowest]+SERUM_ABOVE_DB[highest])
    scale = min(scale, 4.0)
    return math.sqrt(scale)/2.0, {b: _clamp(100.0+(SERUM_ABOVE_DB[b]-n/scale)/SERUM_UD_DB, *SERUM_UD_RANGE)
                                  for b, n in need.items()}


def serum_gain_db(level_db: float, params: dict, band: int) -> float:
    """Gain of one band of a multiband FXComp for a steady signal at `level_db`, makeup included."""
    upper = serum_knee(params['kParamThresh'], params[f'kParamThreshUD{band}'], band)
    lower = upper-_clamp(params[f'kParamDeadband{band}'], *SERUM_DEADBAND_DB)
    ratio = params['kParamRatio']
    change = params[f'kParamRatio{band}']*(1.0-1.0/ratio)*min(0.0, upper-level_db) \
        + params['kParamRatioBelow']*(2.0*params[f'kParamRatioBelow{band}']-1.0)*max(0.0, lower-level_db)
    slope = SERUM_MAKEUP_SLOPE if ratio >= 100.0 else 31.0/30.0
    makeup = 1.0+(params.get('kParamMakeup', 1.0)-1.0)*slope
    return min(change, SERUM_MAX_BOOST_DB)+params[f'kParamGain{band}']+20.0*math.log10(makeup)


def convert(values: dict, detector: dict | None = None) -> tuple:
    """Vital compressor settings -> ([(module, type id, plainParams)], [(kind, feature, message)]).

    `values` holds Vital's settings under the names in KEYS. Each band's thresholds, ratios and
    gain are reproduced by the same band of Serum's multiband compressor, so the curve holds at
    any level; with fewer bands enabled a split is moved out of the way instead. `detector`
    overrides, per band name, how far above the mean square Vital's detector is taken to read.
    """
    notes = []
    mode = int(round(_clamp(values['enabled_bands'], 0, 3)))
    active = VITAL_ENABLED[mode]
    attack, release = values['attack'], values['release']
    kinds = {'low': 'low', 'band': 'wide' if mode else 'band', 'high': 'high'}
    reads = {b: detector_db(kinds[b], b, attack, release) for b in active}
    reads.update({b: d for b, d in (detector or {}).items() if b in active})

    lo, hi = SERUM_DEADBAND_DB
    knees, bands = {}, {}
    for band in active:
        i = BANDS.index(band)
        get = lambda name: values[f'{band}_{name}']
        upper_ratio = _clamp(get('upper_ratio'), 0.0, 1.0)
        lower_ratio = _clamp(get('lower_ratio'), -1.0, 1.0)
        if _clamp(get('lower_threshold'), -80.0, 0.0) <= VITAL_LOWER_OFF_DB:
            lower_ratio = 0.0
        upper = _clamp(get('upper_threshold'), -80.0, 0.0)-reads[band]
        lower = _clamp(get('lower_threshold'), -80.0, 0.0)-reads[band]
        deadband = _clamp(upper-lower, lo, hi)
        miss = upper-deadband-lower                          # how far Serum's lower threshold is off
        if upper_ratio == 0.0:
            upper = lower+deadband                           # only the lower threshold matters
        elif lower_ratio != 0.0 and abs(miss) > 0.5:
            # Both are in use: the upper threshold stays. A lower one out of reach is raised, or
            # dropped when raising it would change more of the range above QUIET_DB than losing it.
            if miss > 0.0 and lower-QUIET_DB < (1.0+math.sqrt(2.0))*miss:
                lower_ratio = 0.0
                notes.append(('clipped', f'compressor_{band}_lower_threshold',
                              f"Serum keeps a band's thresholds {lo:g} to {hi:g} dB apart; this one lies {upper-lower:.0f} dB under the upper one and is left out."))
            else:
                notes.append(('clipped', f'compressor_{band}_lower_threshold',
                              f"Serum keeps a band's thresholds {lo:g} to {hi:g} dB apart; this one is moved by {miss:+.1f} dB."))
        if upper_ratio or lower_ratio:
            knees[i] = upper
        bands[i] = dict(ratio=upper_ratio, below=(lower_ratio+1.0)/2.0, deadband=deadband,
                        gain=_clamp(get('gain'), -VITAL_GAIN_DB, VITAL_GAIN_DB), name=band)

    thresh, uds = serum_thresholds(knees)
    for i, wanted in knees.items():
        miss = serum_knee(thresh, uds[i], i)-wanted
        if abs(miss) > 0.5:
            notes.append(('clipped', f'compressor_{bands[i]["name"]}_upper_threshold',
                          f"The bands' thresholds are further apart than Serum reaches; this one is off by {miss:+.1f} dB."))

    # Gain beyond a band's 24 dB goes to the makeup, which sits in the wet path like Vital's gains.
    excess = max(0.0, max(b['gain'] for b in bands.values())-SERUM_BAND_GAIN_DB)
    for b in bands.values():
        if b['gain']-excess < -SERUM_BAND_GAIN_DB:
            notes.append(('clipped', f'compressor_{b["name"]}_gain',
                          f"Serum's band gain stops at -{SERUM_BAND_GAIN_DB:g} dB (wanted {b['gain']-excess:.1f} dB)."))
        b['gain'] = max(-SERUM_BAND_GAIN_DB, b['gain']-excess)

    # A band Vital does not run lies beyond the moved split and only holds what leaks into it.
    # Under 20 Hz that is left out, or it would stay loud while the band above it is turned down;
    # over 20 kHz it passes at the middle band's gain, as muting it would dull the top octave.
    split = (120.0 if 'low' in active else SERUM_SPLIT_HZ[0], 2500.0 if 'high' in active else SERUM_SPLIT_HZ[1])
    bands.setdefault(0, dict(ratio=0.0, below=0.5, deadband=lo, gain=-SERUM_BAND_GAIN_DB))
    bands.setdefault(2, dict(ratio=0.0, below=0.5, deadband=lo, gain=bands[1]['gain']))

    attack_ms, release_ms = vital_times('band', attack, release)
    params = {'kParamMultiband': 1.0, 'kParamWet': 100.0*_clamp(values['mix'], 0.0, 1.0),
              'kParamThresh': thresh, 'kParamRatio': SERUM_LIMIT_RATIO, 'kParamRatioBelow': 1.0,
              # Vital's times are for the squared signal; Serum's are amplitude times, twice as long.
              'kParamAttack': _clamp(2.0*attack_ms, *SERUM_TIME_MS),
              'kParamRelease': _clamp(2.0*release_ms, *SERUM_TIME_MS),
              'kParamMakeup': 1.0+(10.0**(excess/20.0)-1.0)/SERUM_MAKEUP_SLOPE,
              'kParamXoverLow': split[0], 'kParamXoverHi': split[1]}
    for i in range(3):
        b = bands[i]
        params.update({f'kParamThreshUD{i}': uds.get(i, 100.0), f'kParamDeadband{i}': b['deadband'],
                       f'kParamRatio{i}': b['ratio'], f'kParamRatioBelow{i}': b['below'],
                       f'kParamGain{i}': b['gain']})

    shown = ', '.join(f'{b} {reads[b]:+.1f}' for b in active)
    notes.append(('approximation', 'compressor',
                  "Vital's detector rides the peaks of the waveform and Serum's reads the average level; "
                  f"the thresholds are lowered by the usual difference ({shown} dB), which varies by a few dB with the sound."))
    if attack_ms > 3.0:
        notes.append(('approximation', 'compressor_attack',
                      f"Serum's multiband compressor reacts at once; Vital's attack ({attack_ms:.0f} ms in the middle band) lets more of each onset through."))
    if 2.0*release_ms > SERUM_TIME_MS[1]:
        notes.append(('clipped', 'compressor_release', f"Serum's release stops at {SERUM_TIME_MS[1]:g} ms (wanted {2.0*release_ms:.0f} ms)."))
    if mode:
        notes.append(('approximation', 'compressor_enabled_bands',
                      'Fewer than three bands: Serum always runs three, so a split is moved to the edge of the audible range.'))
    return [('FXComp', 5, params)], notes
