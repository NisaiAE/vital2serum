"""Distortion transfer curves, sample holding, filtering and dry/wet conversion.

Amplitudes use Vital's internal scale: a level-0.7071 oscillator peaks at 0.5,
and the clipping modes stop at 1.0. X-Shaper curves cover both input polarities.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import bisect
import math
from .curves import remap_curve

SOFT_CLIP, HARD_CLIP, LINEAR_FOLD, SINE_FOLD, BIT_CRUSH, DOWN_SAMPLE = range(6)
# Vital settings read by convert(), without their `distortion_` prefix.
KEYS = ('type', 'drive', 'mix', 'filter_order', 'filter_cutoff', 'filter_blend', 'filter_resonance')
MIN_DRIVE_DB, MAX_DRIVE_DB = -30.0, 30.0

# --- Vital -------------------------------------------------------------------------------------
# Types 0..4 are memoryless curves of the input. The clips and folds scale it by
# g = 10^(drive / 20); bit crush and down sample read the drive as a position t = (drive + 30) / 60.
#   soft clip    tanh(g x)                  hard clip   clamp(g x, -1, 1)
#   linear fold  1 - |((g x + 1) mod 4) - 2|   (a triangle between -1 and 1 that starts as g x)
#   sine fold    sin(pi/2 g x)
#   bit crush    round(x / q) q with q = t^2  (no ceiling; -30 dB leaves the signal alone)
# Down sample holds its input. A new sample is taken VITAL_HOLD_HZ (1 - t)^2 times a second,
# whatever the host rate (44.1, 48 and 96 kHz measured), and never more often than every second
# sample of the oversampled stream. Two such holds run a quarter of a period apart, one feeding
# the even and one the odd samples; at the default 2x oversampling the output is their mean.
# +30 dB never takes a new sample.
VITAL_HOLD_HZ = 44532.0
VITAL_HOLD_OFFSET = 0.25
# The filter is a 12 dB state-variable filter inside the wet path, before (order 1) or after
# (order 2) the curve: (low + band s + high s^2) / (1 + s / Q + s^2) around 8.1758 * 2^(cutoff / 12) Hz.
VITAL_Q = (0.5, 15.5)                # Q = 0.5 + 15.5 resonance^3 (0.50, 0.74, 2.44, 7.05 at 0, .25, .5, .75)
# The mix is a straight crossfade: (1 - mix) dry + mix wet.

# --- Serum -------------------------------------------------------------------------------------
# kXShaperAsym reads curve A of the rack entry's `flex` pair as one transfer curve over both
# polarities (kXShaper mirrors a curve drawn for one polarity and was not used): the input runs
# from -SHAPER_IN at x = 0 to +SHAPER_IN at x = 1 and clamps there, the output from -SHAPER_OUT
# to +SHAPER_OUT, both linear, straight between the points. kParamDrive fades from curve A to
# curve B. The output passes a 12 Hz first-order high-pass (the plain modes have none) and comes
# 6.4 samples late; Serum delays the dry path by 7.
SHAPER_IN = 1.650
SHAPER_OUT = 0.824
SHAPER_ENTRIES = 480                 # a curve with 484 entries is refused when the state loads
SHAPER_HIGHPASS_HZ = 12.0
SINE_SEGMENTS = 8                    # straight pieces per quarter wave where the curve has room
# kLinFold and kSinFold (not used, kept for reference): the gain ahead of the fold is
# 0.5 + 1065 (drive / 100)^3 instead of the clip modes' 0.5 + 2 (drive / 100)^3, the fold turns at
# +-0.82, and every further turn comes later than the one before (by 8 % for the linear fold, 25 %
# for the sine fold, whose first quarter is a straight line), so only the first turn or two agree
# with Vital's evenly spaced folds.
SERUM_FOLD_GAIN = (0.5, 1065.0)
# kDownsample holds 1 + round(SERUM_HOLD_S * rate * (drive / 100)^2) samples (44.1, 48, 96 kHz).
SERUM_HOLD_S = 0.002
# dry/wet: the wet weight is W = sin^2(wet / 100 * pi / 2), the dry weight 1 - W^2, and
# kParamLevelOut ((2x)^2 in amplitude) follows the sum. kParamWet defaults to 100, kParamDrive to 25.
LEVEL_OUT_MAX = 4.0
# The filter (kParamPrePost 1 before, 2 after the curve, wet path only) is the same state-variable
# family. kParamFreq follows the voice filter's scale, kParamLPHP crossfades low-pass (0),
# band-pass (50) and high-pass (100) linearly, and kParamBW sets Q (larger is narrower):
SERUM_FILTER_BASE, SERUM_FILTER_SPAN = -0.39, 137.86     # MIDI note at kParamFreq 0, semitones to 1
SERUM_Q = [(0.075, 0.730), (0.15, 0.782), (0.3, 0.898), (0.5, 0.996), (0.75, 1.143), (1.0, 1.267),
           (1.5, 1.505), (2.0, 1.788), (2.5, 2.052), (3.0, 2.438), (3.5, 2.896), (4.0, 3.441),
           (4.5, 4.087), (5.0, 5.201), (5.5, 6.620), (6.0, 9.025), (6.5, 13.644), (7.0, 28.124)]
# With the filter on Serum's wet path also loses 0.7 dB at 7 kHz and 3.2 dB at 14 kHz.


def _clamp(x: float, low: float, high: float) -> float:
    return min(high, max(low, x))


def _interp(x: float, points: list) -> float:
    if x <= points[0][0]: return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1: return y0+(y1-y0)*(x-x0)/(x1-x0)
    return points[-1][1]


# --- Vital's laws --------------------------------------------------------------------------------
def vital_gain(drive_db: float) -> float:
    return 10.0**(_clamp(drive_db, MIN_DRIVE_DB, MAX_DRIVE_DB)/20.0)


def vital_position(drive_db: float) -> float:
    return (_clamp(drive_db, MIN_DRIVE_DB, MAX_DRIVE_DB)-MIN_DRIVE_DB)/(MAX_DRIVE_DB-MIN_DRIVE_DB)


def vital_step(drive_db: float) -> float:
    """Quantisation step of Vital's bit crush."""
    return vital_position(drive_db)**2


def vital_shape(kind: int, x: float, drive_db: float) -> float:
    """Vital's output for the input sample x (types 0..4)."""
    u = vital_gain(drive_db)*x
    if kind == SOFT_CLIP: return math.tanh(u)
    if kind == HARD_CLIP: return _clamp(u, -1.0, 1.0)
    if kind == LINEAR_FOLD: return 1.0-abs((u+1.0) % 4.0-2.0)
    if kind == SINE_FOLD: return math.sin(0.5*math.pi*u)
    if kind == BIT_CRUSH:
        step = vital_step(drive_db)
        return math.floor(x/step+0.5)*step if step > 0.0 else x
    raise ValueError('down sample is not a curve')


def vital_hold_hz(drive_db: float) -> float:
    """How often each of Vital's two down-sample holds takes a new sample."""
    return VITAL_HOLD_HZ*(1.0-vital_position(drive_db))**2


def vital_filter(cutoff: float, blend: float, resonance: float) -> tuple:
    """(Hz, Q, (low, band, high)) of Vital's distortion filter."""
    side = _clamp(blend, 0.0, 2.0)-1.0
    weights = (max(-side, 0.0), math.sqrt(1.0-side*side), max(side, 0.0))
    return 8.1758*2.0**(cutoff/12.0), VITAL_Q[0]+VITAL_Q[1]*_clamp(resonance, 0.0, 1.0)**3, weights


# --- Serum's laws --------------------------------------------------------------------------------
def serum_mix(wet: float) -> tuple:
    """(dry weight, wet weight) at kParamWet."""
    weight = math.sin(_clamp(wet, 0.0, 100.0)/100.0*math.pi/2.0)**2
    return 1.0-weight*weight, weight


def serum_hold(drive: float, sample_rate: float = 44100.0) -> int:
    """Samples kDownsample holds at kParamDrive."""
    return 1+int(math.floor(SERUM_HOLD_S*sample_rate*(_clamp(drive, 0.0, 100.0)/100.0)**2+0.5))


def serum_fold_gain(drive: float) -> float:
    """Gain ahead of kLinFold / kSinFold at kParamDrive."""
    return SERUM_FOLD_GAIN[0]+SERUM_FOLD_GAIN[1]*(_clamp(drive, 0.0, 100.0)/100.0)**3


def serum_filter(freq: float, lphp: float, bw: float) -> tuple:
    """(Hz, Q, (low, band, high)) of Serum's distortion filter."""
    note = SERUM_FILTER_BASE+SERUM_FILTER_SPAN*freq
    mix = _clamp(lphp, 0.0, 100.0)/50.0
    weights = (1.0-mix, mix, 0.0) if mix <= 1.0 else (0.0, 2.0-mix, mix-1.0)
    q = math.exp(_interp(bw, [(b, math.log(q)) for b, q in SERUM_Q]))
    return 8.1758*2.0**(note/12.0), q, weights


def serum_shaper(curve: dict, x: float) -> float:
    """Output of kXShaperAsym for the input sample x (ahead of its high-pass and kParamLevelOut)."""
    position = _clamp((x/SHAPER_IN+1.0)/2.0, 0.0, 1.0)
    xs, ys = curve['xVals'], curve['yVals']
    i = max(0, bisect.bisect_right(xs, position)-1)        # a step is two entries on one x; the later one holds
    level = ys[i] if i == len(xs)-1 else ys[i]+(ys[i+1]-ys[i])*(position-xs[i])/(xs[i+1]-xs[i])
    return (1.0-2.0*level)*SHAPER_OUT                      # yVals run downwards


# --- mapping ---------------------------------------------------------------------------------------
def _fold_points(gain: float, span: float) -> list:
    points = [(0.0, 0.0)]
    turn = 1
    while turn < gain*span:
        points.append((turn/gain, 1.0 if turn % 4 == 1 else -1.0)); turn += 2
    points.append((span, 1.0-abs((gain*span+1.0) % 4.0-2.0)))
    return points


def _sine_points(gain: float, span: float) -> list:
    quarters = gain*span                                   # quarter waves on one side
    room = (SHAPER_ENTRIES-1)//2                           # segments one side may use
    step = 1.0/min(SINE_SEGMENTS, max(1, int(room/quarters))) if quarters <= room else quarters/room
    count = int(math.ceil(quarters/step-1e-9))
    return [(i*step/gain, math.sin(0.5*math.pi*i*step)) for i in range(count)]+[(span, math.sin(0.5*math.pi*quarters))]


def _crush_points(step: float, span: float) -> list:
    points = [(0.0, 0.0)]
    limit = (SHAPER_ENTRIES-3)//4                          # two entries per riser, both polarities
    count = int(math.floor(span/step+0.5))
    for j in range(1, min(count, limit)+1):
        points += [((j-0.5)*step, (j-1)*step), ((j-0.5)*step, j*step)]
    # Past the last riser that fits the curve runs on straight: the steps left out are smaller
    # than 1 / 119 of the level they sit at.
    points.append((span, count*step if count <= limit else span))
    return points


def shaper_curve(kind: int, drive_db: float, span: float = SHAPER_IN) -> tuple:
    """(curve, top) drawing Vital's linear fold, sine fold or bit crush for kXShaperAsym.

    The curve covers inputs up to +-`span`; its top stands for the output level `top`, so the
    module needs top / SHAPER_OUT of gain behind it and, for a span above SHAPER_IN,
    SHAPER_IN / span ahead of it. The linear fold is exact, the sine fold drawn in straight
    pieces (eight per quarter wave, four at +30 dB), and bit crush keeps its exact steps for the
    innermost 119 levels of either polarity.
    """
    if kind == LINEAR_FOLD: half, top = _fold_points(vital_gain(drive_db), span), 1.0
    elif kind == SINE_FOLD: half, top = _sine_points(vital_gain(drive_db), span), 1.0
    elif kind == BIT_CRUSH:
        step = vital_step(drive_db)
        half = _crush_points(step, span) if step > 0.0 else [(0.0, 0.0), (span, span)]
        top = max(y for _, y in half)
    else: raise ValueError('only the folds and bit crush are drawn as curves')
    points = [(-x, -y) for x, y in reversed(half[1:])]+half
    return remap_curve([((x/span+1.0)/2.0, (y/top+1.0)/2.0) for x, y in points]), top


def soft_clip_curve(drive_db: float = MAX_DRIVE_DB) -> dict:
    """A symmetric tanh curve with extra points around its saturation knees."""
    gain = vital_gain(drive_db)
    half = [(0.0,0.0)]
    def segment(x0,y0,x1,y1):
        error=max(abs(math.tanh(gain*(x0+(x1-x0)*t))-(y0+(y1-y0)*t)) for t in (.25,.5,.75))
        if error>2e-5:
            middle=(x0+x1)/2;level=math.tanh(gain*middle)
            segment(x0,y0,middle,level);segment(middle,level,x1,y1)
        else: half.append((x1,y1))
    segment(0.0,0.0,SHAPER_IN,math.tanh(gain*SHAPER_IN))
    points = [(-x,-y) for x,y in reversed(half[1:])]+half
    return remap_curve([((x/SHAPER_IN+1.0)/2.0,(y+1.0)/2.0) for x,y in points])


def drive_input_level(drive_db: float) -> float:
    """Flat EQ output position attenuating a fixed +30 dB curve to the wanted drive."""
    return 0.5*10.0**((_clamp(drive_db,MIN_DRIVE_DB,MAX_DRIVE_DB)-MAX_DRIVE_DB)/40.0)


def partial_drive_params(drive_db: float, mix: float) -> tuple:
    """Wet position and per-stage output level for a mixed, fixed soft-clip curve.

    The input is attenuated to set the drive. Five equal output gain stages restore
    its dry part while the wet position keeps the clipped part at the wanted mix.
    """
    pre_gain=10.0**((_clamp(drive_db,MIN_DRIVE_DB,MAX_DRIVE_DB)-MAX_DRIVE_DB)/20.0)
    wet,amplitude=mix_params(mix,1.0/SHAPER_OUT,pre_gain)
    return wet,0.5*math.sqrt(amplitude**0.2)


def downsample_drive(drive_db: float, sample_rate: float = 44100.0) -> float:
    """kParamDrive of kDownsample holding as long as each of Vital's holds at `drive_db`.

    Serum holds a whole number of samples at the host's rate, which a preset cannot know;
    44.1 kHz is assumed unless given. Above +23.7 dB Vital holds longer than Serum can.
    """
    rate = vital_hold_hz(drive_db)
    if rate >= sample_rate: return 0.0
    if rate <= 0.0: return 100.0
    return 100.0*math.sqrt(min(1.0, (1.0/rate-1.0/sample_rate)/SERUM_HOLD_S))


def mix_params(mix: float, wet_gain: float = 1.0, pre_gain: float = 1.0) -> tuple:
    """(kParamWet, output gain) giving Vital's (1 - mix) dry + mix wet.

    `wet_gain` is the gain Serum's wet signal needs to stand at Vital's, `pre_gain` a gain stage
    placed ahead of the module (the dry signal passes it too). Serum's weights are tied together
    (dry = 1 - wet^2), so the wet weight W solves W^2 + r W - 1 = 0 with
    r = (1 - mix) / (mix wet_gain pre_gain), and the output gain makes up the rest. Exact for any
    mix; the gain lies between 1 / pre_gain and wet_gain.
    """
    mix = _clamp(mix, 0.0, 1.0)
    if mix <= 0.0: return 0.0, 1.0/pre_gain
    wet_unit = mix*wet_gain*pre_gain
    if wet_unit == 0.0: return 0.0, 1.0/pre_gain
    dry = 1.0-mix
    denominator = math.hypot(dry, 2.0*wet_unit)+dry
    weight = 2.0*wet_unit/denominator
    return 200.0/math.pi*math.asin(math.sqrt(weight)), denominator/(2.0*pre_gain)


def filter_params(order: int, cutoff: float, blend: float, resonance: float) -> tuple:
    """(Serum filter parameters, gain the wet path then needs) for Vital's distortion filter.

    Cutoff and Q carry over; a blend between two shapes keeps their proportion, and Serum's
    crossfade then sits low + band (or band + high) below Vital's sum. Serum's Q stops at 0.73,
    so below resonance 0.245 its response stands up to 3.3 dB above Vital's around the cutoff.
    """
    if order not in (1, 2): return {'kParamPrePost': 0.0}, 1.0
    _, q, (low, band, high) = vital_filter(cutoff, blend, resonance)
    if blend <= 1.0: gain = low+band; lphp = 50.0*band/gain
    else: gain = band+high; lphp = 50.0+50.0*high/gain
    bw = _interp(math.log(q), [(math.log(q_), b) for b, q_ in SERUM_Q])
    return dict(kParamPrePost=float(order), kParamFreq=_clamp((cutoff-SERUM_FILTER_BASE)/SERUM_FILTER_SPAN, 0.0, 1.0),
                kParamLPHP=lphp, kParamBW=bw), gain


def mix_control_scales(values: dict, clip: tuple | None = None, span: float = SHAPER_IN) -> tuple | None:
    """Fixed wet and input gains, provided both mix endpoints fit the native output range."""
    endpoints=[]
    for mix in (0.0,1.0):
        modules,notes=convert(dict(values,mix=mix),clip=clip,span=span)
        if any(kind=='clipped' for kind,_,_ in notes): return None
        amplitude=(2.0*modules[-1][2]['kParamLevelOut'])**2
        if not math.isfinite(amplitude) or not 0.0<amplitude<=LEVEL_OUT_MAX: return None
        endpoints.append(amplitude)
    return endpoints[1],1.0/endpoints[0]


def register_mix_control(ctx, values: dict, clip: tuple | None, span: float, index: int) -> None:
    """Keep wet and output gain together when one mix routing controls a fixed transfer."""
    from .modulation import Destination, active_modulations, amount_modulators, mix_route_supported
    mods=active_modulations(ctx.v);into=amount_modulators(mods)
    moving=[(i,m) for i,m in mods.items() if i not in ctx.v.folded and m.target.startswith('distortion_')
            and (abs(m.amount)>1e-12 or into.get(i))]
    if len(moving)!=1 or moving[0][1].target!='distortion_mix': return
    slot,route=moving[0]
    if not mix_route_supported(ctx.v,mods,into,slot,ctx.r): return
    if route.stereo: return
    scales=mix_control_scales(values,clip,span)
    if scales is None: return
    wet_gain,pre_gain=scales
    def response(mix):
        wet,amplitude=mix_params(mix,wet_gain,pre_gain)
        return wet/100.0,0.5*math.sqrt(amplitude)
    policy=(2e-5,480)
    destinations=[Destination('FXDistortion',index,'kParamWet',law=lambda raw:response(raw)[0],curve_policy=policy),
                  Destination('FXDistortion',index,'kParamLevelOut',law=lambda raw:response(raw)[1],curve_policy=policy)]
    ctx.static_aux_law_targets.add('distortion_mix')
    def written():
        ctx.r.note('approximation','distortion_mix',
                   'A single mix control follows the wet share and output gain together; the fixed transfer and native filter differences remain.')
    def omitted():
        ctx.r.note('unsupported','distortion_mix',
                   'The complete dry and wet response could not be written together; the stored mix is retained.')
    if not hasattr(ctx,'atomic_law_destinations'):ctx.atomic_law_destinations={}
    ctx.atomic_law_destinations[slot]=(destinations,written,omitted)


def convert(values: dict, clip: tuple | None = None, span: float = SHAPER_IN,
            sample_rate: float = 44100.0) -> tuple:
    """Vital distortion settings -> ([(module, type id, plainParams, entry keys)], [(kind, feature, message)]).

    `values` holds Vital's settings under the names in KEYS. The folds and bit crush become a
    drawn curve (kXShaperAsym), down sample becomes kDownsample; for the two clip types the
    caller passes `clip` = (Serum mode, kParamDrive, output gain in dB[, gain of a stage it placed
    ahead]) from its own tables, and the mix and the filter are still solved here. `span` is the
    input level the curve must cover: above SHAPER_IN a flat EQ module is returned ahead of the
    distortion to lower the level, and the output level restores it. The fourth item of a module
    holds keys for the rack entry itself (`flex`).
    """
    notes = []
    kind = int(round(_clamp(values['type'], 0, 5)))
    drive = _clamp(values['drive'], MIN_DRIVE_DB, MAX_DRIVE_DB)
    order = int(round(_clamp(values['filter_order'], 0, 2)))
    filt, filter_gain = filter_params(order, values['filter_cutoff'], values['filter_blend'], values['filter_resonance'])
    modules, extra, pre_gain = [], {}, 1.0
    if kind in (LINEAR_FOLD, SINE_FOLD, BIT_CRUSH):
        if span > LEVEL_OUT_MAX*SHAPER_IN:
            notes.append(('clipped', 'distortion_input', 'The input range exceeds the range a Serum shaper and its gain stages can cover.'))
        span = _clamp(span, SHAPER_IN, LEVEL_OUT_MAX*SHAPER_IN)
        curve_span = span*(filter_gain if order == 1 else 1.0)
        curve, top = shaper_curve(kind, drive, curve_span)
        if kind == SINE_FOLD and vital_gain(drive)*curve_span*SINE_SEGMENTS > (SHAPER_ENTRIES-1)//2:
            notes.append(('approximation', 'distortion_drive', 'The shaper point limit reduces the sine-fold resolution at high drive and input levels.'))
        mode, serum_drive, wet_gain, pre_gain = 'kXShaperAsym', 0.0, top/SHAPER_OUT, SHAPER_IN/span
        extra = {'flex': [curve, dict(curve)]}
        if pre_gain < 1.0:
            modules.append(('FXEQ', 7, {'kParamLevelOut': 0.5*math.sqrt(pre_gain)}, {}))
        notes.append(('approximation', 'distortion_type',
                      "Vital's curve is drawn into Serum's X-Shaper; inputs beyond the drawn range stop at its edge, and it adds a 12 Hz high-pass and a short processing delay."))
    elif kind == DOWN_SAMPLE:
        mode, serum_drive, wet_gain = 'kDownsample', downsample_drive(drive, sample_rate), 1.0
        notes.append(('approximation', 'distortion_type',
                      "Serum holds a whole number of samples and one hold where Vital averages two; the hold length assumes a 44.1 kHz host."))
        if vital_hold_hz(drive) < 1.0/(SERUM_HOLD_S+1.0/sample_rate):
            notes.append(('clipped', 'distortion_drive', "Vital holds its samples longer than Serum's down sampler can."))
    else:
        if clip is None: raise ValueError('the clip types need the caller\'s mode, drive and gain')
        mode, serum_drive, wet_gain = clip[0], clip[1], 10.0**(clip[2]/20.0)
        if len(clip) > 3: pre_gain = clip[3]
    if order and vital_filter(values['filter_cutoff'], values['filter_blend'], values['filter_resonance'])[1] < SERUM_Q[0][1]-0.01:
        notes.append(('approximation', 'distortion_filter_resonance',
                      "Serum's distortion filter cannot be as flat as Vital's at low resonance; it stands up to 3 dB higher around the cutoff."))
    # A pre-filter's shape gain must reach the curve's input. Curves are drawn with that
    # wider input scale; clip callers choose their drive on that scale. Sample holding
    # is linear, so its pre-filter gain can instead be restored at the output.
    output_filter_gain = filter_gain if order != 1 or kind == DOWN_SAMPLE else 1.0
    wet, level = mix_params(values['mix'], wet_gain*output_filter_gain, pre_gain)
    if level > LEVEL_OUT_MAX*(1.0+1e-9):
        notes.append(('clipped', 'distortion', 'The output level needed is above the +12 dB a Serum module gives.'))
    params = dict(kParamMode=mode, kParamDrive=serum_drive, kParamWet=wet,
                  kParamLevelOut=min(1.0, 0.5*math.sqrt(level)), **filt)
    modules.append(('FXDistortion', 0, params, extra))
    return modules, notes
