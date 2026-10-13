"""Vital's effects on Serum 2's FX rack; all are sonic approximations.
Numeric laws/tables are derived from btesser/serum2vital/fx_common.py.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math
import numpy as np
from . import compressor, modulation_fx
from .errors import ConversionError

EFFECTS = ['chorus','compressor','delay','distortion','eq','filter_fx','flanger','phaser','reverb']
# Interior points of the upstream measured Serum 2 synced-delay intervals.
DELAY_KNOBS = {12:.004,11:.01,10:.02,9:.03869670710330132,8:.08316586760018929,
               7:.14,6:.22,5:.33,4:.5}


def decode_order(code: int) -> list[str]:
    if not isinstance(code,int) or not 0<=code<math.factorial(9):
        raise ConversionError('Invalid Vital effect order')
    digits=[0]*9
    for i in range(8,0,-1): code,digits[i]=divmod(code,i+1)
    available=list(range(9)); order=[0]*9
    for i in range(8,-1,-1): order[i]=available.pop(i-digits[i])
    return [EFFECTS[i] for i in order]


def interp(x: float, points: list[tuple[float,float]]) -> float:
    if x<=points[0][0]: return points[0][1]
    for (a,b),(c,d) in zip(points,points[1:]):
        if x<=c: return b+(d-b)*(x-a)/(c-a)
    return points[-1][1]


def inverse(x: float, points: list[tuple[float,float]]) -> float:
    return interp(x,sorted((y,z) for z,y in points))


def inverse_wet(v: float, scale: float = 1.0) -> float:
    # Match the wet-path amplitude, not the dry-path mix law.
    gain=min(1,max(0,math.sin(min(1,max(0,v))*math.pi/2)/scale))
    return math.asin(math.sqrt(gain))*2/math.pi*100


def wet_mix(mix: float, wet_gain_db: float, overlap: tuple = ()) -> tuple[float,float]:
    """(kParamWet, output gain dB) giving Vital's dry and wet parts on a Serum effect.

    Vital mixes with equal power: dry cos(m pi/2), wet sin(m pi/2). Serum's wet share is
    sin^2(w pi/2) of a straight crossfade, and its output level comes after the mix (both read
    off the effect tails of chorus and reverb). `wet_gain_db` is how much louder Vital's wet
    path is than Serum's at full wet; share and level are solved for both parts at once.
    `overlap` is what the sum of the two parts then still differs by (dB against mix): Serum's
    wet signal lines up with the dry one more than Vital's does.
    """
    m=min(1.0,max(0.0,mix))*math.pi/2
    k=10**(wet_gain_db/20)
    dry,wet=math.cos(m),k*math.sin(m)
    share=wet/(dry+wet) if dry+wet>0 else 0.0
    gain_db=20*math.log10(max(1e-6,dry+wet))+(interp(mix,list(overlap)) if overlap else 0.0)
    return math.asin(math.sqrt(share))*2/math.pi*100,gain_db


def delay_mix(mix: float, style: int) -> tuple[float,float]:
    """Solve the Delay's quadratic dry share and wet level together."""
    angle=min(1.0,max(0.0,mix))*math.pi/2
    dry=math.cos(angle)
    wet=math.sin(angle)/(math.sqrt(.5) if style>=2 else 1.0)
    # With W=sin^2(nativeWet*pi/2), Delay gives dry C*(1-W^2), wet C*W.
    gain=(math.hypot(dry,2*wet)+dry)/2
    share=min(1.0,max(0.0,wet/gain))
    return math.asin(math.sqrt(share))*2/math.pi*100,20*math.log10(gain)


def delay_mix_control(ctx, style: int, index: int) -> None:
    """Register one bounded mix source as an atomic pair of native controls."""
    from .modulation import Destination, active_modulations, amount_modulators, mix_route_supported
    mods=active_modulations(ctx.v);into=amount_modulators(mods)
    moving=[(i,m) for i,m in mods.items() if i not in ctx.v.folded
            and m.target.startswith(('delay_','delay_aux_')) and (abs(m.amount)>1e-12 or into.get(i))]
    if len(moving)!=1 or moving[0][1].target!='delay_dry_wet': return
    slot,route=moving[0]
    if not mix_route_supported(ctx.v,mods,into,slot,ctx.r): return
    if route.stereo: return
    def response(mix):
        wet,gain_db=delay_mix(mix,style)
        return wet/100,level_out(gain_db)
    policy=(2e-5,480)
    destinations=[Destination('FXDelay',index,'kParamWet',law=lambda raw:response(raw)[0],curve_policy=policy),
                  Destination('FXDelay',index,'kParamLevelOut',law=lambda raw:response(raw)[1],curve_policy=policy)]
    ctx.static_aux_law_targets.add('delay_dry_wet')
    def written():
        ctx.r.note('approximation','delay_dry_wet',
                   'A single mix control follows the dry share and wet level together; native timing, filtering and stereo routing differences remain.')
    def omitted():
        ctx.r.note('unsupported','delay_dry_wet',
                   'The complete dry and wet response could not be written together; the stored mix is retained.')
    if not hasattr(ctx,'atomic_law_destinations'):ctx.atomic_law_destinations={}
    ctx.atomic_law_destinations[slot]=(destinations,written,omitted)


# Vital's chorus wet path over Serum's at full wet (dB, steady saw), by chorus_voices: Vital adds
# its voice pairs up, Serum's chorus has one fixed structure. Depth, feedback and filter move
# the difference by about a dB; a longer delay or negative feedback add 2 to 3 dB more.
CHORUS_WET_DB = {1:0.2,2:2.6,3:2.9,4:5.9}
# With both parts at Vital's level, the mix still came out louder in between (detuned stack and
# single saw alike): the difference to take out, by Vital's dry/wet.
CHORUS_OVERLAP_DB = ((0.0,0.0),(0.3,-2.1),(0.6,-1.2),(1.0,0.0))
REVERB_OVERLAP_DB = ((0.0,0.0),(0.3,-1.4),(0.6,-0.8),(1.0,0.0))
# The same for the reverb by Vital's size, which thins the wet signal out as the room grows.
# Serum's plate holds a steady tone 5 to 10 dB lower than its own tail would suggest, so no one
# level fits both; these sit nearer the tail (within about 3 dB over decay times of 0.6 to 25 s)
# and leave a held note's reverb 3 to 7 dB low on the plate. The hall fits both within 2 dB.
REVERB_WET_DB = {'kPlate':[(0.2,-6.9),(0.5,-8.7),(0.9,-12.3)],'kHall':[(0.2,-6.0),(0.5,-0.5),(0.9,-4.0)]}


def tempo_seconds(tempo: int, sync: int, beats_per_second: float) -> float:
    # Vital tempo 8 is a quarter note = 1 beat; dotted/triplet multiply time.
    if tempo==0: raise ConversionError('Freeze cannot be represented as a finite effect period')
    return 2**(8-tempo)/beats_per_second*({2:1.5,3:2/3}.get(sync,1))


# --- shared voice/FX filter mapping -----------------------------------------
# Serum's cutoff control spans 137.86 semitones upwards from MIDI note -0.39 (8 Hz). Fitted from
# the distance between the two synths' responses at cutoff notes 36..108: with note / 135 Serum
# sat 0.4 semitones high at note 36 and 1.9 at note 108 (Vital 1.0.7, Serum 2.1.5).
SERUM_CUTOFF_SPAN = 137.86
SERUM_CUTOFF_BASE = -0.39
# Serum's key tracking is a switch: the cutoff then follows the played note one for one from
# note 32. Vital's follows keytrack * (note - 60). The Note source reads note / 127.
SERUM_KEYTRACK_ROOT = 32.0
VITAL_KEYTRACK_ROOT = 60.0
NOTE_SOURCE_CENTRE = 63.5


def serum_cutoff(note: float) -> float:
    """kParamFreq (unclamped) placing Serum's cutoff on the MIDI note Vital's cutoff value names."""
    return (note-SERUM_CUTOFF_BASE)/SERUM_CUTOFF_SPAN


RESONATOR_CUTOFF_SPAN = 136.8
# The loop-filter controls are semitone offsets from the delay frequency. Values above 80
# flatten out as the one-pole corner reaches its upper limit.
RESONATOR_LOW_OFFSETS = (-128.5,-115.9,-102.1,-87.7,-75.2,-63.1,-50.7,-38.2,-25.4,-12.6,0.22,13.24,26.0,39.1,52.45,65.6,76.0,79.7,79.7,79.7,79.7)
RESONATOR_HIGH_OFFSETS = (-160.0,-160.0,-158.8,-100.3,-77.3,-64.6,-51.6,-38.7,-25.86,-12.85,-0.08,12.82,25.1,37.8,50.1,62.7,73.6,80.1,80.1,80.1,80.1)
RESONATOR_BAND_LOW = (0.13,6.55,12.98,19.42,25.89,32.39,38.93,45.49,52.04,58.43,64.39,69.37,72.8,74.6,75.1,75.1,75.1,75.1,75.1,75.1,75.1)
RESONATOR_BAND_HIGH = (0.0,-6.41,-12.81,-19.19,-25.59,-31.99,-38.31,-44.65,-51.04,-57.43,-63.83,-70.32,-76.77,-82.8,-88.4,-94.3,-99.9,-105.2,-110.2,-114.7,-118.7)
RESONATOR_LOOP_DELAY = {'':0.0,'L6':0.24,'H6':0.12,'HL6':0.37}


def resonator_cutoff(note: float) -> float:
    return note/RESONATOR_CUTOFF_SPAN


def comb_feedback(style: int, resonance: float) -> float:
    if style%3:
        return min(1.0,max(0.0,resonance))
    value=2*min(1.0,max(0.0,resonance))-1
    return math.copysign(math.sqrt(abs(value)),value)


def comb_fit(style: int, note: float, resonance: float, blend: float, transpose: float) -> tuple:
    """Closest delay-loop type, variable, feedback, cutoff and gain for a comb response.

    The small-signal response is fitted on a logarithmic frequency grid. This keeps both the
    resonant teeth and the loop's low/high-pass corner in the choice; the two filters put their
    loop filters in different places, so some settings have no close single-filter substitute.
    The delay is fixed by the cutoff, including the negative flange's half-length delay.
    """
    hz=lambda n: 440*2**((n-69)/12)
    fc=hz(note)
    f=np.geomspace(max(8.0,fc/32),min(20000.0,fc*32),192)
    w=np.tan(np.pi*f/88200.0)
    def lp(corner):
        return 1/(1+1j*w/np.tan(np.pi*min(31000.0,max(1.0,corner))/88200.0))
    if style<3:
        low=lp(fc*2**(transpose/12))
        loop=(min(1.0,2-blend)*low+min(1.0,blend)*(1-low))*(1-lp(fc/16))
    else:
        high_corner=fc*2**(max(-48.0,transpose-48*blend)/12)
        low_corner=min(31000.0,fc*2**((transpose+48*blend)/12))
        loop=(1+high_corner/low_corner)*(1-lp(high_corner))*lp(low_corner)
    g=comb_feedback(style,resonance)
    scale=1-abs(2*resonance-1)/2 if style%3==0 else math.sqrt(.5)/(1+resonance)
    delay=np.exp(-2j*np.pi*f*((.5 if style%3==2 else 1.0)/fc+1/88200.0))
    response=(scale*loop/(1-g*loop*delay) if style%3==0 else
              scale*(1+loop*delay)/(1-g*loop*delay) if style%3==1 else
              scale*(1-loop*delay)/(1+g*loop*delay))
    db=lambda a: np.maximum(-45.0,20*np.log10(np.maximum(1e-12,np.abs(a))))
    target=db(response)
    sign=1 if style%3==1 or (style%3==0 and g>=0) else -1
    family='Comb' if style%3==0 else 'Flange'
    base=note+(12 if style%3==2 else 0)
    best=None
    for suffix,extra in RESONATOR_LOOP_DELAY.items():
        fitted_note=base+12*math.log2(1/(1+hz(base)*(1/88200.0-extra/44100.0)))
        fitted_fc=hz(fitted_note)
        delayed=sign*np.exp(-2j*np.pi*f*(1/fitted_fc+extra/44100.0))
        for var in (range(0,101,5) if suffix else (50,)):
            if suffix=='L6':
                serum_loop=lp(fitted_fc*2**(RESONATOR_LOW_OFFSETS[var//5]/12))
            elif suffix=='H6':
                serum_loop=1-lp(fitted_fc*2**(RESONATOR_HIGH_OFFSETS[var//5]/12))
            elif suffix=='HL6':
                serum_loop=(1-lp(fitted_fc*2**(RESONATOR_BAND_HIGH[var//5]/12)))*lp(min(20000.0,fitted_fc*2**(RESONATOR_BAND_LOW[var//5]/12)))
            else:
                serum_loop=np.ones_like(f)
            feedbacks=(0.0,) if abs(g)<1e-6 else np.unique(np.clip(abs(g)+np.linspace(-.25,.05,7),0,.97))
            for feedback in feedbacks:
                numerator=feedback**2 if family=='Comb' else 1.0
                serum=.5/(1+feedback)*(1+numerator*serum_loop*delayed)/(1-feedback*serum_loop*delayed)
                delta=db(serum)-target
                error=float(np.std(delta))
                if best is None or error<best[0]:
                    best=(error,family+suffix+('P' if sign>0 else 'N'),float(var),float(100*feedback),fitted_note,-float(np.mean(delta)))
    return best


# Small-signal responses of both filters (saw at C1 through cutoff note 72, harmonics read off the
# spectrum) were matched per Vital model and resonance: the Serum resonance whose response has
# the same shape, and how far Serum then sits above Vital in dB. Vital keeps the level steady as
# resonance rises, Serum's multimode types do not. Shape residuals are below 0.3 dB up to
# resonance 0.85 for analog, digital and ladder low-pass, and 0.2 to 1.2 dB for dirty 24 dB.
RESONANCE_GRID = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.85,1.0)
FILTER_FIT = {       # (model, slope): (kParamReso, level dB) along RESONANCE_GRID
    ('analog',12):    ((0,3,6,12,22,34,49,66,97,100),   (0.0,1.4,3.0,4.1,5.1,6.0,6.8,7.6,8.7,10.0)),
    ('analog',24):    ((3,5,10,17,25,36,50,66,97,100),  (0.0,1.5,2.9,4.0,5.0,5.9,6.8,7.5,8.7,11.0)),
    ('dirty',12):     ((1,4,7,14,24,37,51,68,99,100),   (-0.8,0.3,1.7,2.9,3.8,4.6,5.3,5.8,6.7,8.0)),
    ('dirty',24):     ((0,2,3,4,7,13,22,38,70,100),     (-0.9,0.0,1.5,2.9,4.1,5.1,6.0,6.9,8.4,9.0)),
    ('digital',12):   ((0,0,2,5,20,40,57,70,81,88),     (0.0,-0.1,0.1,0.5,1.1,1.9,3.1,4.5,6.9,9.5)),
    ('digital',24):   ((3,3,4,9,24,42,58,70,81,88),     (0.0,-0.1,0.0,0.4,1.0,1.9,3.1,4.5,6.9,9.5)),
    # the ladder's low-pass on Serum's MgL types
    ('ladder',12):    ((0,11,21,32,42,53,63,74,89,98),  (0.0,-0.9,-1.6,-2.3,-2.9,-3.5,-4.0,-4.6,-5.3,-5.6)),
    ('ladder',24):    ((0,17,33,48,62,74,85,92,97,98),  (0.0,-0.9,-1.6,-2.2,-2.8,-3.3,-3.8,-4.1,-4.2,-4.6)),
    # its other blends on the multimode types: the shapes agree only loosely (1 to 4 dB)
    ('ladder mix',12):((0,2,4,7,11,16,22,29,40,100),    (0.0,0.4,0.8,1.0,1.2,1.4,1.5,1.6,1.9,2.2)),
    ('ladder mix',24):((0,2,4,5,8,11,14,17,20,25),      (0.0,0.7,1.3,1.6,1.8,2.3,2.4,2.4,2.4,2.4)),
    # the diode ladder on LadderEMS, whose corner sits about 8 semitones higher (DIODE_CUTOFF)
    ('diode',24):     ((7,7,8,10,14,21,32,46,76,95),    (-0.7,-0.7,0.0,1.2,2.5,3.8,4.8,5.4,6.1,5.8)),
}
DIODE_CUTOFF = ((-9.1,-9.1,-8.9,-8.7,-8.1,-7.9,-7.9,-7.9,-7.9,-7.9),      # semitones, high-pass blend 0
                (-5.7,-5.4,-5.4,-5.1,-5.1,-5.4,-5.4,-5.9,-7.1,-7.9))      # and blend 1
# Serum below Vital, in dB, where the blend is not a plain low-, band- or high-pass (level only;
# the shapes of in-between blends differ by 1 to 5 dB).
BLEND_LEVEL_DB = {
    (0,12):((0,0.0),(0.5,4.3),(1,0.0),(1.5,1.5),(2,0.0)), (1,24):((0,0.0),(0.5,7.1),(1,0.0),(1.5,3.2),(2,0.0)),
    (2,24):((0,0.0),(1,2.4),(2,0.0)), (3,24):((0,1.5),(0.5,0.0),(1,-5.0),(1.5,-1.0),(2,2.4)),
    (4,24):((0,6.5),(1,6.5),(1.5,0.5),(2,2.4)),
}
# Vital's styles 3 and 4 start as a 24 dB band-pass and end as a notch at the cutoff. In between,
# style 3 ("notch pass swap") is a flat response with two notches closing in on the cutoff, which
# Serum's peak shape stands in for; style 4 (band / peak / notch) raises the floor under the peak:
# 26 dB below it at blend 0.5, 17 dB at 1, 6 dB at 1.5. kParamVar of BPN24 with the same picture:
STYLE_VAR = {3:((0,0.0),(0.5,25.0),(1,50.0),(1.5,50.0),(2,100.0)), 4:((0,0.0),(0.5,12.0),(1,24.0),(1.5,42.0),(2,100.0))}
FILTER_KINDS = {0:'analog',1:'dirty',2:'ladder',3:'digital',4:'diode'}


def filter_fit(key: tuple, resonance: float) -> tuple[float,float]:
    """(kParamReso, dB Serum sits above Vital) for a Vital resonance, from FILTER_FIT."""
    reso,level=FILTER_FIT[key]
    return interp(resonance,list(zip(RESONANCE_GRID,reso))),interp(resonance,list(zip(RESONANCE_GRID,level)))


def filter_type(model: int, style: int, blend: float) -> tuple[str,float,tuple]:
    """(Serum type, kParamVar, FILTER_FIT key) for a Vital model, style and blend.

    Vital's styles 2 to 4 (notch blend, notch spread, band / peak / notch) are 24 dB shapes:
    their ends measure as Serum's 24 dB low-, band- and high-pass and its 24 dB notch.
    """
    kind=FILTER_KINDS[model]
    slope=12 if style==0 else 24
    if kind=='diode':
        return 'LadderEMS',0.0,('diode',24)
    if kind=='ladder' and style in (0,1):
        if blend==0: return f'MgL{slope}',0.0,('ladder',slope)
        return f'LBH{slope}',blend*50,('ladder mix',slope)
    key=('analog' if kind=='ladder' else kind,slope)
    if style in (0,1): return f'LBH{slope}',blend*50,key
    if style==2: return 'LNH24',blend*50,key
    return 'BPN24',interp(blend,list(STYLE_VAR[style])),key


# Vital runs the input of every filter through a saturator, drive or no drive: the output follows
# c * tanh(k * x), where x = 1 is the peak of one full-level oscillator (measured as static
# input/output curves, Vital 1.0.7). (c, k) per model and slope at drive 0, 2.5, ... 20 dB.
# The analog filter is tanh(g x) / sqrt(g) exactly; the others are within a few percent.
VITAL_FILTER_SATURATION = {
    ('analog',12):[(0.999,1.00),(0.865,1.34),(0.749,1.78),(0.649,2.36),(0.562,3.15),(0.486,4.19),(0.421,5.56),(0.365,7.33),(0.316,9.62)],
    ('analog',24):[(0.999,1.00),(0.865,1.34),(0.749,1.78),(0.649,2.36),(0.562,3.15),(0.486,4.19),(0.421,5.54),(0.365,7.30),(0.316,9.54)],
    ('dirty',12):[(0.714,1.49),(0.714,2.39),(0.714,3.58),(0.714,5.16),(0.714,7.24),(0.714,10.13),(0.715,14.00),(0.715,19.05),(0.716,25.00)],
    ('dirty',24):[(0.714,1.91),(0.714,3.05),(0.714,4.57),(0.714,6.58),(0.714,9.24),(0.715,12.88),(0.715,17.80),(0.716,23.93),(0.716,30.17)],
    ('ladder',12):[(1.000,1.00),(0.923,1.37),(0.832,2.06),(0.752,3.04),(0.673,4.65),(0.599,7.24),(0.531,11.29),(0.468,17.45),(0.412,24.41)],
    ('ladder',24):[(1.000,1.01),(0.950,1.42),(0.888,2.17),(0.833,3.17),(0.775,4.72),(0.716,6.71),(0.655,8.99),(0.594,11.24),(0.534,12.77)],
    ('digital',12):[(1.072,1.11),(0.894,1.59),(0.763,2.19),(0.656,2.96),(0.566,3.98),(0.489,5.32),(0.423,7.13),(0.366,9.50),(0.317,12.67)],
    ('digital',24):[(0.971,1.25),(0.817,1.76),(0.699,2.40),(0.602,3.24),(0.520,4.36),(0.449,5.81),(0.388,7.75),(0.336,10.25),(0.291,13.51)],
}
# Serum's filter is linear until its drive stage clips: a gain against the drive setting (the
# same for every type measured) into a cubic soft clip, v - 4/27 v^3 up to v = 1.5. The ceiling
# sits well above Vital's, at 3.31 oscillator peaks (2.82 for the LBH24 family) (Serum 2.1.5).
SERUM_DRIVE_GAIN = [(0,1.00),(5,1.21),(10,1.46),(15,1.75),(20,2.07),(25,2.43),(30,2.85),(35,3.31),(40,3.83),(45,4.40),
                    (50,5.03),(55,5.73),(60,6.49),(65,7.33),(70,8.23),(75,9.21),(80,10.27),(85,11.40),(90,12.61),
                    (95,13.88),(100,15.22)]
SERUM_FILTER_CEILING = 3.31
# What remained between the converted filter and Vital for a saw at half scale (one oscillator at
# level 0.7071, cutoff note 72): dB Serum sat above Vital, by Vital's drive (rows: 0, 5, 10, 15,
# 20 dB) and resonance (columns: 0, 0.25, 0.5, 0.75). It collects what the small-signal fit and
# the clipping model leave out: Serum's drive running out at 100 %, the resonant peak meeting the
# clipper, and saturation curves that are not quite tanh. It holds for signals of about that level.
RESIDUAL_DRIVES = (0.0,5.0,10.0,15.0,20.0)
RESIDUAL_RESONANCES = (0.0,0.25,0.5,0.75)
FILTER_RESIDUAL_DB = {
    ('analog',12):((-0.4,-0.4,-0.3,-0.3),(-0.3,-0.3,-0.3,-1.7),(0.1,-0.1,-0.2,-1.5),(-0.3,-0.7,-1.0,-1.6),(-2.6,-3.6,-4.3,-4.5)),
    ('analog',24):((0.3,-0.4,-0.3,-0.2),(0.4,-0.3,-0.4,-1.7),(0.7,-0.1,-0.4,-1.8),(1.1,-0.1,-0.7,-1.8),(-1.0,-2.7,-3.6,-4.4)),
    ('dirty',12):((0.3,-0.6,-0.7,-1.6),(0.6,3.0,3.6,2.1),(-1.4,1.5,2.8,2.1),(-3.4,-1.8,-0.3,0.0),(-4.3,-4.3,-3.3,-2.3)),
    ('dirty',24):((1.2,1.9,1.3,-0.0),(1.2,5.0,5.3,3.9),(-1.7,2.6,3.6,2.8),(-4.0,-1.0,0.2,0.3),(-5.0,-3.8,-3.2,-2.6)),
    ('ladder',12):((-0.4,-0.3,-0.1,0.6),(0.9,1.4,1.8,2.2),(3.1,3.9,4.4,5.5),(-0.0,0.8,1.4,3.5),(-2.2,-2.0,-0.9,1.1)),
    ('ladder',24):((-0.3,-0.2,-0.1,3.5),(1.3,1.7,1.9,6.6),(3.2,3.7,4.3,8.0),(0.0,0.3,1.0,5.6),(-2.1,-3.0,-2.0,2.2)),
    ('digital',12):((1.0,1.3,0.9,-0.4),(1.4,1.6,0.9,-1.3),(1.2,1.3,0.5,-2.4),(-1.9,-1.5,-1.5,-4.8),(-4.1,-3.4,-3.5,-6.9)),
    ('digital',24):((1.9,1.4,0.9,-0.4),(2.0,1.4,0.5,-1.6),(1.4,0.8,-0.0,-3.1),(-0.5,-0.9,-1.6,-5.4),(-2.2,-2.5,-3.3,-6.6)),
}
# The same beyond resonance 0.75, without drive and with the resonance on a harmonic. Serum's
# resonance ends near Vital's 0.87, so Vital's peak outgrows it; the ladder is the other way
# round (Vital's saturates), and only half of that excess is taken out.
HIGH_RESONANCE_DB = {
    'analog':((0.75,0.0),(0.85,-0.4),(0.93,-2.7),(1.0,-4.5)),
    'digital':((0.75,0.0),(0.85,-3.5),(0.93,-5.2),(1.0,-6.6)),
    'ladder':((0.75,0.0),(0.85,4.0),(1.0,4.0)),
}


RESIDUAL_LEVEL = 0.5              # source amplitude the residuals were measured at, in oscillator full scales
RESIDUAL_HOT_LEVEL = 1.5          # three such oscillators: no shortfall left


def source_level(v,prefix: str | None=None) -> float | None:
    """Stored source amplitudes entering a filter; a preceding filter's output is unknown."""
    destinations=None
    if prefix in ('filter_1_','filter_2_'):
        reverse=bool(v.get('filter_1_on') and v.get('filter_1_filter_input'))
        forward=not reverse and bool(v.get('filter_2_on') and v.get('filter_2_filter_input'))
        if ((prefix=='filter_1_' and reverse and v.get('filter_2_on')) or
            (prefix=='filter_2_' and forward and v.get('filter_1_on'))):
            return None
        destinations=(0,2) if prefix=='filter_1_' else (1,2)
    total=0.0
    for p in ('osc_1_','osc_2_','osc_3_','sample_'):
        if v.get(p+'on') and (destinations is None or v.get(p+'destination') in destinations):
            total+=min(1.0,max(0.0,v.get(p+'level')))**2
    return total


def filter_residual(kind: str, slope: int, drive_db: float, resonance: float) -> float:
    """dB the converted filter sits above Vital at a typical level (FILTER_RESIDUAL_DB), interpolated."""
    grid=FILTER_RESIDUAL_DB.get((kind,slope))
    if grid is None: return 0.0
    columns=[interp(drive_db,[(d,row[i]) for d,row in zip(RESIDUAL_DRIVES,grid)]) for i in range(len(RESIDUAL_RESONANCES))]
    value=interp(resonance,list(zip(RESIDUAL_RESONANCES,columns)))
    return value+interp(resonance,list(HIGH_RESONANCE_DB.get(kind,((0.0,0.0),))))


SERUM_FILTER_CEILINGS = {'LBH24':2.82,'LNH24':2.82,'BPN24':2.82}
# Input scale of the cubic clip against tanh: 0.85 lies closest over the whole bend, 1 is exact for
# quiet signals; 0.95 keeps the error within about 0.5 dB below the knee and 0.9 dB above it.
SOFT_CLIP_SCALE = 0.95


def filter_drive(kind: str, slope: int, serum_type: str, drive_db: float) -> tuple[float,float]:
    """(Serum drive %, output gain dB) placing Serum's clip where Vital's filter saturates.

    The ceiling is matched through the output level and the bend through the drive, so the
    converted filter is right for quiet signals, for clipped ones and close in between,
    whatever level the oscillators feed it.
    """
    points=VITAL_FILTER_SATURATION[(kind,slope)]
    position=min(max(drive_db,0.0),20.0)/2.5
    low=min(int(position),len(points)-2);mix=position-low
    ceiling,bend=(a+(b-a)*mix for a,b in zip(points[low],points[low+1]))
    serum_ceiling=SERUM_FILTER_CEILINGS.get(serum_type,SERUM_FILTER_CEILING)
    gain=min(SERUM_DRIVE_GAIN[-1][1],max(1.0,SOFT_CLIP_SCALE*bend*serum_ceiling))
    return inverse(gain,[(d/100,g) for d,g in SERUM_DRIVE_GAIN])*100,20*math.log10(ceiling/serum_ceiling)


def filter_gain(kind: str,slope: int,name: str,key: tuple,style: int,blend: float,
                drive: float,resonance: float,input_level: float | None) -> tuple[float,float]:
    """Drive position and wet-path correction, shared by static and moving controls."""
    serum_drive,gain_db=filter_drive('analog' if kind=='diode' else kind,slope,name,drive)
    gain_db-=filter_fit(key,resonance)[1]-(0.0 if kind=='diode' else filter_fit(key,0.0)[1])
    residual=filter_residual('ladder' if key[0]=='ladder mix' else kind,slope,drive,resonance)
    if residual<0 and serum_drive>99.5 and input_level is not None:
        residual*=min(1.0,max(0.0,RESIDUAL_HOT_LEVEL-input_level)/(RESIDUAL_HOT_LEVEL-RESIDUAL_LEVEL))
    gain_db-=residual
    table=BLEND_LEVEL_DB.get((style,slope))
    if table and kind!='diode': gain_db+=interp(blend,list(table))
    return serum_drive,gain_db


def filter_dry_gain(serum_type: str,drive: float) -> float:
    """Small-signal dry gain of a voice filter; LadderEMS drives only its wet path."""
    return 1.0 if serum_type=='LadderEMS' else (1+min(100.0,max(0.0,drive))/100)**4


def filter_mix_controls(mix: float,gain_db: float,dry_gain: float=1.0,*,fx: bool=False) -> tuple[float,float]:
    """Normalized Wet and LevelOut reproducing Vital's linear dry/wet weights.

    Voice filters linearly mix an already driven dry path. FX filters leave dry undriven,
    use W=sin²(Wet*pi/2), and weight dry by 1-W².
    """
    mix=min(1.0,max(0.0,mix));gain=10**(gain_db/20)
    dry=(1-mix)/dry_gain;wet=mix*gain
    amplitude=(dry+math.hypot(dry,2*wet))/2 if fx else dry+wet
    share=wet/amplitude
    position=2/math.pi*math.asin(math.sqrt(min(1.0,max(0.0,share)))) if fx else share
    return position,level_out(20*math.log10(max(1e-6,amplitude)))


def level_out(gain_db: float) -> float:
    """Serum's filter kParamLevelOut: amplitude is (2x)^2, so 0.5 is unity (measured on 2.1.5)."""
    return min(1.0,max(0.0,0.5*math.sqrt(10**(gain_db/20.0))))


PHASER_RESONANCE_GRID = (0.0,.5,.85,1.0)
# (feedback %, wet %, makeup dB), by polarity and allpass count. The partial wet setting
# supplies the floor under Vital's resonant notches; matching feedback alone leaves them deep.
PHASER_FIT = {
    (0,4):((0,30,50,60),(100,95,85,80),(.65,4.40,7.15,8.1)),
    (0,8):((0,30,40,50),(100,95,95,90),(.65,3.82,6.13,7.1)),
    (1,4):((0,40,90,97),(100,75,75,75),(.65,3.92,6.01,7.5)),
    (1,8):((0,40,80,97),(100,75,65,65),(.65,3.27,4.90,6.0)),
}


def phaser_fit(style: int, stages: int, resonance: float) -> tuple[float,float,float]:
    return tuple(interp(resonance,list(zip(PHASER_RESONANCE_GRID,row))) for row in PHASER_FIT[(style,stages)])


def resonator_params(ctx,prefix: str,model: int,style: int,blend: float) -> dict | None:
    v,r=ctx.v,ctx.r
    from .modulation import active_modulations, amount_modulators
    if prefix=='filter_fx_':
        r.note('unsupported',prefix,'The FX resonator mix and feedback differ from the voice filter; filter bypassed.')
        return None
    if prefix in ('filter_1_','filter_2_') and v.get('filter_1_on') and v.get('filter_2_on'):
        r.note('unsupported',prefix,'Combined voice-filter routing and resonator input levels have no close counterpart; filter bypassed.')
        return None
    mods=active_modulations(v)
    into=amount_modulators(mods)
    moved={m.target for i,m in mods.items() if i not in v.folded and (abs(m.amount)>1e-12 or i in into)}
    if v.get(prefix+'mix')<=0 and prefix+'mix' not in moved:
        r.note('approximation',prefix,'The fully dry resonator is bypassed.')
        return None
    if any(prefix+suffix in moved for suffix in ('blend','blend_transpose','drive')):
        r.note('unsupported',prefix,'Moving stage counts or loop filter shapes have no matching single-filter control; filter bypassed.')
        return None
    resonance=ctx.unit(v.get(prefix+'resonance'),prefix+'resonance')
    if resonance>.850001:
        r.note('unsupported',prefix,'Very high resonator feedback and its nonlinear gain have no calibrated counterpart; filter bypassed.')
        return None
    note=ctx.clip(v.get(prefix+'cutoff'),8,136,prefix+'cutoff')
    drive=ctx.clip(v.get(prefix+'drive'),0,20,prefix+'drive')
    transpose=ctx.clip(v.get(prefix+'blend_transpose'),-84,84,prefix+'blend_transpose')
    internal_wet=1.0
    if model==6:
        if style not in (1,2,4,5):
            r.note('unsupported',prefix,'Comb output filtering has no close single-filter counterpart; filter bypassed.',style=style)
            return None
        error,name,var,reso,note,gain_db=comb_fit(style,note,resonance,blend,transpose)
        if error>4.5:
            r.note('unsupported',prefix,'The comb output filter or narrow loop band has no close single-filter counterpart; filter bypassed.',style=style)
            return None
        feedback=comb_feedback(style,resonance)
        factor=reso/(100*abs(feedback)) if abs(feedback)>1e-6 else 1.0
        law=lambda raw: min(.97,max(0.0,comb_feedback(style,raw)*factor))
        r.note('approximation',prefix,'The closest comb or flange loop is used; its feedback saturation and loop/output filters differ.',target_type=name,user_impact='alternative')
        if drive:
            r.note('approximation',prefix+'drive','The comb model does not use the filter input drive control.')
    else:
        if style>1 or blend>1.05 or drive>.01:
            r.note('unsupported',prefix,'Twelve-stage or driven phaser filtering has no close counterpart; filter bypassed.',style=style)
            return None
        stages=4 if blend<.5 else 8
        name=f'Phase{stages*6}'+('P' if style==0 else 'N')
        reso,wet,gain_db=phaser_fit(style,stages,resonance)
        internal_wet=wet/100
        var=50.0
        note+=11.9
        law=lambda raw: phaser_fit(style,stages,min(1.0,max(0.0,raw)))[0]/100
        r.note('approximation',prefix,'The closest four- or eight-stage phaser is used; feedback and blends between stage counts differ.',target_type=name,user_impact='alternative')
    ctx.filter_notes[prefix]=note
    ctx.filter_laws[prefix]=law
    ctx.filter_cutoffs[prefix]=resonator_cutoff
    ctx.filter_spans[prefix]=RESONATOR_CUTOFF_SPAN
    ctx.filter_disabled_targets[prefix]=('drive','blend','blend_transpose')
    wet_gain=10**(gain_db/20)
    share=lambda raw: internal_wet*raw*wet_gain/(1-raw+raw*wet_gain)
    ctx.filter_mix[prefix]=lambda raw: share(min(1.0,max(0.0,raw)))
    mix=ctx.unit(v.get(prefix+'mix'),prefix+'mix')
    return dict(kParamType=name,kParamFreq=ctx.clip(resonator_cutoff(note),0,1,prefix+'cutoff'),
                kParamReso=reso,kParamVar=var,kParamDrive=0.0,kParamWet=100*share(mix),
                kParamLevelOut=level_out(20*math.log10(max(1e-6,1-mix+mix*wet_gain))))


def filter_params(ctx, prefix: str, *, voice: bool=False) -> dict | None:
    """Serum filter parameters for a Vital filter block, or None when the model is unsupported.

    Besides the parameters, ctx.filter_notes[prefix] gets the MIDI note of the converted cutoff
    (for key tracking) and ctx.filter_laws[prefix] the resonance scale (for modulation).
    """
    v,r=ctx.v,ctx.r
    model=v.integer(prefix+'model',0,7); style=v.integer(prefix+'style',0,9)
    blend=ctx.clip(v.get(prefix+'blend'),0,2,prefix+'blend')
    if not hasattr(ctx,'filter_notes'): ctx.filter_notes,ctx.filter_laws,ctx.filter_mix={},{},{}
    if not hasattr(ctx,'filter_cutoffs'): ctx.filter_cutoffs,ctx.filter_spans,ctx.filter_disabled_targets={},{},{}
    if model in (6,7): return resonator_params(ctx,prefix,model,style,blend)
    if model not in FILTER_KINDS or style>4:
        r.note('unsupported',prefix,'Formant and unknown filter models are not converted; filter bypassed.',model=model,style=style)
        return None
    kind=FILTER_KINDS[model]
    name,var,key=filter_type(model,style,blend)
    slope=key[1]
    resonance=ctx.unit(v.get(prefix+'resonance'),prefix+'resonance')
    reso,_=filter_fit(key,resonance)
    note=ctx.clip(v.get(prefix+'cutoff'),8,136,prefix+'cutoff')
    if kind=='diode':
        mix=min(1.0,blend)
        note+=sum(w*interp(resonance,list(zip(RESONANCE_GRID,row))) for w,row in zip((1-mix,mix),DIODE_CUTOFF))
        if blend>1: r.note('approximation',prefix+'blend',"The diode filter's high-pass half has no counterpart; LadderEMS stands in for the low-pass side.",user_impact='alternative')
    elif key[0]=='ladder mix' or (kind=='ladder' and style>1):
        r.note('approximation',prefix,"Vital's ladder filter away from its low-pass is approximated by Serum's multimode filter.",target_type=name)
    elif style>1 or abs(blend-round(blend))>0.03:
        r.note('approximation',prefix+'blend','Blends between the pure filter shapes differ in level and shape; the level is corrected on average.',target_type=name)
    drive=ctx.clip(v.get(prefix+'drive'),0,20,prefix+'drive')
    mix=ctx.unit(v.get(prefix+'mix'),prefix+'mix')
    from .modulation import active_modulations, amount_modulators
    mods=active_modulations(v);into=amount_modulators(mods)
    moving={m.target for i,m in mods.items() if i not in v.folded and (abs(m.amount)>1e-12 or into.get(i))}
    input_level=source_level(v,prefix if voice else None)
    serum_drive,gain_db=filter_gain(kind,slope,name,key,style,blend,drive,resonance,input_level)
    legacy_mix=voice and name!='LadderEMS' and serum_drive>99.5 and (0<mix<1 or prefix+'mix' in moving)
    if legacy_mix:
        input_level=source_level(v)
        serum_drive,gain_db=filter_gain(kind,slope,name,key,style,blend,drive,resonance,input_level)
        r.note('approximation',prefix+'mix','Dry saturation at the drive limit has no level-independent correction; the previous mix and input-level correction are retained.')
    if input_level is None and serum_drive>99.5:
        r.note('approximation',prefix+'drive','The preceding filter changes the input level; the drive residual uses the reference level without extrapolating from raw source levels.')
    if not hasattr(ctx,'filter_notes'): ctx.filter_notes,ctx.filter_laws,ctx.filter_mix={},{},{}
    ctx.filter_notes[prefix]=note
    ctx.filter_cutoffs[prefix]=serum_cutoff
    ctx.filter_spans[prefix]=SERUM_CUTOFF_SPAN
    ctx.filter_laws[prefix]=lambda raw,key=key: filter_fit(key,min(1.0,max(0.0,raw)))[0]/100
    # Voice-filter drive also amplifies the dry path. The rack filter's dry path is undriven and
    # its mix has a different scale. LevelOut follows either mix, so both weights need correction.
    dry_bypass=False
    if voice and mix<=1e-12:
        if prefix+'mix' not in moving:
            serum_drive,gain_db=0.0,0.0
            ctx.filter_disabled_targets[prefix]=('drive',)
            dry_bypass=True
    if not dry_bypass:
        r.note('approximation',prefix+'drive',"Vital's filter saturates at the level of one oscillator; Serum's drive and output level are set to clip at the same place.")
        if prefix+'drive' in moving:
            r.note('approximation',prefix+'drive_modulation',
                   'Complex or partial-wet drive modulation retains a linear Serum control range; the static saturation and output-gain conversion require correlated control destinations.',user_impact='control_partial')
    if prefix+'blend' in moving:
        r.note('approximation',prefix+'blend_modulation','Moving blend keeps the converted type and linear control range; style-specific shape and gain corrections remain fixed.',user_impact='control_partial')
    if not voice and any(prefix+control in moving for control in ('mix','resonance')):
        r.note('approximation',prefix+'makeup','Moving rack-filter mix or resonance retains the output correction at the stored controls; correlated output-gain modulation is not mapped.',user_impact='control_partial')
    dry_gain=filter_dry_gain(name,serum_drive) if voice and not legacy_mix else 1.0
    controls=lambda raw: filter_mix_controls(raw,gain_db,dry_gain,fx=not voice)
    ctx.filter_mix[prefix]=lambda raw: controls(raw)[0]
    if voice and not legacy_mix:
        if not hasattr(ctx,'filter_makeup_laws'): ctx.filter_makeup_laws={}
        ctx.filter_makeup_laws[prefix]={'mix':controls,
            'resonance':lambda raw: filter_mix_controls(mix,filter_gain(kind,slope,name,key,style,blend,drive,min(1.0,max(0.0,raw)),input_level)[1],dry_gain),
            'drive':lambda raw: filter_mix_controls(mix,filter_gain(kind,slope,name,key,style,blend,min(20.0,max(0.0,raw)),resonance,input_level)[1]),
            'drive_position':lambda raw: filter_gain(kind,slope,name,key,style,blend,min(20.0,max(0.0,raw)),resonance,input_level)[0]/100}
        if mix<1 and dry_gain>1:
            r.note('approximation',prefix+'mix','The driven dry path is corrected at small signal; voice-filter saturation can still change its shape at larger inputs.')
    elif legacy_mix and (prefix+'mix' in moving or prefix+'resonance' in moving):
        r.note('approximation',prefix+'makeup','The uncalibrated dry saturation retains the fixed output correction for moving mix or resonance.')
    wet,level=controls(mix)
    return dict(kParamType=name,
                kParamFreq=min(1.0,max(0.0,serum_cutoff(note))),
                kParamReso=reso,
                kParamDrive=serum_drive,
                kParamWet=100*wet,
                kParamVar=var,
                kParamLevelOut=level)


# --- EQ ---------------------------------------------------------------------
# Both equalisers are plain second-order sections; fitting them to measured responses gave the
# same family on either side (Vital 1.0.7 and Serum 2.1.5, residual below 0.05 dB), so a band
# converts exactly once its corner frequency and Q are translated:
#   pass   s^2 / (s^2 + s/Q + 1) or its low-pass mirror: same frequency, same Q
#   peak   Vital keeps its poles at Q whatever the gain; Serum's peak is symmetric, which is the
#          same curve at Q / sqrt(gain)
#   shelf  the same curve; Serum places its poles a quarter of the gain away from the corner it
#          shows, so its corner goes to Vital's cutoff times gain^(1/4), or over it for the high shelf
EQ_SHELF,EQ_PEAK,EQ_PASS = 0,1,2
VITAL_EQ_MAX_GAIN_DB = 15.0          # the gain control stops here whatever the stored value
# Q of Vital's bands against the resonance control.
VITAL_EQ_Q = [(0.0,0.5),(0.2,0.501),(0.3,0.511),(0.3163,0.516),(0.4,0.564),(0.4473,0.625),(0.5,0.742),
              (0.6,1.223),(0.7,2.324),(0.8,4.563),(0.9,8.737),(1.0,16.0)]
SERUM_EQ_NOTCH_DB = -24.0


def vital_eq_q(resonance: float) -> float:
    return math.exp(interp(resonance,[(r,math.log(q)) for r,q in VITAL_EQ_Q]))


def serum_eq_reso(q: float) -> float:
    """Serum's resonance value for a pass or peak band: Q = 0.1 + 18.9 (reso / 100)^4."""
    return 100*((min(19.0,max(0.1,q))-0.1)/18.9)**0.25


def serum_shelf_reso(q: float) -> float:
    """Serum's resonance value for a shelf: Q = 0.4 + 1.741 (reso / 100)^3."""
    return 100*((min(2.141,max(0.4,q))-0.4)/1.741)**(1/3)


def convert_eq(ctx,entry) -> None:
    """Vital's three EQ bands onto Serum's two-band EQ; a third band gets a second EQ module."""
    v,r=ctx.v,ctx.r
    low=high=None;either=[]
    for side in ('low','band','high'):
        p=f'eq_{side}_'
        mode=v.integer(p+'mode',0,1)
        gain=ctx.clip(v.get(p+'gain'),-VITAL_EQ_MAX_GAIN_DB,VITAL_EQ_MAX_GAIN_DB,p+'gain')
        cutoff=ctx.note_hz(ctx.clip(v.get(p+'cutoff'),8,136,p+'cutoff'))
        q=vital_eq_q(ctx.unit(v.get(p+'resonance'),p+'resonance'))
        amplitude=10**(gain/40)                         # square root of the band's gain
        if side=='band':
            if mode==1:
                # Vital's notch is infinitely deep; Serum's deepest cut with the same poles stands in.
                depth=10**(SERUM_EQ_NOTCH_DB/40)
                either.append(dict(type=EQ_PEAK,freq=cutoff,gain=SERUM_EQ_NOTCH_DB,reso=serum_eq_reso(q/depth)))
                r.note('approximation',p+'mode',"Vital's notch becomes Serum's deepest peak cut (-24 dB).",user_impact='alternative')
            elif abs(gain)>0.05:
                either.append(dict(type=EQ_PEAK,freq=cutoff,gain=gain,reso=serum_eq_reso(q/amplitude)))
            continue
        if mode==1:
            band=dict(type=EQ_PASS,freq=cutoff,gain=0.0,reso=serum_eq_reso(q))
        elif abs(gain)>0.05:
            if q>2.141+1e-6:
                r.note('clipped',p+'resonance',"Serum's shelf cannot resonate as sharply; its highest setting is used.")
            shift=math.sqrt(amplitude)
            band=dict(type=EQ_SHELF,freq=cutoff*shift if side=='low' else cutoff/shift,gain=gain,reso=serum_shelf_reso(q))
        else:
            continue
        if side=='low': low=band
        else: high=band
    # Serum's first band is the low side (low shelf / high-pass), its second the high side.
    modules=[[low,high]]
    for band in either:
        slot=next((m for m in modules if None in m),None)
        if slot is None:
            slot=[None,None];modules.append(slot)
        slot[slot.index(None)]=band
    v.ignore_prefix('eq_')
    if not any(b for m in modules for b in m):
        return
    for module in modules:
        params={}
        for n,band in enumerate(module,1):
            if band is None: continue
            params.update({f'kParamType{n}':float(band['type']),f'kParamFreq{n}':ctx.clip(band['freq'],10,20000,'eq_freq'),
                           f'kParamGain{n}':band['gain'],f'kParamReso{n}':band['reso']})
        if params: entry('FXEQ',7,params)


# --- distortion ---------------------------------------------------------------
# Vital's soft clip is tanh(gain * x) and its hard clip a plain limit at full scale (measured
# with a sine, Vital 1.0.7). Serum's modes were recorded as input/output curves at 21 drive
# settings each (2.1.5); the rows below are the drive and output gain whose curve lies closest
# to Vital's over the lower 70 % of full scale. Serum's drive adds a gain of 0.5 + 2 (drive / 100)^3
# ahead of the curve, and every module's output level is (2 * kParamLevelOut)^2.
#               Vital drive dB: (Serum drive, output gain dB)
SOFT_CLIP = [
    # Smooth from the smallest signal up, like tanh; its upper harmonics run a little strong.
    ('kTube',12.75,[(-30,0.0,-23.98),(-24,0.0,-17.98),(-18,0.5,-12.02),(-12,3.0,-6.21),(-9,6.0,-3.45),(-6,11.5,-0.89),
                    (-3,21.0,1.28),(0,34.5,2.92),(1.5,42.5,3.48),(3,51.5,3.81),(4.5,60.0,4.00),(6,67.5,4.07),
                    (7.5,75.0,3.93),(9,80.5,3.81),(10.5,85.0,3.64),(12,89.0,3.35),(13.5,91.5,3.17)]),
    # Closer once the signal is driven well into the curve; it stays clean at very low levels.
    ('kSoftSat',23.0,[(12,54.5,1.32),(13.5,59.5,1.42),(15,66.0,1.45),(16.5,72.0,1.51),(18,79.0,1.54),(20,89.0,1.57),
                      (22,100.0,1.59),(24,100.0,1.70)]),
    ('kDiode1',math.inf,[(22,65.5,2.72),(24,76.5,2.48),(27,92.0,2.22),(30,100.0,2.16)]),
]
SERUM_HARD_CLIP_CEILING = 0.8278          # of Vital's, which clips at full scale
GAIN_STAGE_MAX = 3.98                     # +12 dB, the most one module's output level adds
HARD_CLIP_MAX_DRIVE_DB = 20*math.log10(2.5/SERUM_HARD_CLIP_CEILING)


def soft_clip(drive_db: float) -> tuple[str,float,float]:
    """(Serum mode, drive, output gain dB) for Vital's soft clip at `drive_db`."""
    for mode,limit,rows in SOFT_CLIP:
        if drive_db<limit:
            return (mode,interp(drive_db,[(d,x) for d,x,_ in rows]),interp(drive_db,[(d,g) for d,_,g in rows]))
    raise AssertionError('unreachable')


def hard_clip(drive_db: float) -> tuple[str,float,float]:
    """(Serum mode, drive, output gain dB) for Vital's hard clip at `drive_db`."""
    gain=10**(drive_db/20)
    wanted=gain*SERUM_HARD_CLIP_CEILING         # gain ahead of Serum's lower ceiling
    if wanted<=0.5:
        # Below Serum's smallest drive gain nothing clips at normal levels; the output makes up the rest.
        return 'kHardClip',0.0,20*math.log10(gain/0.5)
    serum_gain=min(2.5,wanted)
    return 'kHardClip',100*((serum_gain-0.5)/2)**(1/3),-20*math.log10(SERUM_HARD_CLIP_CEILING)


# --- compressor -------------------------------------------------------------
def convert_compressor(ctx,entry) -> None:
    """Vital's multiband compressor on Serum's multiband mode (laws and limits in compressor.py)."""
    v,r=ctx.v,ctx.r
    modules,notes=compressor.convert({k:v.get('compressor_'+k) for k in compressor.KEYS})
    for name,type_id,params in modules: entry(name,type_id,params)
    for kind,feature,message in notes:
        details={'user_impact':'control_partial'} if feature=='compressor_attack' else {}
        r.note(kind,feature,message,**details)
    v.ignore_prefix('compressor_')


def fx_filter_targets(ctx,prefix: str,index: int,params: dict) -> None:
    """Modulation targets and key tracking for a filter at position `index` of the FX rack.

    The rack is shared by all voices, so per-voice sources act through the latest note there.
    Serum's FX filter has no key tracking of its own; the Note source moves its cutoff instead.
    """
    from .modulation import Destination
    v,r=ctx.v,ctx.r
    span=ctx.filter_spans[prefix]
    cutoff=ctx.filter_cutoffs[prefix]
    ctx.target(prefix+'cutoff','FXFilter',index,'kParamFreq',128/span)
    ctx.target(prefix+'resonance','FXFilter',index,'kParamReso',law=ctx.filter_laws[prefix])
    ctx.target(prefix+'mix','FXFilter',index,'kParamWet',law=ctx.filter_mix[prefix])
    for suffix,parameter in (('drive','kParamDrive'),('blend','kParamVar')):
        if suffix in ctx.filter_disabled_targets.get(prefix,()): continue
        ctx.target(prefix+suffix,'FXFilter',index,parameter)
    keytrack=ctx.clip(v.get(prefix+'keytrack'),-1,1,prefix+'keytrack')
    if abs(keytrack)>0.005:
        note=ctx.filter_notes[prefix]+keytrack*(NOTE_SOURCE_CENTRE-VITAL_KEYTRACK_ROOT)
        params['kParamFreq']=min(1.0,max(0.0,cutoff(note)))
        ctx.fixed_routings.append(('note',Destination('FXFilter',index,'kParamFreq'),100*keytrack*127/span,True))
        r.note('approximation',prefix+'keytrack',"Serum's FX filter has no key tracking; the Note source moves its cutoff, so all voices follow the latest note.",user_impact='alternative')


def convert_effects(ctx) -> None:
    v,s,r=ctx.v,ctx.s,ctx.r
    order=decode_order(v.integer('effect_chain_order',0,362879))
    entries=[]
    def val(prefix,key): return v.get(prefix+'_'+key)
    def entry(name,type_id,params,extra=None):
        entries.append({name:{'plainParams':{'kParamEnable':1.0,**params}},
                        'kUIParamMixOrGain':0.0,'type':type_id,**(extra or {})})
        return len(entries)-1
    def gain_stage(amplitude):
        # Serum's EQ with no band set is flat, and its output level gives up to 12 dB of clean gain.
        while amplitude>1.0+1e-9:
            stage=min(amplitude,GAIN_STAGE_MAX);amplitude/=stage
            entry('FXEQ',7,{'kParamLevelOut':level_out(20*math.log10(stage))})
    gain_stage(getattr(ctx,'fx_gain',1.0))
    if getattr(ctx,'extra_fx_filter',None):
        fx_filter_targets(ctx,ctx.extra_fx_prefix,len(entries),ctx.extra_fx_filter)
        entries.append({'FXFilter':{'plainParams':ctx.extra_fx_filter},'kUIParamMixOrGain':0.0,'type':8})
    for effect in order:
        if not v.get(effect+'_on'):
            v.ignore_prefix(effect+'_'); continue
        if effect=='eq':
            convert_eq(ctx,entry); r.mapped.append(effect); continue
        if effect=='compressor':
            convert_compressor(ctx,entry); r.mapped.append(effect); continue
        if effect=='filter_fx':
            params=filter_params(ctx,'filter_fx_')
            if params is None:
                v.ignore_prefix('filter_fx_'); continue
            fx_filter_targets(ctx,'filter_fx_',len(entries),params)
            entry('FXFilter',8,dict(kParamEnable=1.0,**params))
            v.ignore_prefix('filter_fx_'); r.mapped.append(effect); continue
        if effect in ('flanger','phaser'):
            mapped=modulation_fx.convert(ctx,effect,len(entries))
            if mapped is not None:
                entry(*mapped)
                r.mapped.append(effect)
            continue
        if effect not in ('chorus','delay','reverb','distortion'):
            r.note('unsupported',effect,'This effect is not converted; it is omitted from the Serum rack.')
            v.ignore_prefix(effect+'_'); continue
        r.note('approximation',effect,'Effect algorithms and mix laws differ; the reverse mapping is not an audio match.')
        if effect=='delay':
            style=v.integer('delay_style',0,3)
            if style==3:
                r.note('unsupported','delay_style','Mid-ping-pong is reduced to ping-pong.')
            syncs=[v.integer(p+'_sync',0,3) for p in ('delay','delay_aux')]
            if style==0:
                syncs[1]=syncs[0]
                v.ignore_prefix('delay_aux_')
            v.used.update(('delay_frequency','delay_tempo','delay_aux_frequency','delay_aux_tempo'))
            use_sync=all(x>0 for x in syncs)
            times=[]; offsets=[]
            for i,prefix in enumerate(('delay','delay_aux')):
                if style==0 and i==1:
                    times.append(times[0]); offsets.append(offsets[0]); continue
                mode=syncs[i]
                tempo=v.integer(prefix+'_tempo',4,12)
                if use_sync:
                    # Serum triplet is 4/3 of the NEXT faster division.
                    index=tempo+1 if mode==3 else tempo
                    if index>12:
                        use_sync=False
                        r.note('unsupported','delay_sync','1/64 triplet cannot use the measured native ladder; frozen at input BPM.')
                        break
                    times.append(DELAY_KNOBS[index]); offsets.append({2:1.5,3:4/3}.get(mode,1))
                else:
                    t=2**(-val(prefix,'frequency')) if mode==0 else tempo_seconds(tempo,mode,ctx.bps)
                    times.append(ctx.clip(t,.001,4.0,prefix+'_time')); offsets.append(1)
            if len(times)!=2:
                times=[];offsets=[]
                for i,prefix in enumerate(('delay','delay_aux')):
                    mode=syncs[i]
                    t=2**(-val(prefix,'frequency')) if mode==0 else tempo_seconds(v.integer(prefix+'_tempo',4,12),mode,ctx.bps)
                    times.append(ctx.clip(t,.001,4,prefix+'_time'));offsets.append(1)
            if not use_sync and any(syncs):
                r.note('unsupported','delay_sync','Mixed/free or out-of-range sync is converted to fixed seconds at the input BPM.')
            if style==0: times[1],offsets[1]=times[0],offsets[0]
            wet,gain_db=delay_mix(ctx.unit(val(effect,'dry_wet'),'delay_wet'),style)
            delay_mix_control(ctx,style,len(entries))
            entry('FXDelay',4,dict(kParamBeatSync=float(use_sync),kParamMode=1.0 if style>=2 else 0.0,
                kParamTimeL=times[0],kParamTimeR=times[1],kParamOffsetL=offsets[0],kParamOffsetR=offsets[1],
                kParamLink=1.0 if style==0 else 0.0,kParamFeedback=ctx.clip(val(effect,'feedback'),-1,1,'delay_feedback')*100,
                kParamFreq=ctx.note_hz(val(effect,'filter_cutoff')),kParamBW=.75+7.5*ctx.unit(val(effect,'filter_spread'),'delay_filter_spread'),
                kParamWet=wet,kParamLevelOut=level_out(gain_db)))
        elif effect=='chorus':
            sync=v.integer('chorus_sync',0,3)
            v.used.update(('chorus_frequency','chorus_tempo'))
            if sync:
                tempo=v.integer('chorus_tempo',0,10)
                if tempo==0:
                    rate=0.0
                    r.note('unsupported','chorus_sync','Freeze is approximated by a zero-rate chorus.')
                else: rate=1/tempo_seconds(tempo,sync,ctx.bps)
                r.note('unsupported','chorus_sync','Chorus tempo sync is frozen to Hz at the input BPM.')
            else: rate=2**val(effect,'frequency')
            cutoff=val(effect,'cutoff');spread=val(effect,'spread')
            wet,gain_db=wet_mix(ctx.unit(val(effect,'dry_wet'),'chorus_wet'),CHORUS_WET_DB[v.integer('chorus_voices',1,4)],CHORUS_OVERLAP_DB)
            entry('FXChorus',3,dict(kParamBeatSync=0.0,kParamRate=ctx.clip(rate,0,20,'chorus_rate'),
                kParamDelay=1000*2**val(effect,'delay_1'),kParamDelay2=1000*2**val(effect,'delay_2'),
                kParamDepth=26*ctx.unit(val(effect,'mod_depth'),'chorus_depth'),
                kParamFeedback=100*ctx.clip(val(effect,'feedback'),-.95,.95,'chorus_feedback'),
                kParamFilt=min(20000,ctx.note_hz(cutoff+spread*96)),kParamFiltMode=0.0,
                kParamWet=wet,kParamLevelOut=level_out(gain_db)))
            r.note('approximation','chorus_voices',"Serum's chorus has one fixed voice structure; the wet level follows Vital's voice count, the texture does not.",user_impact='alternative')
            r.note('approximation','chorus_filter','The upper cutoff is retained; the separate lower cutoff is not retained.',user_impact='control_partial')
        elif effect=='reverb':
            size=ctx.unit(val(effect,'size'),'reverb_size')
            factor=interp(size,[(0,1.52),(.25,1.16),(.35,1),(.5,.9),(.75,.81),(1,.76)])
            rt60=1.45*2**ctx.clip(val(effect,'decay_time'),-6,6,'reverb_decay_time')*factor
            if rt60<3:
                table=[(0,.2),(10,.4),(20,.94),(35,3.05),(50,7.9),(65,11.9)]
                target=ctx.clip(rt60,.2,11.9,'reverb_rt60')
                serum_size=inverse(target,table);kind='kPlate';decay=0;scale=2
                r.note('approximation','reverb_size','Short decay uses a Plate size fitted to RT60, rather than preserving the size knob.',user_impact='alternative')
            else:
                kind='kHall';scale=1;serum_size=size*100
                floor=interp(serum_size,[(0,3),(50,3.2),(80,5.5),(100,7)])
                if rt60<floor:
                    serum_size=inverse(rt60,[(0,3),(50,3.2),(80,5.5),(100,7)])
                    r.note('approximation','reverb_size','Hall size reduced to avoid its minimum decay floor.',user_impact='alternative')
                decay=30+35*math.log(ctx.clip(rt60,3,60,'reverb_rt60')/3)
            wet,gain_db=wet_mix(ctx.unit(val(effect,'dry_wet'),'reverb_wet'),interp(size,REVERB_WET_DB[kind]),REVERB_OVERLAP_DB)
            entry('FXReverb',6,dict(kParamType=kind,kParamSize=serum_size,kParamDelay=decay,
                kParamWet=wet,kParamLevelOut=level_out(gain_db),
                kParamPreDelay=ctx.clip(val(effect,'delay'),0,.3,'reverb_delay'),kParamWidth=100.0))
            r.note('unsupported','reverb_tone','Reverb internal modulation, pre-filters and shelves are left at Serum defaults.')
            v.ignore_prefix('reverb_')
        elif effect=='distortion':
            ctx.convert_distortion(entry,gain_stage,len(entries))
        r.mapped.append(effect)
    # Vital's stereo routing sits after its effects and mixes each channel into the other:
    # left' = left cos(a) + right sin(a) with a = (1 - routing) pi/4 (measured). That is a mid
    # gain of cos + sin and a side gain of cos - sin, which Serum's utility effect gives with
    # its width (side gain = width / 100, mid untouched) and output level.
    routing=ctx.unit(v.get('stereo_routing'),'stereo_routing')
    if routing<1-1e-6:
        if v.integer('stereo_mode',0,1)==0:
            angle=(1-routing)*math.pi/4
            mid,side=math.cos(angle)+math.sin(angle),math.cos(angle)-math.sin(angle)
            entry('FXUtils',12,{'kParamWidth':100*side/mid,'kParamLevelOut':level_out(20*math.log10(mid))})
            r.mapped.append('stereo_routing')
        else:
            r.note('unsupported','stereo_mode','The rotating stereo mode is not converted; the stereo image stays as it is.')
    else: v.used.add('stereo_mode')
    s.module('FXRack0')['FX']=entries
    for i in (1,2): s.module(f'FXRack{i}')['FX']=[]
