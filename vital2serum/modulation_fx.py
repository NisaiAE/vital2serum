"""Modulated effects on Serum's FX rack.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import copy
import math
from .curves import lfo_curve
from .defaults import EFFECT_DEFAULTS
from .modulation import Destination, active_modulations

# FXPhaser's frequency modulation spans 20 Hz to 20 kHz. Vital's depth is in semitones.
PHASER_FREQ_MIN = 20.0
PHASER_FREQ_MAX = 20000.0
PHASER_FREQ_SPAN = 12*math.log2(PHASER_FREQ_MAX/PHASER_FREQ_MIN)
PHASER_CENTRE_OFFSET = -11.0


def frequency_position(note: float) -> float:
    hz=440*2**((note+PHASER_CENTRE_OFFSET-69)/12)
    return min(1.0,max(0.0,math.log2(hz/PHASER_FREQ_MIN)*12/PHASER_FREQ_SPAN))


def mix_position(mix: float) -> float:
    # The Phaser itself mixes linearly in Vital; Serum's wet share is sin^2(w*pi/2).
    return 2/math.pi*math.asin(math.sqrt(min(1.0,max(0.0,mix))))


def _values(ctx,effect: str) -> dict:
    return {suffix:ctx.v.get(effect+'_'+suffix) for suffix in EFFECT_DEFAULTS[effect] if suffix!='on'}


def _free_slot(ctx) -> int | None:
    mods=active_modulations(ctx.v)
    referenced={m.source for m in mods.values()}
    destinations={m.target for m in mods.values()}
    for i in range(8,0,-1):
        prefix=f'lfo_{i}_'
        if f'lfo_{i}' not in referenced and not any(k.startswith(prefix) for k in destinations):
            return i-1
    return None


def _native(ctx,values: dict,params: dict,reason: str) -> None:
    sync=int(values['sync']);tempo=int(values['tempo'])
    rate=2**values['frequency'] if not sync else (0.0 if tempo==0 else ctx.bps/2**(8-tempo)/({2:1.5,3:2/3}.get(sync,1)))
    params.update(kParamRate=rate,kParamDepth=min(100.0,2.7*values['mod_depth']),
                  kParamWidth=min(200.0,360*values['phase_offset']))
    ctx.r.note('approximation','phaser_clock',reason+' Serum\'s native sweep is fixed to Hz at the input tempo and differs in shape and initial phase.',user_impact='control_fixed' if sync else 'alternative')


def _clock(ctx,values: dict,index: int,depth: float) -> bool:
    slot=_free_slot(ctx)
    if slot is None:
        return False
    sync=int(values['sync']);tempo=int(values['tempo'])
    p=dict(kParamMode='Free',kParamSmooth=0.0,kParamPhase=0.0)
    if sync==0:
        p.update(kParamBeatSync=0.0,kParamRate=2**values['frequency'])
    elif tempo==0:
        p.update(kParamBeatSync=0.0,kParamRate=0.0)
    else:
        step=tempo-1.0
        if sync==2:step-=2/3;p['kParamDotted']=1.0
        elif sync==3:step+=2/3;p['kParamTriplets']=1.0
        if step<0:
            # The slower FX period still fits Serum's free-Hz range at the input tempo.
            hz=ctx.bps/2**(8-tempo)/({2:1.5,3:2/3}.get(sync,1))
            p.update(kParamBeatSync=0.0,kParamRate=hz)
            ctx.r.note('approximation','phaser_sync','The slowest dotted period is fixed to Hz at the input tempo.',user_impact='control_fixed')
        else:p.update(kParamBeatSync=1.0,kParamRate=100*(step/14)**4)
    # The quarter-cycle rotation starts at zero displacement; a free clock keeps running across notes.
    curve=lfo_curve([(0,.5),(.25,1),(.75,0),(1,.5)],[0.,0.,0.,0.],False)
    name=f'LFO{slot}'
    original=copy.deepcopy(ctx.s.state.get(name))
    module=ctx.s.set_params(name,p)
    module.update(curveData=curve,pathData={})
    def written():
        ctx.r.note('approximation','phaser_clock','The free triangle clock preserves the sweep period; its initial phase may differ from Vital\'s effect clock.')
        if values['phase_offset']>1e-6:
            ctx.r.note('approximation','phaser_phase_offset','The shared triangle clock reduces the stereo phase offset to a mono sweep.',user_impact='control_partial')
    def omitted():
        if original is None:ctx.s.state.pop(name,None)
        else:ctx.s.state[name]=original
        params=ctx.s.state['FXRack0']['FX'][index]['FXPhaser']['plainParams']
        _native(ctx,values,params,'No modulation slot is available for the triangle clock.')
    ctx.optional_routings.append((f'lfo_{slot+1}',Destination('FXPhaser',index,'kParamFreq'),
                                  200*depth/PHASER_FREQ_SPAN,True,written,omitted))
    return True


def convert(ctx,effect: str,index: int) -> tuple[str,int,dict] | None:
    """Return a rack entry and register its clock and supported modulation targets."""
    if effect=='flanger':
        ctx.v.ignore_prefix('flanger_')
        ctx.r.note('unsupported','flanger','Serum\'s Flanger cannot independently match Vital\'s centre delay and depth; the effect is omitted.')
        return None
    if effect!='phaser':raise ValueError('Unknown modulated effect')
    values=_values(ctx,effect)
    for key,lo,hi in (('center',8,136),('blend',0,2),('dry_wet',0,1),('feedback',0,1),
                      ('mod_depth',0,48),('phase_offset',0,1),('frequency',-5,2),('sync',0,3),('tempo',0,10)):
        values[key]=ctx.clip(values[key],lo,hi,'phaser_'+key)
    for key in ('sync','tempo'):
        values[key]=ctx.v.integer('phaser_'+key,0,3 if key=='sync' else 10)
    blend=values['blend'];mix=values['dry_wet'];depth=values['mod_depth']
    if depth>0 and values['sync']!=0 and values['tempo']==0:
        ctx.v.ignore_prefix('phaser_')
        ctx.r.note('unsupported','phaser_freeze','A frozen Phaser with nonzero depth is omitted; its stationary response cannot be matched reliably.')
        return None
    poles=2+2*math.floor(blend+.5)
    freq=440*2**((values['center']+PHASER_CENTRE_OFFSET-69)/12)
    params=dict(kParamRate=0.0,kParamBeatSync=0.0,kParamDepth=0.0,kParamDepth2=0.0,
                kParamNumPoles=float(poles),kParamFreq=ctx.clip(freq,PHASER_FREQ_MIN,PHASER_FREQ_MAX,'phaser_frequency_hz'),
                kParamFeedback=100*values['feedback'],kParamWet=100*mix_position(mix),kParamWidth=0.0,kParamLevelOut=.5)
    moving_mix=any(m.target=='phaser_dry_wet' for m in active_modulations(ctx.v).values())
    if depth>0 and (mix>0 or moving_mix):
        low=440*2**((values['center']+PHASER_CENTRE_OFFSET-depth-69)/12)
        high=440*2**((values['center']+PHASER_CENTRE_OFFSET+depth-69)/12)
        if low<PHASER_FREQ_MIN or high>PHASER_FREQ_MAX:
            ctx.r.note('clipped','phaser_sweep','The Phaser sweep reaches Serum\'s frequency limits.',
                       low_hz=low,high_hz=high,target_min_hz=PHASER_FREQ_MIN,target_max_hz=PHASER_FREQ_MAX)
        if not _clock(ctx,values,index,depth):
            _native(ctx,values,params,'All LFOs are in use.')
    ctx.target('phaser_center','FXPhaser',index,'kParamFreq',law=frequency_position)
    ctx.target('phaser_feedback','FXPhaser',index,'kParamFeedback')
    ctx.target('phaser_dry_wet','FXPhaser',index,'kParamWet',law=mix_position)
    ctx.r.note('approximation','phaser','Phaser stages, feedback and intermediate blends use the nearest Serum response; the effect algorithms differ.')
    ctx.v.ignore_prefix('phaser_')
    return 'FXPhaser',2,params
