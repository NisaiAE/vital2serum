"""Represent supported global gain after the effect rack.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math
import struct
from dataclasses import dataclass
from .defaults import DEFAULTS
from .modulation import (Destination, RAMP_OFF, RANGES, SOURCES, STATIC_SOURCES,
                         active_modulations, amount_modulators, static_amount)

CURVE_POLICY=(2e-7,480)

@dataclass(frozen=True)
class Control:
    slot: int
    source: str
    coefficient: float
    chain: tuple[int,...]

def amplitude(raw):
    return 0.0 if raw<=0.0 else 10**((math.sqrt(raw)-80.0)/20.0)

def master_value(raw,boost=1.0):
    return amplitude(raw)/math.sqrt(2.0)*boost

def volume_routes(v):
    mods=active_modulations(v);into=amount_modulators(mods)
    routes=[(i,m) for i,m in mods.items() if m.target=='volume' and (abs(m.amount)>1e-12 or into.get(i))]
    return mods,into,routes

def bounded_volume(v):
    stored=v.settings.get('volume',DEFAULTS['volume'])
    offset=v.offsets.get('volume',0.0)
    raw=stored+offset
    lo,hi=RANGES['volume']
    if not isinstance(raw,(int,float)) or not math.isfinite(raw):return False
    if lo<=raw<=hi:return True
    # A stored upper endpoint may round upward in binary32. Its exact alias
    # uses the same clamped source value; offsets and other outside values do not.
    alias=struct.unpack('<f',struct.pack('<f',hi))[0]
    return offset==0.0 and alias>hi and stored==alias

def control(v,ids):
    if not bounded_volume(v):return None,'The requested global gain is outside the supported source range.'
    mods,into,routes=volume_routes(v)
    if len(routes)!=1:return None,'A single active global gain control is required.'
    i,m=routes[0];moved={n.target for n in mods.values()}
    if abs(v.get('velocity_track'))>1e-6:return None,'Velocity tracking and moving global gain retain the original placement.'
    if m.source not in STATIC_SOURCES or m.source in moved or m.stereo or m.bipolar:
        return None,'Only a single unipolar stored global control has a supported output response.'
    if ('FXEQ','kParamLevelOut') not in ids:return None,'The output gain destination identifier is unavailable.'
    fixed=static_amount(v,mods,into,i,moved)
    if fixed is None or abs(fixed[0])<=1e-12:return None,'The complete fixed amount response is unavailable.'
    coefficient,chain=fixed
    for j in chain:
        n=mods[j];prefix=f'modulation_{j}_'
        if j!=i and n.source==m.source:return None,'The primary control also moves its own gain amount.'
        if n.source not in SOURCES or n.stereo:return None,'An amount source is unavailable.'
        lo,hi=RANGES.get(n.source,(0.0,1.0))
        if not all(math.isfinite(x) for x in (lo,hi)) or lo<0.0 or hi>1.0:
            return None,'A source range outside the unit interval is unavailable.'
        if (not math.isfinite(n.power) or any(not math.isfinite(x) or not 0.0<=x<=1.0 for point in n.points for x in point)
                or any(not math.isfinite(x) for x in n.powers)):
            return None,'A remap response outside the unit interval is unavailable.'
        if any(v.settings.get(prefix+k,RAMP_OFF)>RAMP_OFF+1e-6 for k in ('ramp_up','ramp_down')):
            return None,'A primary or auxiliary routing ramp needs a time-dependent response.'
    return Control(i,m.source,coefficient,tuple(chain)),''

def preserve(v,ids,disabled=False):
    candidate,_=control(v,ids) if not disabled else (None,'')
    return {candidate.slot} if candidate is not None else set()

def direct(c):
    return any(c.v.get(('sample_' if slot==3 else f'osc_{slot+1}_')+'destination')>=4 for slot in c.levels)

def clear_clipped_master(c):
    c.r.notes[:]=[n for n in c.r.notes if not (n.get('kind')=='clipped' and n.get('feature')=='master_volume')]

def gain_stages(gain,available):
    """Reserve the smallest complete product within each native stage range."""
    if not math.isfinite(gain) or gain<=0.0 or available<1:return None
    fixed=[];remaining=gain
    while remaining>4.0:
        if len(fixed)+2>available:return None
        fixed.append(4.0);remaining/=4.0
    return tuple(fixed+[remaining])

def register(c,ids,disabled=False):
    """Register a complete output response before native modulation allocation."""
    c.post_master_failed=False;c.post_master_failure=''
    if disabled:return False
    if not bounded_volume(c.v):return False
    mods,into,routes=volume_routes(c.v)
    candidate,reason=control(c.v,ids) if routes else (None,'')
    if routes and candidate is None:return False
    rack=c.s.state.get('FXRack0',{}).get('FX')
    if direct(c) or not isinstance(rack,list) or len(rack)>=10:
        if candidate is not None:c.post_master_failed=True;c.post_master_failure='An independent Direct branch or a full rack prevents an output gain stage.'
        return False
    base=c.v.get('volume');boost=c.master_boost
    if not math.isfinite(boost) or boost<=0:return False
    if candidate is None:
        desired=master_value(base,boost)
        if not math.isfinite(desired) or desired<=1.0:return False
        if abs(c.s.state['Global0']['plainParams'].get('kParamMasterVolume',0)-1.0)>2e-6:return False
        stages=gain_stages(desired,10-len(rack))
        if stages is None:
            c.post_master_failed=True;c.post_master_failure='The complete fixed output gain stages cannot be reserved in the rack.';return False
        index=len(rack)
        for gain in stages:
            rack.append({'FXEQ':{'plainParams':{'kParamEnable':1.0,'kParamLevelOut':.5*math.sqrt(gain)}},'kUIParamMixOrGain':0.0,'type':7})
        clear_clipped_master(c)
        c.r.mapped.append('master_output_gain')
        c.r.note('approximation','master_volume','A final flat output stage restores static master gain above the native master range.',gain=desired,output_stage=index)
        return True
    lo,hi=RANGES['volume'];maximum=max(base,max(lo,min(hi,base+(hi-lo)*candidate.coefficient)))
    wanted=master_value(maximum,boost)
    master=min(1.0,wanted);stages=gain_stages(wanted/master,10-len(rack)) if math.isfinite(wanted) and wanted>0.0 else None
    if stages is None:
        c.post_master_failed=True;c.post_master_failure='The complete output gain stages cannot be reserved in the rack.';return False
    if any(c.source_gain.get(mods[j].source,1.0)!=1.0 for j in candidate.chain):
        c.post_master_failed=True;c.post_master_failure='An amount source has an unsupported depth correction.';return False
    reference=amplitude(maximum);output_factor=wanted/master
    if len(stages)>1:output_factor/=4.0**(len(stages)-1)
    law=lambda raw:.5*math.sqrt(amplitude(raw)/reference*output_factor)
    for gain in stages[:-1]:
        rack.append({'FXEQ':{'plainParams':{'kParamEnable':1.0,'kParamLevelOut':.5*math.sqrt(gain)}},'kUIParamMixOrGain':0.0,'type':7})
    index=len(rack)
    rack.append({'FXEQ':{'plainParams':{'kParamEnable':1.0,'kParamGain1':0.0,'kParamGain2':0.0,'kParamLevelOut':law(base)}},'kUIParamMixOrGain':0.0,'type':7})
    c.s.state['Global0']['plainParams']['kParamMasterVolume']=master
    dest=Destination('FXEQ',index,'kParamLevelOut',law=law,curve_policy=CURVE_POLICY)
    c.destinations['volume']=dest;c.static_aux_law_targets.add('volume')
    def written():
        clear_clipped_master(c)
        c.r.note('approximation','master_volume','The complete global control follows a final flat output stage after the effects; fixed auxiliary control positions are retained in its response.',output_stage=index,static_master=master,response_gain_bound=wanted)
    def omitted():
        c.post_master_failed=True;c.post_master_failure='The complete output curve and modulation resources could not be reserved.'
    if not hasattr(c,'atomic_law_destinations'):c.atomic_law_destinations={}
    c.atomic_law_destinations[candidate.slot]=([dest],written,omitted)
    return True
