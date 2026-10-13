"""Keep native ring-modulation balance and carrier gain correlated.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from . import fm
from .curves import vital_remap
from .modulation import (Destination, RAMP_OFF, SOURCES, active_modulations,
                         amount_modulators)
from .ranges import RANGES

CURVE_POLICY=(2e-5,480)


@dataclass
class Control:
    prefix: str
    route: object
    ratio: float | None
    level: float
    gain: float
    maximum_gain: float


def _amount_bounds(base, route):
    # Each remap segment and the following power are monotone; knots contain its extrema.
    readings=[route.value(0.),route.value(1.)]
    readings.extend(vital_remap(1.-y,route.power,False) for _,y in route.points)
    return sorted(max(0.,min(1.,base+route.amount*value)) for value in (min(readings),max(readings)))


def prepare(ctx, carrier: int, ratio: float | None) -> bool:
    """Reserve gain headroom for one unit-source RM amount on a fixed signal structure."""
    prefix=f'osc_{carrier}_';target=prefix+'distortion_amount'
    if prefix in ctx.disabled_ring_controls:return False
    if ratio is not None and abs(ratio-fm.SERUM_RM_GAIN)<=1e-10:return False
    v=ctx.v;mods=active_modulations(v);into=amount_modulators(mods)
    active={i:m for i,m in mods.items() if i not in v.folded and (abs(m.amount)>1e-12 or into.get(i))}
    roots=[m for m in active.values() if m.target==target]
    if len(roots)!=1:return False
    route=roots[0]
    domain=RANGES.get(route.source,(0.,1.))
    if (route.source not in SOURCES or route.stereo or route.bipolar or into.get(route.slot)
            or domain[0]<0 or domain[1]>1 or abs(route.amount)<=1e-12):return False
    if any(v.settings.get(f'modulation_{route.slot}_'+key,RAMP_OFF)>RAMP_OFF+1e-6
           for key in ('ramp_up','ramp_down')):return False
    moved={m.target for m in active.values()}
    if any(prefix+key in moved for key in ('level','on','distortion_type','unison_voices',
           'unison_blend','stereo_spread','stack_style')):return False
    modulator=fm.modulator(carrier,int(v.get(prefix+'distortion_type')))
    if modulator is None:return False
    other=f'osc_{modulator}_'
    if other+'on' in moved:return False
    if ratio is not None and any(key.startswith(other) and key[len(other):] not in
                                ('level','pan','destination') for key in moved):return False
    if ratio is not None and (v.get(other+'unison_voices')!=1 or v.get(other+'stack_style')!=0):return False
    if ratio is not None and v.get(other+'distortion_type') in (7,8,9,10,11,12):return False
    if ratio is not None and (not math.isfinite(ratio) or ratio<=0):return False
    level,gain,feature=ctx.levels[carrier-1]
    bounds=_amount_bounds(v.get(target),route)
    maximum=max(fm.rm_gain(amount,ratio) for amount in bounds)
    if not math.isfinite(maximum) or maximum<=0:return False
    control=Control(prefix,route,ratio,level,gain,maximum)
    ctx.ring_controls.append(control)
    ctx.levels[carrier-1]=(level,gain*maximum,feature)
    return True


def register(ctx) -> None:
    """Write complete Volume/Warp responses together or request a pure re-conversion."""
    lowered=max(ctx.master_boost,ctx.fx_gain)
    for control in ctx.ring_controls:
        prefix=control.prefix;slot=int(prefix[4])-1;target=prefix+'distortion_amount'
        ratio=control.ratio
        dependent=fm.modulator(slot+1,int(ctx.v.get(prefix+'distortion_type')))
        other=f'osc_{dependent}_'
        if ratio is not None and (any(n['kind'] in ('unsupported','clipped') and
                  n.get('feature','').startswith(other) for n in ctx.r.notes) or
                  any(key.startswith(other) and key[len(other):] not in ('level','pan','destination')
                      and key not in ctx.v.used and ctx.v.changed(key) for key in ctx.v.settings)):
            ctx.failed_ring_controls.add(prefix)
            continue
        def position(amount,c=control):
            return c.level*math.sqrt(c.gain*fm.rm_gain(amount,c.ratio)/lowered)
        base=position(ctx.v.get(target))
        maximum=control.level**2*control.gain*control.maximum_gain/lowered
        if not math.isfinite(base) or not 0<=base<=1 or maximum>1+1e-12:
            ctx.failed_ring_controls.add(prefix)
            continue
        ctx.s.state[f'Oscillator{slot}']['plainParams']['kParamVolume']=base*base
        destinations=[]
        if ratio is not None:
            destinations.append(Destination('WTOsc',slot,'kParamWarp',
                                law=lambda amount,r=ratio:fm.rm_warp(amount,r),curve_policy=CURVE_POLICY))
        destinations.append(Destination('Oscillator',slot,'kParamVolume',law=position,curve_policy=CURVE_POLICY))
        def written(p=prefix,r=ratio,maximum=maximum):
            ctx.r.note('approximation',p+'ring_gain',
                       'One ring-modulation amount follows the native balance and carrier gain together; source, unison and engine differences remain.',
                       source_ratio=r,maximum_carrier_amplitude=maximum)
        def omitted(p=prefix):ctx.failed_ring_controls.add(p)
        if not hasattr(ctx,'atomic_law_destinations'):ctx.atomic_law_destinations={}
        ctx.atomic_law_destinations[control.route.slot]=(destinations,written,omitted)
