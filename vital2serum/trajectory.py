"""Single-source wavetable trajectories for correlated oscillator controls.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .assets import _asset, wav_bytes
from .modulation import RAMP_OFF, SOURCES, active_modulations, amount_modulators
from .ranges import RANGES
from .spectral import TRAJECTORY_MORPHS, TableSpectrum
from .wavetable import (BAKED_MORPHS, FRAME_SIZE, PHASE_WARPS, SWEEP_FRAMES,
                        phase_warp, render_wavetable, spectral_morph)


def source_grid(count: int = SWEEP_FRAMES) -> np.ndarray:
    """Uniform source samples including both clamped endpoints."""
    if count<1:raise ValueError('A trajectory needs at least one frame')
    return np.linspace(0.0,1.0,count)


def position_law(count: int = SWEEP_FRAMES) -> tuple[float, float]:
    """TablePos base and percent depth selecting round((count - 1) * source).

    Serum selects floor(count * (TablePos - 1) / 255). The half-bucket base keeps
    each endpoint sample at its bucket centre while the depth covers every interval.
    """
    if count<1:raise ValueError('A trajectory needs at least one frame')
    return (1.0+255.0*.5/count,100.0*(count-1)/count) if count>1 else (1.0,0.0)


@dataclass
class Trajectory:
    prefix: str
    source: str
    routes: tuple
    bases: dict[str, float]
    warp_kind: int
    warp_phase: float
    morph_kind: int

    def values(self, raw: float) -> dict[str, float]:
        values=dict(self.bases)
        for route in self.routes:
            lo,hi=RANGES[route.target]
            values[route.target]+=route.amount*(hi-lo)*route.value(raw)
        return {key:max(RANGES[key][0],min(RANGES[key][1],value)) for key,value in values.items()}


def correlated_controls(v, prefix: str) -> Trajectory | None:
    """Moving frame, warp and morph controls driven by one raw source."""
    used=set(v.used)
    try:
        return _correlated_controls(v,prefix)
    finally:
        v.used.clear();v.used.update(used)


def _correlated_controls(v, prefix: str) -> Trajectory | None:
    mods=active_modulations(v);into=amount_modulators(mods)
    active={i:m for i,m in mods.items() if i not in v.folded and (abs(m.amount)>1e-12 or into.get(i))}
    kind=v.integer(prefix+'distortion_type',0,12)
    morph_kind=v.integer(prefix+'spectral_morph_type',0,11)
    if kind not in (0,*PHASE_WARPS) or morph_kind not in (0,*BAKED_MORPHS,*TRAJECTORY_MORPHS):
        return None
    if morph_kind and abs(v.get(prefix+'spectral_morph_phase')-.5)>1e-6:
        return None
    targets={prefix+'wave_frame'}
    if kind:targets.add(prefix+'distortion_amount')
    if morph_kind:targets.add(prefix+'spectral_morph_amount')
    routes=tuple(m for m in active.values() if m.target in targets)
    if morph_kind in TRAJECTORY_MORPHS and {m.target for m in routes}!={prefix+'spectral_morph_amount'}:
        return None
    single_spectral=morph_kind in TRAJECTORY_MORPHS
    if (len({m.target for m in routes})<2 and not single_spectral) or len({m.source for m in routes})!=1:
        return None
    source=routes[0].source
    if source not in SOURCES or any(m.stereo or into.get(m.slot) for m in routes):
        return None
    domain=RANGES.get(source,(0.0,1.0))
    if domain[0]<0.0 or domain[1]>1.0:
        return None
    if any(m.target==prefix+'unison_voices' and m.source==source for m in active.values()):
        return None
    if any(v.settings.get(f'modulation_{m.slot}_{key}',RAMP_OFF)>RAMP_OFF+1e-6
           for m in routes for key in ('ramp_up','ramp_down')):
        return None
    moved={m.target for m in active.values()}
    if any(prefix+key in moved for key in ('distortion_phase','spectral_morph_phase','distortion_type','spectral_morph_type')):
        return None
    if (v.get(prefix+'unison_voices')>1 or prefix+'unison_voices' in moved) and any(abs(v.get(prefix+key))>1e-9 or prefix+key in moved
           for key in ('frame_spread','distortion_spread','spectral_morph_spread')):
        return None
    bases={prefix+key:v.get(prefix+key) for key in ('wave_frame','distortion_amount','spectral_morph_amount')}
    return Trajectory(prefix,source,routes,bases,kind,v.get(prefix+'distortion_phase'),morph_kind)


def render_trajectory(frames: np.ndarray, plan: Trajectory, count: int = SWEEP_FRAMES) -> np.ndarray:
    """Evaluate all original route shapes, sum each control, clamp, then morph and warp."""
    source=np.asarray(frames,dtype=np.float64)
    morph_moves=any(m.target==plan.prefix+'spectral_morph_amount' for m in plan.routes)
    if plan.morph_kind and not morph_moves and plan.morph_kind not in TRAJECTORY_MORPHS:
        source=spectral_morph(source,plan.morph_kind,plan.bases[plan.prefix+'spectral_morph_amount'])
    new_spectrum=TableSpectrum.from_frames(source) if plan.morph_kind in TRAJECTORY_MORPHS else None
    out=[]
    for raw in source_grid(count):
        values=plan.values(float(raw))
        index=min(int(values[plan.prefix+'wave_frame']),source.shape[0]-1)
        frame=source[index:index+1]
        if new_spectrum is not None:
            frame=new_spectrum.frame(plan.morph_kind,values[plan.prefix+'spectral_morph_amount'],index).reshape(1,FRAME_SIZE)
        elif plan.morph_kind and morph_moves:
            frame=spectral_morph(frame,plan.morph_kind,values[plan.prefix+'spectral_morph_amount'])
        if plan.warp_kind:
            frame=phase_warp(frame,plan.warp_kind,values[plan.prefix+'distortion_amount'],plan.warp_phase)
        out.append(frame)
    return np.concatenate(out)


def trajectory_asset(table: dict, report, feature: str, plan: Trajectory, headroom: float = 1.0):
    frames,notes=render_wavetable(table,feature,full=plan.morph_kind in TRAJECTORY_MORPHS)
    for note in notes: report.note('approximation',feature,note)
    rendered=render_trajectory(frames,plan)
    flat=rendered.reshape(-1).astype(np.float64)
    gain=max(float(np.abs(flat).max()),headroom)
    flat=(flat/gain).astype(np.float32)
    name=table.get('name') if isinstance(table.get('name'),str) else ''
    asset=_asset(wav_bytes(flat.astype('<f4').tobytes(),1,44100,True,True),'Tables',
                 rendered.shape[0]*FRAME_SIZE,pcm=[flat.tolist()],display_name=name or 'Vital table')
    asset.gain=gain
    return asset
