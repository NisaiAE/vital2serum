"""Pair a bounded moving PD warp with a closed carrier phase bank.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import copy
import math
import numpy as np
from . import codec, fm
from .assets import _asset, wav_bytes
from .curves import remap_curve
from .errors import ConversionError
from .modulation import MAX_SLOTS, RAMP_OFF, SOURCES, active_modulations, amount_modulators
from .serum import plain
from .trajectory import position_law
from .wavetable import render_wavetable

FRAMES=256
MAX_POINTS=480
TOLERANCE=2e-5


def paired_points(xs,ws):
    """Bound polynomial secants of 1-8*w^4 on each serialized linear warp segment.

    The bound precedes coordinate rounding and excludes native parameter smoothing.
    """
    if len(xs)!=len(ws) or len(xs)<2 or any(b<=a for a,b in zip(xs,xs[1:])):
        raise ValueError('Invalid complete warp response')
    points={float(x):1.-8.*float(w)**4 for x,w in zip(xs,ws)}
    pending=list(zip(xs[:-1],xs[1:],ws[:-1],ws[1:]))
    if len(points)>MAX_POINTS:raise ValueError('Phase curve capacity exhausted')
    while pending:
        a,b,w0,w1=pending.pop()
        # max|q''| * interval_width^2 / 8 for a linear w on this interval.
        if 12.*max(w0*w0,w1*w1)*(w1-w0)**2<=TOLERANCE:continue
        x=(a+b)/2.;w=(w0+w1)/2.
        if not a<x<b or x in points or len(points)>=MAX_POINTS:
            raise ValueError('Phase curve capacity or coordinate resolution exhausted')
        points[x]=1.-8.*w**4
        pending.extend(((a,x,w0,w),(x,b,w,w1)))
    return sorted(points.items())


def _samples(c,slot):
    wt=c.s.state[f'Oscillator{slot}'].get(f'WTOsc{slot}',{})
    if (wt.get('numFrames')!=2048 or wt.get('numChannels')!=1 or wt.get('sampleRate')!=44100
            or wt.get('flex') not in ({},None)):
        return None
    owned={'plainParams','flex','numChannels','numFrames','sampleRate','embeddedWTData',
           'relativePathToWT','interpolateAfterLoad','tableDisplayName'}
    if set(wt)-owned:return None
    if c.embed:
        data=wt.get('embeddedWTData')
    else:
        reference=wt.get('relativePathToWT')
        asset=next((a for a in c.assets.values() if a.serum_reference==reference),None)
        if asset is None or asset.gain!=1:return None
        data=asset.pcm[0] if len(asset.pcm)==1 else None
    if data is None:return None
    values=np.asarray(data,dtype=np.float32)
    if values.shape!=(2048,) or not np.isfinite(values).all() or np.max(np.abs(values))>1:
        return None
    return values


def _eligible(c,carrier,mods,into):
    v=c.v;p=f'osc_{carrier}_';target=p+'distortion_amount'
    if p in c.disabled_phase_controls or not v.get(p+'on') or v.get(p+'distortion_type') not in (7,8):return None
    roots=[m for i,m in mods.items() if i not in v.folded and m.target==target and (abs(m.amount)>1e-12 or into.get(i))]
    if len(roots)!=1:return None
    m=roots[0]
    if (m.source not in ('macro_control_1','macro_control_2','macro_control_3','macro_control_4','env_2')
            or m.bipolar or m.stereo or m.power!=0 or m.points or into.get(m.slot)
            or v.get(target)!=0 or not 0<m.amount<=.35):return None
    if any(v.settings.get(f'modulation_{m.slot}_'+key,RAMP_OFF)>RAMP_OFF+1e-6 for key in ('ramp_up','ramp_down')):return None
    active={i:n for i,n in mods.items() if i not in v.folded and (abs(n.amount)>1e-12 or into.get(i))}
    other=f'osc_{fm.modulator(carrier,int(v.get(p+"distortion_type")))}_'
    moved={n.target for n in active.values()}
    if any(key.startswith(p) and key!=target for key in moved):return None
    if any(key.startswith(other) and key[len(other):] not in ('level','pan','destination') for key in moved):return None
    if m.source in moved or any(key.startswith(m.source+'_') for key in moved):return None
    fixed={'phase':.5,'random_phase':0,'unison_voices':1,'unison_detune':0,'stack_style':0,
           'stereo_spread':1,'wave_frame':0,'spectral_morph_type':0,'frame_spread':0,
           'distortion_spread':0,'spectral_morph_spread':0}
    if any(v.get(prefix+key)!=value for prefix in (p,other) for key,value in fixed.items()):return None
    if (not v.get(other+'on') or v.get(other+'distortion_type')!=0 or
            v.get(p+'transpose')!=0 or v.get(other+'transpose')!=0 or
            v.get(p+'tune')!=0 or v.get(other+'tune')!=0 or
            v.get(p+'midi_track')!=1 or v.get(other+'midi_track')!=1):return None
    # Do not change a carrier that also supplies another active cross-modulation chain.
    for j in (1,2,3):
        q=f'osc_{j}_';kind=int(v.get(q+'distortion_type'))
        if (j!=carrier and (v.get(q+'on') or q+'on' in moved) and
                (fm.modulator(j,kind)==carrier or q+'distortion_type' in moved) and
                (v.get(q+'distortion_amount')>0 or q+'distortion_amount' in moved or
                 q+'distortion_type' in moved)):
            return None
    if c.master_boost>1+1e-9 or c.fx_gain>1+1e-9:return None
    for note in c.r.notes:
        feature=note.get('feature','')
        affected=(feature.startswith(p) or feature==m.source or feature.startswith(m.source+'_') or
                  (feature.startswith(other) and feature[len(other):] not in ('level','pan','destination')))
        if note.get('kind') in ('unsupported','clipped') and affected:return None
    supplier=int(other[4])-1
    for slot in (carrier-1,supplier):
        body=c.s.state[f'Oscillator{slot}']
        owned={'kParamEnable','kParamPan','kParamOctave','kParamPitch','kParamFine',
               'kParamUnison','kParamDetune','kParamVolume'}
        if set(plain(body))-owned:return None
    donor_params=plain(c.s.state[f'Oscillator{supplier}'].get(f'WTOsc{supplier}',{}))
    if (set(donor_params)-{'kParamTablePos','kParamInitialPhase','kParamRandomPhase'} or
            donor_params.get('kParamTablePos')!=1 or donor_params.get('kParamInitialPhase')!=0 or
            donor_params.get('kParamRandomPhase')!=0):return None
    donor=_samples(c,supplier);base=_samples(c,carrier-1)
    if donor is None or base is None:return None
    sine=np.sin(2*np.pi*np.arange(2048)/2048).astype(np.float32)
    if not np.array_equal(donor,sine):return None
    try:
        original,_=render_wavetable(v.settings['wavetables'][int(other[4])-1],other+'wavetable')
    except (ConversionError,ValueError,KeyError,TypeError,IndexError):return None
    if original.shape!=(1,2048) or not np.array_equal(original[0].astype(np.float32),donor):return None
    wp=plain(c.s.state[f'Oscillator{carrier-1}'][f'WTOsc{carrier-1}'])
    if (set(wp)-{'kParamTablePos','kParamInitialPhase','kParamRandomPhase','kParamWarpMenu','kParamWarp'} or
            wp.get('kParamWarpMenu')!=fm.MENUS[int(v.get(p+'distortion_type'))] or
            wp.get('kParamWarp')!=0 or wp.get('kParamTablePos')!=1):return None
    record=next((r for r in c.r.modulations if r.get('vital_slot')==m.slot),None)
    if not record or record.get('status')!='mapped':return None
    return p,m,base


def _bank(base):
    # The unique real Nyquist bin has no independent quadrature for phase rotation.
    # Keep only a residue at the binary64 representation floor; do not alter its gain.
    nyquist=math.fsum(float(x) if i%2==0 else -float(x) for i,x in enumerate(base))
    floor=np.finfo(float).eps*math.fsum(abs(float(x)) for x in base)
    if abs(nyquist)>floor:
        raise ValueError('Carrier Nyquist component cannot retain phase-rotation amplitude')
    frequencies=np.fft.fftfreq(2048)*2048;spectrum=np.fft.fft(base.astype(float))
    frames=np.array([np.fft.ifft(spectrum*np.exp(2j*np.pi*frequencies*q)).real
                     for q in np.arange(FRAMES)/(FRAMES-1)],dtype=np.float32)
    frames[0]=base;frames[-1]=base
    if not np.isfinite(frames).all() or np.max(np.abs(frames))>1:
        raise ValueError('Carrier phase bank exceeds unchanged source headroom')
    flat=frames.reshape(-1)
    return _asset(wav_bytes(flat.astype('<f4').tobytes(),1,44100,True,True),'Tables',len(flat),
                  pcm=[flat.tolist()],display_name='Anonymous phase bank')


def apply(c,ids):
    """Reserve all representations, then commit; failures request a pure conversion retry."""
    mods=active_modulations(c.v);into=amount_modulators(mods)
    active={k:v for k,v in c.s.state.items() if k.startswith('ModSlot') and v.get('destModuleParamName')}
    used={int(k[7:]) for k in active};plans=[];failed=set()
    for carrier in (1,2,3):
        eligible=_eligible(c,carrier,mods,into)
        if eligible is None:continue
        p,m,base=eligible;index=carrier-1
        try:
            if c.source_gain.get(m.source,1.)!=1.:raise ValueError('Phase source gain differs')
            number=ids.get(('WTOsc','kParamTablePos'));warp_id=ids.get(('WTOsc','kParamWarp'))
            free=next((i for i in range(MAX_SLOTS) if i not in used),None)
            if number is None or warp_id is None or free is None:raise ValueError('Complete phase destination or slot unavailable')
            slots=[v for v in active.values() if v.get('destModuleTypeString')=='WTOsc' and v.get('destModuleID')==index]
            if len(slots)!=1:raise ValueError('Native carrier has more than one control dependency')
            slot=slots[0];sp=plain(slot)
            if (slot.get('destModuleParamName')!='kParamWarp' or slot.get('destModuleParamID')!=warp_id
                    or slot.get('source')!=[SOURCES[m.source],0] or sp.get('kParamBipolar',0)!=0
                    or sp.get('kParamMainCurveData')!=1):raise ValueError('Incomplete primary warp response')
            if set(sp)-{'kParamAmount','kParamBipolar','kParamMainCurveData'}:raise ValueError('Unmeasured warp flags')
            curve=slot['flex'][0]
            if any(abs(x-.5)>1e-10 for x in curve['curveVals']):raise ValueError('Nonlinear native warp interpolation')
            xs=np.asarray(curve['xVals'],dtype=np.float32).astype(float)
            ws=sp['kParamAmount']/100*(1-np.asarray(curve['yVals'],dtype=np.float32).astype(float))
            if (not np.isfinite(xs).all() or xs.ndim!=1 or xs[0]!=0 or xs[-1]!=1 or
                    not np.isfinite(ws).all() or min(ws)<0 or 8*max(ws)**4>=1):
                raise ValueError('PD response leaves the complete no-wrap range')
            points=paired_points(xs,ws);asset=_bank(base)
            plans.append((index,p,free,slot,number,points,asset));used.add(free)
        except (ValueError,KeyError,TypeError,OverflowError):failed.add(p)
    if failed:return failed
    if not plans:return set()
    replacement=copy.deepcopy(c.s)
    for index,p,free,slot,number,points,asset in plans:
        wt=replacement.state[f'Oscillator{index}'][f'WTOsc{index}'];wp=plain(wt)
        old_reference=wt.get('relativePathToWT');table_base,depth=position_law(FRAMES)
        wt.update(numFrames=asset.frames,interpolateAfterLoad=0,tableDisplayName=asset.display_name)
        wp.update(kParamInitialPhase=0.,kParamTablePos=table_base)
        if c.embed:wt['embeddedWTData']=codec.Float32List(asset.pcm[0])
        else:wt['relativePathToWT']=asset.serum_reference
        new=copy.deepcopy(slot);new.update(destModuleParamName='kParamTablePos',destModuleParamID=number)
        new['plainParams']=dict(kParamAmount=depth,kParamBipolar=0.,kParamMainCurveData=1.)
        new['flex']=[remap_curve(points),{}];replacement.state[f'ModSlot{free}']=new
        if c.embed:
            ordinal=sum(bool(c.v.get(f'osc_{j}_on')) for j in range(1,index+1))
            c.r.assets[ordinal]=dict(embedded=True,name=asset.display_name,sample_count=asset.frames,channels=1,sample_rate=44100)
        else:
            shared=any(body.get(f'WTOsc{i}',{}).get('relativePathToWT')==old_reference for i,body in
                       ((i,replacement.state[f'Oscillator{i}']) for i in range(3)) if i!=index)
            old=next((a for a in c.assets.values() if a.serum_reference==old_reference),None)
            record=dict(path=asset.relative_path,bytes=len(asset.data),sample_count=asset.frames,channels=1,sample_rate=44100)
            registered=asset.relative_path in c.assets
            if old and not shared:
                c.assets.pop(old.relative_path,None)
                c.r.assets[:]=[a for a in c.r.assets if a.get('path')!=old.relative_path]
            if not registered:c.r.assets.append(record)
            c.assets[asset.relative_path]=asset
        c.r.note('approximation',p+'phase_response',
                 'A bounded moving FM amount follows a closed carrier phase bank paired with the native warp curve. Table quantization, parameter smoothing, source and output-stage differences remain.',
                 source=slot['source'],
                 phase_entries=len(points),phase_slot=free)
    c.s=replacement
    return set()
