"""Vital -> Serum 2 mapping. This is a lossy reverse, not a bijection.

The reference converter's known units and enums are inverted where possible.
Unknown state is reported rather than assigned guessed parameter identifiers.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
import re
from . import codec, distortion_model, filter_structure, fm, phase_modulation, post_master, ring_modulation, sampler, trajectory, unison, voice_gain
from .assets import Asset, wavetable, sample
from .curves import curve_value, envelope_tail_curve, lfo_curve, power_scale, sampled_curve
from .defaults import DEFAULTS
from .effects import (FILTER_KINDS, NOTE_SOURCE_CENTRE, SERUM_CUTOFF_BASE, SERUM_CUTOFF_SPAN, SERUM_KEYTRACK_ROOT, VITAL_KEYTRACK_ROOT,
                      HARD_CLIP_MAX_DRIVE_DB, convert_effects, decode_order, filter_params, hard_clip, serum_cutoff, soft_clip)
from .errors import ConversionError
from .modulation import (Destination, KNOWN_IDS, SOURCES, STATIC_SOURCES, _source_curve, active_modulations, amount_modulators,
                         convert_modulations, finalize_modulation_reporting, mix_route_supported, static_amount)
from .ranges import RANGES
from .report import Report
from .serum import SerumPreset, clean_template, plain
from .vital import VitalPreset
from .spectral import STATIC_MORPHS, TRAJECTORY_MORPHS, static_eligible
from .wavetable import BAKED_MORPHS, MORPH_PHASE_DISPERSE, PHASE_WARPS, serum_table_position, sweep_position


@dataclass
class Result:
    preset: SerumPreset
    assets: list[Asset]
    report: Report


class Context:
    def __init__(self,v: VitalPreset,s: SerumPreset,r: Report,embed: bool=True):
        self.v,self.s,self.r=v,s,r
        self.embed=embed
        self.assets: dict[str,Asset]={}
        self.destinations: dict[str,Destination]={}
        self.supplementary_destinations: dict[str,list[Destination]]={}
        self.static_aux_law_targets: set[str]=set()
        self.complete_control_buses: dict[int,tuple]={}
        self.bps=self.clip(v.get('beats_per_minute'),1/3,5,'beats_per_minute')
        self.primary_filter: int | None=None
        self.filters_written: int=0
        self.voice_filter_slots: dict[int,int]={}
        self.native_filter_routing: bool=False
        # Filter routes that cannot be represented per voice keep the FX-rack approximation.
        self.extra_fx_filter: dict | None=None
        self.fixed_routings: list=[]       # (source, Destination, amount %, bipolar) written as they are
        self.optional_routings: list=[]    # written after user routes; callbacks handle success or fallback
        # Amplitude wanted from each source (oscillators 0..2, sample 3) before Serum's 0 dB
        # ceiling: (Vital level, gain on top of level squared, feature name).
        self.levels: dict[int,tuple[float,float,str]]={}
        self.master_boost: float=1.0
        # Gain to restore ahead of the effects when a level-dependent effect follows (see _levels).
        self.fx_gain: float=1.0
        # Depth correction per modulation source (a free-running LFO whose smoothing differs).
        self.source_gain: dict[str,float]={}

    def clip(self,n: float,lo: float,hi: float,feature: str) -> float:
        if not isinstance(n,(int,float)) or not math.isfinite(n):
            raise ConversionError(f'{feature}: expected a finite number')
        result=max(lo,min(hi,n))
        if abs(n-result)>1e-7:
            self.r.note('clipped',feature,'Value clamped to the supported target range.',original=n,converted=result)
        return float(result)

    def unit(self,n: float,feature: str) -> float:
        return self.clip(n,0,1,feature)

    @staticmethod
    def note_hz(note: float) -> float:
        if not -128<=note<=256: raise ConversionError('Frequency note outside allowed range')
        return 440*2**((note-69)/12)

    def target(self,key: str,module: str,index: int,parameter: str,scale: float=1,nonlinear: bool=False,law=None):
        self.destinations[key]=Destination(module,index,parameter,scale,nonlinear,law)

    def convert_distortion(self,entry,gain_stage,index: int):
        values={key:self.v.get('distortion_'+key) for key in distortion_model.KEYS}
        kind=self.v.integer('distortion_type',0,5)
        values['type']=kind
        values['filter_order']=self.v.integer('distortion_filter_order',0,2)
        drive=self.clip(values['drive'],-30,30,'distortion_drive')
        values['drive']=drive
        mods=active_modulations(self.v);into=amount_modulators(mods)
        moving=[(i,m) for i,m in mods.items() if i not in self.v.folded and m.target=='distortion_drive'
                and (abs(m.amount)>1e-12 or into.get(i))]
        dynamic=False;partial=False;bus=None;input_position=None
        original_values=dict(values)
        fold=kind in (distortion_model.LINEAR_FOLD,distortion_model.SINE_FOLD)
        span=max(distortion_model.SHAPER_IN,sum(level**2*gain for level,gain,_ in self.levels.values()))
        fold_room=(span<=distortion_model.LEVEL_OUT_MAX*distortion_model.SHAPER_IN and
                   (kind!=distortion_model.SINE_FOLD or
                    distortion_model.vital_gain(30)*span*4<=(distortion_model.SHAPER_ENTRIES-1)//2))
        if moving and kind in (distortion_model.SOFT_CLIP,distortion_model.HARD_CLIP,
                               distortion_model.LINEAR_FOLD,distortion_model.SINE_FOLD):
            moved={m.target for m in mods.values()}
            fixed=static_amount(self.v,mods,into,moving[0][0],moved) if len(moving)==1 else None
            if len(moving)==1 and fixed is None and not fold:
                i,m=moving[0];aux=into.get(i,())
                if len(aux)==1:
                    j=aux[0];a=mods[j]
                    if (abs(m.amount)<1e-12 and not m.bipolar and not m.stereo and m.source.startswith('lfo_')
                            and not a.bipolar and not a.stereo and a.source.startswith('lfo_') and a.amount>0
                            and not into.get(j) and abs(m.power)<.01 and abs(a.power)<.01 and not m.points and not a.points
                            and self.source_gain.get(m.source,1.0)==1.0 and self.source_gain.get(a.source,1.0)==1.0):
                        bus=(i,m,a)
            filter_moves=any(j not in self.v.folded and m.target.startswith('distortion_filter_')
                             and (abs(m.amount)>1e-12 or into.get(j)) for j,m in mods.items())
            mix_moves=any(j not in self.v.folded and m.target=='distortion_mix'
                          and (abs(m.amount)>1e-12 or into.get(j)) for j,m in mods.items())
            order=decode_order(int(self.v.get('effect_chain_order')))
            future=sum(2 if effect=='eq' else 1 for effect in order[order.index('distortion')+1:] if self.v.get(effect+'_on'))
            future+=int(self.v.get('stereo_routing')<1-1e-6 and self.v.integer('stereo_mode',0,1)==0)
            # This helper has been calibrated with at most nine rack entries.
            rack_room=index+6+future<=9
            partial=(kind==distortion_model.SOFT_CLIP and 0.0<values['mix']<1.0-1e-7
                     and values['filter_order']==0 and not mix_moves and len(moving)==1
                     and not moving[0][1].stereo and not moving[0][1].bipolar and fixed is not None
                     and self.source_gain.get(moving[0][1].source,1.0)==1.0 and rack_room)
            if moving and fold and not fold_room:
                self.r.note('unsupported','distortion_drive',
                            'The input range leaves too few shaper points for a moving fold drive; the stored drive is retained.')
            if (partial or (values['mix']>=1.0-1e-7 and values['filter_order'] in ((0,) if fold else (0,2))
                    and (not fold or fold_room) and not filter_moves and len(moving)==1
                    and not moving[0][1].stereo and not moving[0][1].bipolar and (fixed is not None or bus is not None))):
                input_position=entry('FXEQ',7,{'kParamLevelOut':distortion_model.drive_input_level(-30 if bus else drive)})
                if bus is None and not partial:
                    self.target('distortion_drive','FXEQ',input_position,'kParamLevelOut',law=distortion_model.drive_input_level)
                    self.static_aux_law_targets.add('distortion_drive')
                drive=values['drive']=distortion_model.MAX_DRIVE_DB
                dynamic=True
                if not partial:
                    self.r.note('approximation','distortion_drive',
                                'A full-wet distortion drive uses a fixed transfer curve and a moving input gain; incomplete helper routings retain the stored drive.')
                if values['filter_order']==2:
                    self.r.note('approximation','distortion_filter',
                                'The fixed native post-filter can differ by several dB at high input levels.')
                for j in fixed[1][1:] if fixed else [bus[2].slot]:
                    if any(self.v.settings.get(f'modulation_{j}_{p}',-10)>-10+1e-6 for p in ('ramp_up','ramp_down')):
                        self.r.note('unsupported',f'modulation_{j}_ramp',
                                    'Auxiliary modulation ramp times are not converted; the auxiliary follows its source at once.')
        if values['filter_order']==1 and kind in (distortion_model.SOFT_CLIP,distortion_model.HARD_CLIP):
            _,filter_gain=distortion_model.filter_params(1,values['filter_cutoff'],values['filter_blend'],values['filter_resonance'])
            drive+=20*math.log10(filter_gain)
        clip=None
        if kind==distortion_model.SOFT_CLIP:
            if drive>30.0:
                self.r.note('clipped','distortion_drive','The pre-filter gain needs more drive than the soft-clip approximation covers.',original=drive,converted=30.0)
            clip=('kXShaperAsym',0.0,20*math.log10(1.0/distortion_model.SHAPER_OUT)) if dynamic else soft_clip(min(30.0,drive))
        elif kind==distortion_model.HARD_CLIP:
            pre_gain=10**(max(0.0,drive-HARD_CLIP_MAX_DRIVE_DB)/20)
            gain_stage(pre_gain)
            clip=(*hard_clip(min(drive,HARD_CLIP_MAX_DRIVE_DB)),pre_gain)
        modules,notes=distortion_model.convert(values,clip=clip,span=span)
        for name,type_id,params,extra in modules:
            if dynamic and kind==distortion_model.SOFT_CLIP and name=='FXDistortion':
                curve=distortion_model.soft_clip_curve()
                extra={'flex':[curve,dict(curve)]}
                if not partial:
                    self.r.note('approximation','distortion_type',
                                'The soft clip is drawn into X-Shaper, which adds a 12 Hz high-pass and a short processing delay.')
            if partial and name=='FXDistortion':
                wet,level=distortion_model.partial_drive_params(original_values['drive'],original_values['mix'])
                params={**params,'kParamWet':wet,'kParamLevelOut':level}
            position=entry(name,type_id,params,extra)
            if name=='FXDistortion' and not dynamic:
                distortion_model.register_mix_control(self,values,clip,span,position)
            if partial and name=='FXDistortion':
                output_positions=[position]+[entry('FXEQ',7,{'kParamLevelOut':level}) for _ in range(4)]
                self._distortion_partial_drive(moving[0][0],input_position,output_positions,original_values)
            if name=='FXDistortion' and values['filter_order']:
                self.target('distortion_filter_cutoff','FXDistortion',position,'kParamFreq',128/distortion_model.SERUM_FILTER_SPAN)
            if name=='FXDistortion' and kind==distortion_model.DOWN_SAMPLE and len(moving)==1 and not moving[0][1].stereo:
                i,m=moving[0]
                fixed=static_amount(self.v,mods,into,i,{z.target for z in mods.values()})
                if fixed is not None:
                    self.target('distortion_drive','FXDistortion',position,'kParamDrive',
                                law=lambda d:distortion_model.downsample_drive(d)/100)
                    self.static_aux_law_targets.add('distortion_drive')
                    self.r.note('approximation','distortion_drive',
                                'The native hold control assumes a 44.1 kHz host rate and has a long-hold upper limit; omitted drive routes retain the stored hold.')
            if name=='FXDistortion' and dynamic and bus is not None:
                self._distortion_drive_bus(bus,input_position,position,original_values)
        for note_kind,feature,message in notes: self.r.note(note_kind,feature,message)
        self.v.ignore_prefix('distortion_')

    def _distortion_partial_drive(self,slot: int,input_position: int,outputs: list[int],values: dict):
        mix=values['mix'];clip_position=outputs[0]
        destinations=[Destination('FXEQ',input_position,'kParamLevelOut',law=distortion_model.drive_input_level),
                      Destination('FXDistortion',clip_position,'kParamWet',
                                  law=lambda drive:distortion_model.partial_drive_params(drive,mix)[0]/100.0)]
        for position in outputs:
            module='FXDistortion' if position==clip_position else 'FXEQ'
            destinations.append(Destination(module,position,'kParamLevelOut',
                                            law=lambda drive:distortion_model.partial_drive_params(drive,mix)[1]))
        self.static_aux_law_targets.add('distortion_drive')
        def on_written():
            self.r.note('approximation','distortion_drive',
                        'A fixed soft-clip curve follows the input gain, wet control, and five correlated output stages together; the dry level can differ slightly at very low input gain.')
            self.r.note('approximation','distortion_type',
                        'The soft clip is drawn into X-Shaper, which adds a 12 Hz high-pass and a short processing delay.')
        def on_omitted():
            entries=self.s.state['FXRack0']['FX']
            for position in [input_position]+outputs[1:]:
                entries[position]['FXEQ']['plainParams']={'kParamEnable':1.0,'kParamLevelOut':.5}
            modules,_=distortion_model.convert(values,clip=soft_clip(values['drive']))
            entry=entries[clip_position];entry.pop('flex',None)
            entry['FXDistortion']['plainParams']={'kParamEnable':1.0,**modules[-1][2]}
            self.r.note('unsupported','distortion_drive',
                        'The correlated dry and wet controls could not be written together; the stored distortion drive is retained.')
        if not hasattr(self,'atomic_law_destinations'):self.atomic_law_destinations={}
        self.atomic_law_destinations[slot]=(destinations,on_written,on_omitted)

    def _distortion_drive_bus(self,bus,input_position: int,clip_position: int,values: dict):
        i,main,aux=bus
        low=distortion_model.drive_input_level(-30);high=distortion_model.drive_input_level(30)
        curve=sampled_curve(lambda x:(distortion_model.drive_input_level(60*x-30)-low)/(high-low))
        auxiliary=sampled_curve(lambda x:min(1.0,2*aux.amount*aux.value(x)))
        def on_written():
            self.r.note('approximation','distortion_drive_aux',
                        'One moving LFO auxiliary is combined before the drive response; the helper control can add a short processing delay.')
        def on_omitted():
            kind=int(values['type']);drive=values['drive'];pre=1.0
            if kind==distortion_model.SOFT_CLIP: clip=soft_clip(drive)
            else:
                pre=10**(max(0.0,drive-HARD_CLIP_MAX_DRIVE_DB)/20)
                clip=(*hard_clip(min(drive,HARD_CLIP_MAX_DRIVE_DB)),pre)
            modules,_=distortion_model.convert(values,clip=clip)
            entries=self.s.state['FXRack0']['FX']
            for position in range(input_position,clip_position):
                gain=min(pre,3.98);pre/=gain
                entries[position]['FXEQ']['plainParams']={'kParamEnable':1.0,'kParamLevelOut':.5*math.sqrt(gain)}
            entry=entries[clip_position];entry.pop('flex',None)
            entry['FXDistortion']['plainParams']={'kParamEnable':1.0,**modules[-1][2]}
            self.r.note('unsupported','distortion_drive_aux',
                        'No complete helper control and modulation group was available; the stored distortion drive is retained.')
        if not hasattr(self,'complete_control_buses'): self.complete_control_buses={}
        self.complete_control_buses[i]=((values['drive']+30)/60,main.source,aux.source,_source_curve(main),auxiliary,
                                       Destination('FXEQ',input_position,'kParamLevelOut'),100*(high-low),curve,on_written,on_omitted)

    def asset(self,a: Asset,feature: str):
        if self.embed:
            self.r.assets.append(dict(embedded=True,name=a.display_name,sample_count=a.frames,
                                      channels=a.channels,sample_rate=a.sample_rate))
        elif a.relative_path not in self.assets:
            self.assets[a.relative_path]=a
            self.r.assets.append(dict(path=a.relative_path,bytes=len(a.data),sample_count=a.frames,
                                      channels=a.channels,sample_rate=a.sample_rate))
        self.r.mapped.append(feature)

    def routing(self,source_slot: int,vital_dest: int,feature: str):
        # Measured on Serum 2.1.5: 'Filter' runs the voice filters and then the FX rack,
        # 'Master' skips the voice filters but still runs the FX rack, and 'Direct' skips both.
        balance=None
        if vital_dest==4:                      # Vital: direct out, past the effects
            target='kRoutingDestDirect'
        elif vital_dest==3:                    # Vital: effects, past the filters
            target='kRoutingDestMaster'
        elif vital_dest in (0,1,2):
            intended=vital_dest+1 if vital_dest!=2 else None
            if intended and not self.v.get(f'filter_{intended}_on'):
                target='kRoutingDestMaster'
            elif self.native_filter_routing and intended in self.voice_filter_slots:
                target='kRoutingDestFilter'
                balance=-100.0 if self.voice_filter_slots[intended]==0 else 100.0
            elif self.primary_filter is None:
                target='kRoutingDestMaster'
                self.r.note('unsupported',feature,'Unavailable filter route bypassed to the FX chain.')
            elif self.filters_written==1 and intended is not None:
                remaining=intended
                if intended!=self.primary_filter and self.v.get('filter_1_on') and self.v.get('filter_2_on'):
                    if self.v.get('filter_1_filter_input'):
                        if intended==2: remaining=1
                    elif self.v.get('filter_2_filter_input'):
                        if intended==1: remaining=2
                target='kRoutingDestFilter' if remaining==self.primary_filter else 'kRoutingDestMaster'
                if intended!=self.primary_filter:
                    self.r.note('unsupported',feature,
                                'The unavailable stage is bypassed; this branch reaches the surviving filter.' if target=='kRoutingDestFilter'
                                else 'The unavailable independent filter branch bypasses to the FX chain.')
            else:
                target='kRoutingDestFilter'
                if vital_dest==2 or intended!=self.primary_filter:
                    self.r.note('unsupported',feature,'Dual/secondary filter routing reduced to the selected primary filter.')
        else:
            target='kRoutingDestMaster'
            self.r.note('unsupported',feature,'Advanced Vital output routing reduced to the FX chain.',destination=vital_dest)
        params=dict(kParamRoutingDest=target,kParamFXBus1Level=0.0,kParamFXBus2Level=0.0)
        if balance is not None: params['kParamFilterBalance']=balance
        self.s.set_params(f'RoutingSlot{source_slot}',params)


def _native_filter_order(v: VitalPreset,active: list[int]) -> tuple[list[int],bool] | None:
    """Two standard filters with independent source inputs, in processing order."""
    if active!=[1,2]: return None
    for i in active:
        if v.integer(f'filter_{i}_model',0,7) not in FILTER_KINDS or v.integer(f'filter_{i}_style',0,9)>4:
            return None
    for p in ('osc_1_','osc_2_','osc_3_','sample_'):
        if v.get(p+'on') and v.integer(p+'destination',0,14) not in (0,1,3,4):
            return None
    # Vital processes the reverse chain first when both serial input switches are set.
    if v.integer('filter_1_filter_input',0,1): return [2,1],True
    return [1,2],bool(v.integer('filter_2_filter_input',0,1))


def _voice_filter(c: Context,selected: int,index: int,params: dict) -> None:
    """Write one voice filter and register its own tracking and modulation targets."""
    p=f'filter_{selected}_'
    keytrack=c.clip(c.v.get(p+'keytrack'),-1,1,p+'keytrack')
    cutoff=c.filter_cutoffs[p];span=c.filter_spans[p];note=c.filter_notes[p]
    if abs(keytrack-1)<0.02 and cutoff(note-(VITAL_KEYTRACK_ROOT-SERUM_KEYTRACK_ROOT))>=0:
        params.update(kParamKeyTrack=1.0,kParamFreq=cutoff(note-(VITAL_KEYTRACK_ROOT-SERUM_KEYTRACK_ROOT)))
    elif abs(keytrack)>0.005:
        params['kParamFreq']=min(1.0,max(0.0,cutoff(note+keytrack*(NOTE_SOURCE_CENTRE-VITAL_KEYTRACK_ROOT))))
        c.fixed_routings.append(('note',Destination('VoiceFilter',index,'kParamFreq'),100*keytrack*127/span,True))
    c.s.set_params(f'VoiceFilter{index}',dict(kParamEnable=1.0,**params))
    c.voice_filter_slots[selected]=index
    c.r.mapped.append(f'filter_{selected}->VoiceFilter{index}')
    c.target(p+'cutoff','VoiceFilter',index,'kParamFreq',128/span)
    c.target(p+'resonance','VoiceFilter',index,'kParamReso',law=c.filter_laws[p])
    c.target(p+'mix','VoiceFilter',index,'kParamWet',law=c.filter_mix[p])
    for suffix,param in [('drive','kParamDrive'),('blend','kParamVar')]:
        if suffix in c.filter_disabled_targets.get(p,()): continue
        laws=getattr(c,'filter_drive_laws' if suffix=='drive' else 'filter_var_laws',{})
        c.target(p+suffix,'VoiceFilter',index,param,law=laws.get(p))
    _filter_makeup_targets(c,p,index)


def _filter_makeup_targets(c: Context,prefix: str,index: int) -> None:
    """Correlated wet/level routes when one standard filter control moves independently."""
    laws=getattr(c,'filter_makeup_laws',{}).get(prefix)
    if laws is None: return
    mods=active_modulations(c.v);into=amount_modulators(mods)
    controls={'mix','resonance','drive','blend'}
    moving=[(i,m) for i,m in mods.items() if i not in c.v.folded and m.target in {prefix+x for x in controls}
            and (abs(m.amount)>1e-12 or into.get(i))]
    if not moving: return
    if (len(moving)!=1 or moving[0][1].target[len(prefix):] not in laws
            or (moving[0][1].target==prefix+'drive' and c.v.get(prefix+'mix')<1.0-1e-7)):
        c.r.note('approximation',prefix+'makeup','Moving multiple filter controls or drive/blend retains the fixed output correction.',user_impact='control_partial')
        return
    i,m=moving[0];control=m.target[len(prefix):]
    if into.get(i) or m.stereo:
        c.r.note('approximation',prefix+'makeup','Auxiliary or stereo modulation retains the fixed output correction.',user_impact='control_partial')
        return
    if control=='resonance' and c.v.get(prefix+'model')==4:
        c.r.note('approximation',prefix+'makeup','The diode cutoff correction also depends on resonance; moving resonance retains the fixed output correction.',user_impact='control_partial')
        return
    if control=='drive':
        c.target(prefix+'drive','VoiceFilter',index,'kParamDrive',law=laws['drive_position'])
        c.r.note('approximation',prefix+'drive_modulation',
                 'A single full-wet drive follows the static saturation and output-gain conversion; input-dependent residuals remain tied to the static input estimate.')
    law=laws[control]
    extras=[]
    if control=='resonance': extras.append(Destination('VoiceFilter',index,'kParamWet',law=lambda raw: law(raw)[0]))
    extras.append(Destination('VoiceFilter',index,'kParamLevelOut',law=lambda raw: law(raw)[1]))
    c.supplementary_destinations[m.target]=extras


def _filters(c: Context):
    v,s,r=c.v,c.s,c.r
    active=[i for i in (1,2) if v.integer(f'filter_{i}_on',0,1)]
    for i in (0,1): s.set_params(f'VoiceFilter{i}',{'kParamEnable':0.0})
    if not active:
        for i in (1,2): v.ignore_prefix(f'filter_{i}_')
        return
    native=_native_filter_order(v,active)
    if native is not None:
        order,serial=native
        c.native_filter_routing=True
        c.primary_filter=order[0]
        for index,selected in enumerate(order):
            _voice_filter(c,selected,index,filter_params(c,f'filter_{selected}_',voice=True))
        for index in (0,1):
            s.set_params(f'RoutingSlot{index+5}',dict(kParamRoutingDest='kRoutingDestFilter' if serial and index==0 else 'kRoutingDestMaster',
                                                    kParamFXBus1Level=0.0,kParamFXBus2Level=0.0))
        c.filters_written=2
        return
    if len(active)==2:
        r.note('approximation','filter_routing','This dual-filter source route cannot be represented by the native voice chain; the supported stages use the FX-rack approximation.',user_impact='alternative')
    written=0
    for selected in active:
        p=f'filter_{selected}_'
        params=filter_params(c,p,voice=written==0)
        if params is None:
            v.ignore_prefix(p); continue
        if written==0:
            c.primary_filter=selected
            _voice_filter(c,selected,0,params)
        else:
            c.extra_fx_filter=dict(kParamEnable=1.0,**params);c.extra_fx_prefix=p
            r.mapped.append(f'filter_{selected}->FXFilter')
            if not v.get(p+'filter_input'):
                r.note('approximation','filter_routing','Vital runs the two filters in parallel; Serum chains them.')
        written+=1
    if not written:
        for i in (1,2): v.ignore_prefix(f'filter_{i}_')
        return
    c.filters_written=written
    for i in (1,2):
        if i not in active: v.ignore_prefix(f'filter_{i}_')
    return


def _fold_fixed_warp_axis(c: Context):
    """Fix a stored warp control when doing so leaves a single wavetable axis.

    Keep auxiliary slots available to other routes: only the routes into this warp are folded.
    """
    v=c.v
    mods=active_modulations(v);into=amount_modulators(mods)
    moved={m.target for m in mods.values()}
    remaining={i:m for i,m in mods.items() if i not in v.folded}
    moving={m.target for m in remaining.values()}
    for i in range(1,4):
        p=f'osc_{i}_';target=p+'distortion_amount'
        if (p not in getattr(c,'disabled_trajectories',()) and
                trajectory.correlated_controls(v,p) is not None):
            continue
        if not v.get(p+'on') or v.integer(p+'distortion_type',0,12) not in PHASE_WARPS:
            continue
        frame_moves=p+'wave_frame' in moving
        morph_moves=(v.integer(p+'spectral_morph_type',0,11) in BAKED_MORPHS
                     and p+'spectral_morph_amount' in moving)
        if int(frame_moves)+int(morph_moves)!=1 or p+'distortion_phase' in moved:
            continue
        if v.get(p+'unison_voices')>1 and abs(v.get(p+'distortion_spread'))>1e-9:
            continue
        roots=[j for j,m in remaining.items() if m.target==target]
        if not roots: continue
        offsets=[]
        for j in roots:
            m=mods[j]
            if m.source not in STATIC_SOURCES or m.source in moved or m.stereo:
                break
            amount=static_amount(v,mods,into,j,moved)
            if amount is None: break
            offsets.append(amount[0]*m.value(v.get(m.source)))
        else:
            v.offsets[target]=v.offsets.get(target,0.0)+sum(offsets)
            for j in roots:
                v.folded.add(j);v.ignore_prefix(f'modulation_{j}_')
            c.r.note('approximation',target,
                     'A fixed macro/mod wheel warp was baked at its stored position to retain the remaining wavetable axis; changing the folded control no longer moves this warp.',
                     slots=roots,remaining_axis='frame' if frame_moves else 'morph',
                     user_impact='control_fixed',sources=sorted({mods[j].source for j in roots}),destination=target)


def _shepard_pitch(c: Context,p: str,slot: int,params: dict,amount: float) -> None:
    """Partial pitch law; spectral blending and wrap phase history stay unconverted."""
    v,r=c.v,c.r
    mods=active_modulations(v);into=amount_modulators(mods)
    active={i:m for i,m in mods.items() if i not in v.folded}
    moved={m.target for m in active.values()}
    if (v.get(p+'spectral_morph_spread')!=0 or v.get(p+'frame_spread')!=0 or v.get(p+'wave_frame')!=0
            or v.get(p+'distortion_type')!=0 or p+'transpose' in moved or p+'wave_frame' in moved
            or p+'spectral_morph_phase' in moved):
        return
    roots=[(i,m) for i,m in active.items() if m.target==p+'spectral_morph_amount'
           and (abs(m.amount)>1e-12 or into.get(i))]
    if len(roots)>1: return
    low=high=amount
    if roots:
        i,m=roots[0]
        if (into.get(i) or not m.source.startswith('lfo_') or m.source not in SOURCES
                or m.stereo or m.bipolar or abs(m.power)>=.01 or m.points):
            return
        low,high=sorted((amount,amount+m.amount))
        source_prefix=m.source+'_'
        if any(source_prefix+x in moved for x in ('phase','frequency','tempo','sync','sync_type','smooth_time','smooth_mode')):
            return
        if v.get(source_prefix+'sync') and not v.get(source_prefix+'tempo'): return
        if v.get(source_prefix+'sync_type') not in (0,1) or v.get(source_prefix+'sync')==4: return
        # Optional routings have a linear source. Exclude sources needing depth compensation.
        if v.get(source_prefix+'sync') in (0,4):
            hz=2**v.get(source_prefix+'frequency')
            tau=2**v.get(source_prefix+'smooth_time')/math.log(32) if v.get(source_prefix+'smooth_mode') else 0.
            native_tau=max(SERUM_MIN_SMOOTH_S,tau) if tau>=.001 else 0.
            gain=math.sqrt((1+(2*math.pi*hz*native_tau)**2)/(1+(2*math.pi*hz*tau)**2))
            if abs(gain-1)>.005: return
        # Keep optional pitch from competing with correlated FX/filter output controls.
        if any(v.get(key) for key in ('filter_1_on','filter_2_on','filter_fx_on','chorus_on','compressor_on',
                                      'delay_on','distortion_on','eq_on','flanger_on','phaser_on','reverb_on')):
            return
    if low<0 or high>1: return
    base=float(params.get('kParamCoarsePit',0.))
    if min(base-12*(1-low),base-12*(1-high)) < -64 or max(base-12*(1-low),base-12*(1-high)) >64:
        return
    correction=-12*(1-amount)
    def written():
        params['kParamCoarsePit']=base+correction
        r.note('approximation',p+'shepard_pitch',
               'Only the Shepard pitch ramp is approximated; spectral blending and wrap phase history remain unconverted.',
               semitones=correction,dynamic=bool(roots))
    if not roots:
        written();return
    i,m=roots[0]
    c.optional_routings.append((m.source,Destination('Oscillator',slot,'kParamCoarsePit'),
                               100*12*m.amount/128,False,written,
                               lambda:r.note('approximation',p+'shepard_pitch',
                                             'The partial pitch ramp needs a modulation slot; the original pitch is retained.')))


def _oscillators(c: Context,allow_placeholder: bool):
    v,s,r=c.v,c.s,c.r
    _fold_fixed_warp_axis(c)
    tables=v.array('wavetables')
    global_transpose=v.integer('voice_transpose',-48,48)
    global_tune=c.clip(v.get('voice_tune'),-1,1,'voice_tune')
    stacks={0:None,1:'kCenter12',2:'kCenter24',3:'kOctave1',4:'kOctave2',5:'kOctaveFifth1',6:'kOctaveFifth2'}
    # Serum's own warps, for what cannot be rendered into the table. kParamWarp runs 0..1 (measured).
    warps={1:'kSync',3:'kQuantize',5:'kBendPosNeg',6:'kPWM',9:'kPD_NOISE',12:'kRM_NOISE'}
    gains,tables_params,cross={},{},[]
    # Destinations a moving source still modulates (stored macro positions are folded by now).
    moving={m.target for i,m in active_modulations(v).items() if i not in v.folded}
    # An oscillator ring-modulating another with a moving amount gets a table at half level:
    # Serum's RM is then Vital's at every amount (fm.SERUM_RM_GAIN), not only at the stored one.
    ring_sources={fm.modulator(i,v.integer(f'osc_{i}_distortion_type',0,12)) for i in range(1,4)
                  if v.integer(f'osc_{i}_on',0,1) and v.integer(f'osc_{i}_distortion_type',0,12) in (10,11)
                  and f'osc_{i}_distortion_amount' in moving}
    for i in range(1,4):
        p=f'osc_{i}_';slot=i-1
        enabled=v.integer(p+'on',0,1)
        m=s.set_params(f'Oscillator{slot}',{'kParamEnable':float(enabled)})
        if not enabled:
            v.ignore_prefix(p); continue
        # A spectral morph or a phase warp with a fixed amount is a fixed table: render it into the
        # frames. Where the amount moves and the table position does not, the table becomes a sweep
        # over the amount and the modulation drives the position instead.
        morph_type=v.integer(p+'spectral_morph_type',0,11)
        morph_amount=c.unit(v.get(p+'spectral_morph_amount'),p+'spectral_morph_amount')
        wtype=v.integer(p+'distortion_type',0,12)
        amount=c.unit(v.get(p+'distortion_amount'),p+'distortion_amount')
        wave_frame=c.clip(v.get(p+'wave_frame'),0,256,p+'wave_frame')
        frame_moves=p+'wave_frame' in moving
        warp_moves=wtype in PHASE_WARPS and p+'distortion_amount' in moving
        morph_moves=morph_type in BAKED_MORPHS and p+'spectral_morph_amount' in moving
        sweep='warp' if warp_moves and not frame_moves else 'morph' if morph_moves and not frame_moves and not warp_moves else None
        native_warp=warp_moves and sweep!='warp'           # frame and warp both move: Serum's own warp
        joint=(trajectory.correlated_controls(v,p) if p not in getattr(c,'disabled_trajectories',())
               and not allow_placeholder else None)
        new_static=static_eligible(v,p)
        morph=(morph_type,morph_amount) if morph_type in BAKED_MORPHS or new_static else None
        if joint is not None and morph_type in TRAJECTORY_MORPHS:
            morph=(morph_type,morph_amount)
            morph_moves=p+'spectral_morph_amount' in moving
        warp=(wtype,amount,c.unit(v.get(p+'distortion_phase'),p+'distortion_phase')) if wtype in PHASE_WARPS and not native_warp else None
        if joint is not None:
            a=trajectory.trajectory_asset(tables[slot] if slot<len(tables) else None,r,p+'wavetable',joint,
                                          fm.SERUM_RM_GAIN if i in ring_sources else 1.0)
            sweep='trajectory';native_warp=False
            _,depth=trajectory.position_law(a.frames//2048)
            c.table_trajectories.append((joint,Destination('WTOsc',slot,'kParamTablePos',depth/100)))
        else:
            a=wavetable(tables[slot] if slot<len(tables) else None,r,p+'wavetable',allow_placeholder,morph,warp,sweep,wave_frame,
                        fm.SERUM_RM_GAIN if i in ring_sources else 1.0)
        c.asset(a,p+'wavetable')
        transpose=c.clip(v.get(p+'transpose')+global_transpose,-48,48,p+'transpose')
        octave=math.trunc(transpose/12);pitch=transpose-octave*12
        # kParamType is left at its default (wavetable); 'kOsc_Wavetable' is never stored by Serum itself.
        voices=v.integer(p+'unison_voices',1,16)
        # Serum's kParamDetuneWid is the unison blend and kParamUnisonStereo the width (measured).
        # Blend, width and the level of the whole stack come from the measured voice levels.
        blend,width,unison_db=unison.match(voices,c.unit(v.get(p+'unison_blend'),p+'unison_blend'),
                                           c.unit(v.get(p+'stereo_spread'),p+'stereo_spread'))
        # Vital level is quadratic; Serum 2 stores the oscillator gain as linear amplitude. The
        # table headroom and the unison difference ride on it; _levels() writes kParamVolume.
        c.levels[slot]=(c.unit(v.get(p+'level'),p+'level'),a.gain*10**(unison_db/20),p+'level')
        detune_range=c.clip(v.get(p+'detune_range'),0,48,p+'detune_range')
        params=dict(kParamEnable=1.0,
            kParamPan=50*c.clip(v.get(p+'pan'),-1,1,p+'pan'),kParamOctave=float(octave),kParamPitch=pitch,
            kParamFine=100*c.clip(v.get(p+'tune')+global_tune,-1,1,p+'tune'),
            kParamUnison=float(voices),
            kParamDetune=unison.serum_detune(c.clip(v.get(p+'unison_detune'),0,10,p+'unison_detune'),detune_range))
        if not v.integer(p+'midi_track',0,1):
            # Without key tracking Vital plays the pitch of C3 plus the transposition on every key;
            # Serum's switch leaves the oscillator on note 0 (8.2 Hz). Put C3 in the octave control
            # and transpose in coarse pitch: a fixed +48 there would leave only +16 semitones for
            # modulation before Serum's +64 ceiling. This fits Vital's whole -48..48 transpose.
            params.update(kParamPitchTrack=0.0,kParamOctave=float(FIXED_PITCH_NOTE/12),
                          kParamPitch=0.0,kParamCoarsePit=transpose)
        if blend is not None: params['kParamDetuneWid']=blend
        if width is not None: params['kParamUnisonStereo']=width
        native_range=unison.serum_range(detune_range)
        if native_range!=2: params['kParamUnisonRange']=float(native_range)
        if abs(native_range-detune_range)>1e-6:
            r.note('approximation',p+'detune_range',
                   'An integer semitone range and compensated detune preserve the outer voice pitches.',
                   original=detune_range,converted=native_range)
        mode=unison.detune_mode(voices,c.clip(v.get(p+'detune_power'),-5,5,p+'detune_power'))
        if mode: params['kParamDetuneMode']=mode
        stack=v.integer(p+'stack_style',0,10)
        if stack not in stacks:
            r.note('unsupported',p+'stack_style','Chord/harmonic unison stack reduced to ordinary unison.',style=stack)
        elif stacks[stack]: params['kParamUnisonStack']=stacks[stack]
        m['plainParams']=params
        nframes=a.frames//2048
        position=(trajectory.position_law(nframes)[0] if joint is not None else
                  sweep_position(amount if sweep=='warp' else morph_amount) if sweep else serum_table_position(wave_frame,nframes))
        wp=dict(kParamTablePos=position,
                kParamInitialPhase=((c.unit(v.get(p+'phase'),p+'phase')+.5)%1)*360,
                kParamRandomPhase=100*c.unit(v.get(p+'random_phase'),p+'random_phase'))
        if joint is not None:
            r.note('approximation',p+'trajectory',
                   'Oscillator controls share one source and are sampled together into a wavetable trajectory; frame quantization and engine smoothing may differ.',
                   source=joint.source,slots=[m.slot for m in joint.routes],controls=sorted({m.target for m in joint.routes}),frames=nframes,table_gain=a.gain,user_impact='baked')
            v.used.add(p+'distortion_spread')
        elif warp is not None:
            r.note('approximation',p+'warp',f"Vital's {PHASE_WARPS[wtype]} warp is rendered into the wavetable"
                   +(', laid out along the table position so its modulation still moves it.' if sweep=='warp' else ' at its stored amount.'),amount=amount,
                   user_impact='baked',controls=[p+'distortion_amount'])
            if p+'distortion_phase' in moving:
                r.note('unsupported',p+'distortion_phase','The warp phase is modulated in Vital; the rendered table cannot follow it.')
            if voices>1 and abs(v.get(p+'distortion_spread'))>1e-6:
                r.note('unsupported',p+'distortion_spread','Per-voice spread of the warp is not converted.')
            else: v.used.add(p+'distortion_spread')
        elif wtype in fm.MENUS:
            cross.append((i,wtype,amount))              # needs the other oscillators: after the loop
        elif wtype==4:
            wp['kParamWarpMenu']='kASYMPos' if amount>=.5 else 'kASYMNeg'
            wp['kParamWarp']=abs(2*amount-1)
            r.note('approximation',p+'warp','Vital Bend mapped to a one-sided Serum Asym warp.',user_impact='alternative')
        elif wtype in (9,12):
            wp.update(kParamWarpMenu=warps[wtype],kParamWarp=fm.pd_warp(amount) if wtype==9 else fm.rm_warp(amount))
            r.note('approximation',p+'warp','Modulation by the sample uses the law measured between oscillators; its depth from the Noise oscillator is unverified.',user_impact='alternative')
        elif wtype in warps:
            wp.update(kParamWarpMenu=warps[wtype],kParamWarp=amount)
            r.note('approximation',p+'warp',"Serum's own warp of the same family is used; its response to the amount is unverified.",user_impact='alternative')
        elif wtype:
            r.note('unsupported',p+'warp','Vital Formant warp is not converted; warp disabled.')
        else:
            v.used.update(p+x for x in ('distortion_phase','distortion_spread'))
        if native_warp:
            r.note('unsupported',p+'warp',"Table position and warp amount both move in Vital; Serum's own warp is used and its response differs.")
        if joint is not None:
            pass
        elif sweep=='warp':
            c.target(p+'distortion_amount','WTOsc',slot,'kParamTablePos')
        elif warp is None and wtype in warps:
            c.target(p+'distortion_amount','WTOsc',slot,'kParamWarp')
        gains[slot]=a.gain;tables_params[slot]=wp
        wt={k:x for k,x in m.get(f'WTOsc{slot}',{}).items()
            if k not in ('relativePathToWT','embeddedWTData','tableDisplayName','interpolateAfterLoad')}
        wt.update(flex={},plainParams=wp,numChannels=1,numFrames=a.frames,sampleRate=a.sample_rate)
        if c.embed:
            # Same layout as presets Serum 2 saves with embedded tables (tag 'Embedded-Data').
            wt.update(embeddedWTData=codec.Float32List(a.pcm[0]),interpolateAfterLoad=0,
                      tableDisplayName=a.display_name[:64] or 'Vital table')
        else:
            wt['relativePathToWT']=a.serum_reference
        m[f'WTOsc{slot}']=wt
        c.routing(slot,v.integer(p+'destination',0,14),p+'destination')
        # The coarse pitch control spans 128 semitones without steps (measured), enough for Vital's
        # 96; the semitone control itself stops at +-12.
        for suffix,param,scale in [('pan','kParamPan',1),('tune','kParamFine',1),('transpose','kParamCoarsePit',96/128)]:
            c.target(p+suffix,'Oscillator',slot,param,scale)
        # Serum's detune control moves on the square root of its stored value (measured).
        c.target(p+'unison_detune','Oscillator',slot,'kParamDetune',
                 law=lambda d,span=detune_range:unison.detune_position(d,span))
        if sweep=='morph': c.target(p+'spectral_morph_amount','WTOsc',slot,'kParamTablePos')
        elif not sweep: c.target(p+'wave_frame','WTOsc',slot,'kParamTablePos')
        c.target(p+'phase','WTOsc',slot,'kParamInitialPhase')
        c.target(p+'random_phase','WTOsc',slot,'kParamRandomPhase')
        if morph_type==10:
            if new_static:
                correction=-12*(1-morph_amount)
                params['kParamCoarsePit']=float(params.get('kParamCoarsePit',0.))+correction
                r.note('approximation',p+'shepard_pitch',
                       'The fixed Shepard spectrum and pitch ramp are rendered together; runtime wrap phase history is not represented.',
                       semitones=correction,dynamic=False)
            else: _shepard_pitch(c,p,slot,params,morph_amount)
        v.used.update(p+x for x in ('spectral_morph_amount','spectral_morph_phase','spectral_morph_spread'))
        if morph is not None:
            r.note('approximation',p+'spectral_morph',f'Spectral morph ({(BAKED_MORPHS | STATIC_MORPHS)[morph_type]}) is rendered into the wavetable'
                   +(', evaluated along the shared-source trajectory.' if joint is not None and morph_moves else
                   ', laid out along the table position so its modulation still moves it.' if sweep=='morph' else ' at its stored amount.'),amount=morph_amount,
                   user_impact='baked',controls=[p+'spectral_morph_amount'])
            if morph_moves and sweep not in ('morph','trajectory'):
                r.note('unsupported',p+'spectral_morph_amount','The morph amount is modulated in Vital; the rendered table cannot follow it.')
            if abs(v.get(p+'spectral_morph_phase')-.5)>1e-6:
                r.note('unsupported',p+'spectral_morph_phase','The spectral morph phase differs from its neutral value and is not represented in the rendered table.')
            if abs(v.get(p+'spectral_morph_spread'))>1e-6:
                r.note('unsupported',p+'spectral_morph_spread','Per-voice spread of the spectral morph is not converted.')
        elif morph_type==MORPH_PHASE_DISPERSE:
            r.note('approximation',p+'spectral_morph',
                   'Phase disperse is not applied under these controls; the wavetable retains its original spectral phases.',
                   user_impact='unsupported')
        elif morph_type:
            r.note('unsupported',p+'spectral_morph','This spectral morph type is not converted; wavetable remains unprocessed.')
        if voices>2 and mode!=unison.detune_mode(voices,0.0):
            r.note('approximation',p+'detune_power',"Voice spacing uses Serum's nearest detune mode; inner voices can sit a few cents off.")
        elif voices>2 and abs(v.get(p+'detune_power'))>=0.01:
            r.note('approximation',p+'detune_power','Voice spacing is linear in Serum; inner voices can sit a few cents off.')
        r.mapped.append(f'oscillator_{i}')
    for i,kind,amount in cross:
        _cross_modulation(c,i,kind,amount,gains,tables_params)
    s.set_params('Oscillator4',{'kParamEnable':0.0})


FIXED_PITCH_NOTE = 48        # the note an oscillator without key tracking plays in Vital


def _cross_modulation(c: Context,i: int,kind: int,amount: float,gains: dict,tables_params: dict):
    """Vital's FM and RM from another oscillator on Serum's PD and RM warps (laws in fm.py)."""
    v,r=c.v,c.r
    slot=i-1;p=f'osc_{i}_';j=fm.modulator(i,kind);q=f'osc_{j}_';wp=tables_params[slot]
    level,gain,feature=c.levels[slot]
    ring=kind>=10
    if j-1 not in gains:
        if ring and ring_modulation.prepare(c,i,None):return
        # Vital reads silence from an oscillator that is off: no FM, and RM leaves (1 - amount).
        if ring: c.levels[slot]=(level,gain*fm.rm_gain(amount,None),feature)
        r.note('info',p+'warp','The modulating oscillator is off; the warp does nothing in Vital either.')
        return
    voices=v.integer(q+'unison_voices',1,16)
    ratio=gains[j-1]*fm.stack_gain(voices,c.unit(v.get(q+'unison_blend'),q+'unison_blend'),c.unit(v.get(q+'stereo_spread'),q+'stereo_spread'))
    if ring:
        wp.update(kParamWarpMenu=fm.MENUS[kind],kParamWarp=fm.rm_warp(amount,ratio))
        if ring_modulation.prepare(c,i,ratio):return
        c.levels[slot]=(level,gain*fm.rm_gain(amount,ratio),feature)
        c.target(p+'distortion_amount','WTOsc',slot,'kParamWarp',law=lambda d,k=ratio: fm.rm_warp(d,k))
    else:
        first,second=fm.pd_warps(amount,ratio)
        wp.update(kParamWarpMenu=fm.MENUS[kind],kParamWarp=first)
        if second: wp.update(kParamWarpMenu2=fm.MENUS[kind],kParamWarp2=second)
        wp['kParamInitialPhase']=(wp['kParamInitialPhase']-fm.pd_lead(amount,ratio))%360
        c.target(p+'distortion_amount','WTOsc',slot,'kParamWarp',law=lambda a,k=ratio: fm.pd_warp(a,k))
    r.note('approximation',p+'warp',"Vital's "+('RM' if ring else 'FM')+f' from oscillator {j} is set on Serum\'s '+('RM' if ring else 'phase distortion')
           +' warp at the measured depth; with a moving amount the '+('level' if ring else 'phase')+' correction holds only at the stored value.',
           user_impact='control_partial' if any(m.target==p+'distortion_amount' and n not in v.folded
                                               for n,m in active_modulations(v).items()) else 'alternative')
    if voices>1:
        r.note('approximation',p+'warp','The modulating oscillator has unison: Vital modulates with the whole stack, Serum with one of its voices.',user_impact='alternative')


def _sample(c: Context):
    v,s,r=c.v,c.s,c.r
    enabled=v.integer('sample_on',0,1)
    s.set_params('Oscillator3',{'kParamEnable':float(enabled)})
    if not enabled:
        v.ignore_prefix('sample_');v.used.add('sample');return
    a=sample(v.settings.get('sample'))
    c.asset(a,'sample');v.used.add('sample')
    # Pitch, start position, loop and pan of Serum's Noise oscillator, as measured in sampler.py.
    keytrack=bool(v.integer('sample_keytrack',0,1))
    ns=sampler.noise_settings(keytrack,c.clip(v.get('sample_transpose'),-48,48,'sample_transpose'),
                              c.clip(v.get('sample_tune'),-1,1,'sample_tune'),bool(v.integer('sample_random_phase',0,1)),
                              bool(v.integer('sample_loop',0,1)),bool(v.integer('sample_bounce',0,1)) and c.embed,
                              a.sample_rate,a.frames,transpose_quantize=v.integer('sample_transpose_quantize',0,8191),
                              voice_transpose=v.integer('voice_transpose',-48,48),
                              voice_tune=c.clip(v.get('voice_tune'),-1,1,'voice_tune'))
    for kind,feature,message in ns['notes']: r.note(kind,feature,message)
    if v.integer('sample_bounce',0,1) and not c.embed:
        r.note('unsupported','sample_bounce','Bounce needs the clip embedded with its reverse; the external file loops forwards only.')
    pan,pan_gain=sampler.pan(c.clip(v.get('sample_pan'),-1,1,'sample_pan'))
    c.levels[3]=(c.unit(v.get('sample_level'),'sample_level'),pan_gain,'sample_level')
    m=s.set_params('Oscillator3',dict(kParamEnable=1.0,kParamPan=pan,**ns['oscillator']))
    noise={k:x for k,x in m.get('NoiseOsc3',{}).items() if k not in ('plainParams','embeddedNoiseData')}
    noise.update(numChannels=a.channels,sampleRate=a.sample_rate,relativePathToNoiseSample=a.serum_reference,
                 plainParams=ns['noise'],**ns['fields'])
    if c.embed:
        noise['embeddedNoiseData']=[codec.Float32List(ch) for ch in (sampler.mirror(a.pcm) if ns['mirror'] else a.pcm)]
    m['NoiseOsc3']=noise
    c.routing(3,v.integer('sample_destination',0,14),'sample_destination')
    c.target('sample_pan','Oscillator',3,'kParamPan')
    # Vital's range for the transposition is +-48 semitones; the colour control has its own scale.
    c.target('sample_transpose','NoiseOsc',3,'kParamColor',law=ns['transpose_law'])
    if keytrack: c.target('sample_tune','NoiseOsc',3,'kParamFine')


MAX_FX_GAIN = 15.8        # two flat EQ modules, 12 dB each


def _levels(c: Context):
    """Write the source volumes, moving gain Serum's oscillators cannot give into the master."""
    v,s,r=c.v,c.s,c.r
    if not c.levels: return
    # A level that modulation can push up needs its headroom too.
    mods=active_modulations(v);into=amount_modulators(mods)
    def reach(feature: str,level: float) -> float:
        for i,m in mods.items():
            if m.target!=feature: continue
            swing=abs(m.amount)+sum(2*abs(mods[j].amount) for j in into.get(i,()))
            level+=min(1.0,swing)*(0.5 if m.bipolar else 1.0 if m.amount>=0 or into.get(i) else 0.0)
        return min(1.0,level)
    peak=max(reach(feature,level)**2*gain for level,gain,feature in c.levels.values())
    lowered=1.0
    if peak>1.0+1e-9:
        direct=any(v.get(('sample_' if slot==3 else f'osc_{slot+1}_')+'destination')==4 for slot in c.levels)
        if not (v.get('distortion_on') or v.get('compressor_on')):
            lowered=c.master_boost=peak
            r.note('approximation','levels','Sources needing more than 0 dB were lowered together and the master raised by the same amount.',
                   gain_db=20*math.log10(peak))
        elif not direct and peak<=MAX_FX_GAIN:
            # A lower level into distortion or compression would change their sound, so the gain
            # is restored by a flat EQ module at the head of the effect rack instead.
            lowered=c.fx_gain=peak
            r.note('approximation','levels','Sources needing more than 0 dB were lowered together; a gain stage ahead of the effects restores the level.',
                   gain_db=20*math.log10(peak))
        else:
            for level,gain,feature in c.levels.values():
                if level**2*gain>1.0+1e-9:
                    r.note('clipped',feature,'Oscillator gain above 0 dB is limited to 0 dB (a level-dependent effect follows).',
                           original=level**2*gain,converted=1.0)
    for slot,(level,gain,feature) in c.levels.items():
        gain/=lowered
        s.state[f'Oscillator{slot}']['plainParams']['kParamVolume']=float(min(1.0,level**2*gain))
        # Both level controls move on the square root of the amplitude (measured for Serum).
        c.target(feature,'Oscillator',slot,'kParamVolume',math.sqrt(gain))


def _nearest_power(shape) -> float:
    """Exponent p whose powerScale(t, p) is closest (least squares) to shape(t) on 0..1."""
    ts=[(i+0.5)/48 for i in range(48)];ys=[shape(t) for t in ts]
    def error(p): return sum((power_scale(t,p)-y)**2 for t,y in zip(ts,ys))
    best=min((0.5*i for i in range(-60,61)),key=error)
    return min((best+0.01*i for i in range(-50,51)),key=error)


def _envelopes(c: Context):
    """Envelope times, levels and stage curves.

    Measured on Vital 1.0.7 and Serum 2.1.5: both bend a stage with the same exponential.
    Vital's attack rises by powerScale(t, -attack_power) and its decay/release fall by
    powerScale(t, power); Serum's stage follows powerScale(t, p) for a curve value of
    100 * curve_value(-p). The amplitude envelope differs: Vital squares the whole envelope,
    Serum squares only the sustain level and moves the amplitude itself along the stage, so the
    curve closest to Vital's squared stage is used there. As a modulation source Serum's first
    envelope is that amplitude, which modulation.SOURCE_LAWS bends back.
    """
    v,s,r=c.v,c.s,c.r
    for i in range(1,5):
        p=f'env_{i}_';params={}
        for part in ('attack','hold','decay','release'):
            n=c.clip(v.get(p+part),0,1.4142135624 if part=='hold' else 2.37842,p+part)
            params['kParam'+part.capitalize()]=min(32,n**4)
            # Vital's time is the 4th power of its control, Serum's 32 s times the 5th power of
            # its control position (measured for all four stages).
            c.target(p+part,'Env',i-1,'kParam'+part.capitalize(),law=lambda raw: min(1.0,max(0.0,raw)**0.8/2))
        sustain=c.unit(v.get(p+'sustain'),p+'sustain')
        # Serum holds the square of its sustain control, on every envelope (measured). That is
        # Vital's squared amplitude envelope as it stands; the others need the square root.
        params['kParamSustain']=sustain if i==1 else math.sqrt(sustain)
        if i==1: c.target(p+'sustain','Env',i-1,'kParamSustain')
        else: c.target(p+'sustain','Env',i-1,'kParamSustain',law=lambda raw: math.sqrt(min(1.0,max(0.0,raw))))
        attack,decay,release=(c.clip(v.get(p+part+'_power'),-20,20,p+part+'_power') for part in ('attack','decay','release'))
        powers=[-attack,decay,release]
        if i==1:
            powers=[_nearest_power(lambda t: power_scale(t,-attack)**2),
                    decay if sustain>1-1e-6 else _nearest_power(lambda t: (lambda u: u*(2-(1-sustain)*u)/(1+sustain))(power_scale(t,decay))),
                    _nearest_power(lambda t: (lambda u: u*(2-u))(power_scale(t,release)))]
        for j,power in enumerate(powers,1):
            params[f'kParamCurve{j}']=100*curve_value(-power)
        if v.get(p+'delay'):
            r.note('unsupported',p+'delay','Envelope delay is not mapped; delay is omitted.')
        s.set_params(f'Env{i-1}',params)
    for i in (5,6): v.ignore_prefix(f'env_{i}_')
    r.mapped.append('envelopes:4')
    r.note('approximation','envelopes','Stage curves of the amplitude envelope are the nearest Serum curve to Vital\'s squared envelope; the other envelopes are exact.')


SERUM_LFO_RATE_STEPS = 14         # synced rate = 100 * (k / 14)^4, k = 5 is one bar, 7 a quarter note
VITAL_TEMPO_BIAS = 0.3
SERUM_TEMPO_BIAS = 1.0/6.0
SERUM_LFO_MAX_DELAY = 4.0
# Both synths smooth an LFO with a one-pole low-pass (measured). Vital's time constant is
# 2^smooth_time / ln 32 seconds; Serum's is 2.5 ms / (1 - smooth / 100)^2, or none at smooth 0.
SERUM_MIN_SMOOTH_S = 0.0025


def synced_delay(seconds: float, beats_per_second: float) -> float:
    """Serum's delay value for a tempo-synced LFO, as close to `seconds` as its note lengths allow.

    A synced LFO snaps its delay to note lengths: the stored value v picks step
    floor(3.5 v + 0.2), and step k lasts 2^(k - 9) bars (measured: 1.0 = a 1/64 note,
    2.0 = a quarter note, 2.75 = one bar). The value returned sits in the middle of its step.
    """
    if seconds<=0: return 0.0
    bar=4.0/beats_per_second
    step=max(1,min(14,round(9+math.log2(seconds/bar))))
    return min(SERUM_LFO_MAX_DELAY,(step+0.3)/3.5)


def _lfo_phase_sources(mods) -> set[str]:
    """Sources reaching an LFO phase, through parameter controls or amount/Aux chains."""
    reverse={}
    for i,m in mods.items():
        slot=f'slot:{i}'
        reverse.setdefault(slot,set()).add(m.source)
        amount=re.fullmatch(r'modulation_(\d+)_amount',m.target)
        lfo=re.match(r'(lfo_\d+)_',m.target)
        target=f'slot:{amount.group(1)}' if amount else lfo.group(1) if lfo else m.target
        reverse.setdefault(target,set()).add(slot)
        if lfo and m.target.endswith('_phase'):
            reverse.setdefault('phase',set()).add(slot)
    seen={'phase'};pending=['phase']
    while pending:
        for source in reverse.get(pending.pop(),()):
            if source not in seen:
                seen.add(source);pending.append(source)
    return seen


def _lfo_tempo_routes(c: Context,mods) -> None:
    """Use the native synchronized ladder for one bounded linear division control."""
    into=amount_modulators(mods)
    active=[(i,m) for i,m in mods.items() if i not in c.v.folded]
    for i in range(1,9):
        p=f'lfo_{i}_'
        routes=[(j,m) for j,m in active if m.target==p+'tempo']
        if len(routes)!=1 or c.v.integer(p+'sync',0,4)!=1:
            continue
        j,m=routes[0]
        if (into.get(j) or m.source not in SOURCES or m.stereo or m.bipolar
                or abs(m.power)>=.01 or m.points or c.source_gain.get(m.source,1.0)!=1.0
                or any(a.target in (p+'sync',p+'sync_type') for _,a in active)):
            continue
        # Keep the fractional offset from fixed controls. Rounding it before adding the
        # moving source would shift every division boundary.
        base=c.v.get(p+'tempo')
        end=base+12.0*m.amount
        if min(base,end)<1.0 or max(base,end)>12.0:
            continue
        step=base-1.0+VITAL_TEMPO_BIAS-SERUM_TEMPO_BIAS
        params=c.s.state[f'LFO{i-1}']['plainParams']
        params.update(kParamRate=100.0*(step/SERUM_LFO_RATE_STEPS)**4,kParamAnchored=0.0)
        c.target(p+'tempo','LFO',i-1,'kParamRate',12.0/SERUM_LFO_RATE_STEPS)
        c.r.note('approximation',p+'tempo',
                 'A bounded linear synchronized division control uses the native rate ladder with continuous phase. Smoothing differences remain as the rate changes.',
                 raw_base=base,raw_end=end,vital_boundary_bias=VITAL_TEMPO_BIAS,serum_boundary_bias=SERUM_TEMPO_BIAS)


def _lfos(c: Context):
    v,s,r=c.v,c.s,c.r
    shapes=v.array('lfos')
    mods=active_modulations(v)
    moving={m.target for j,m in mods.items() if j not in v.folded}
    phase_sources=_lfo_phase_sources(mods)
    for i in range(1,9):
        p=f'lfo_{i}_';sync=v.integer(p+'sync',0,4);mode=v.integer(p+'sync_type',0,5)
        terminal_hold=(sync==0 and mode==2 and abs(v.get(p+'phase'))<1e-9
                       and f'lfo_{i}' not in phase_sources
                       and not any(p+x in moving for x in ('phase','frequency','tempo','sync','sync_type',
                                                         'smooth_time','smooth_mode','delay_time')))
        # The ordinary curve LFO is Serum's default type; 'Path' is a different LFO and must not be set.
        params=dict(kParamPhase=360*c.unit(v.get(p+'phase'),p+'phase'))
        delay=c.clip(v.get(p+'delay_time'),0,4,p+'delay_time')
        smooth=2**c.clip(v.get(p+'smooth_time'),-10,4,p+'smooth_time')/math.log(32) if v.integer(p+'smooth_mode',0,1) else 0.0
        serum_smooth=0.0
        if smooth>=0.001:
            # Serum cannot smooth less than 2.5 ms; that is still closer to Vital's default 1.6 ms
            # than no smoothing, which would turn every step of a shape into a click.
            serum_smooth=max(SERUM_MIN_SMOOTH_S,smooth)
            params['kParamSmooth']=max(0.5,100*(1-math.sqrt(SERUM_MIN_SMOOTH_S/serum_smooth)))
        if v.get(p+'fade_time')>0:
            # Vital 1.0.7 ignores this setting in every LFO mode (measured), so no rise is written.
            r.note('info',p+'fade_time','Vital does not apply the LFO fade-in (1.0.7, 1.6.4); none is written.')
        if sync in (0,4):
            frequency=c.clip(v.get(p+'frequency'),-7,9,p+'frequency')
            hz=2**frequency
            # Free rate is stored in Hz up to 100; the 10x switch covers the rest (measured).
            params['kParamBeatSync']=0.0
            if hz>100.0:
                params['kParamRate10x']=1.0; params['kParamRate']=min(100.0,hz/10.0)
            else:
                params['kParamRate']=hz
            params['kParamDelay']=delay                 # seconds while the LFO runs free (measured)
            # At a fixed rate the two smoothings pass different amounts of the LFO; the depth of
            # its routings makes up the difference at the LFO's own frequency.
            passed=lambda tau: 1/math.sqrt(1+(2*math.pi*hz*tau)**2)
            if not terminal_hold and abs(passed(smooth)/passed(serum_smooth)-1)>0.005:
                c.source_gain[f'lfo_{i}']=passed(smooth)/passed(serum_smooth)
                r.note('approximation',p+'smooth_time','LFO smoothing differs between the synths; the modulation depth is corrected at the LFO rate.',
                       factor=c.source_gain[f'lfo_{i}'])
            if sync==4: r.note('unsupported',p+'sync','LFO keytracking reduced to a fixed Hz rate.')
        else:
            if p+'tempo' in v.offsets:
                # Modulated divisions use floor(index + 0.3); a positive offset of 0.7
                # selects the next faster division.
                stored=round(v.settings.get(p+'tempo',DEFAULTS.get(p+'tempo',7)))
                tempo=max(0,min(12,math.floor(stored+v.offsets[p+'tempo']+VITAL_TEMPO_BIAS)));v.used.add(p+'tempo')
            else:
                tempo=v.integer(p+'tempo',0,12)
            if tempo==0:
                params.update(kParamBeatSync=0.0,kParamRate=0.0,kParamDelay=delay)
                r.note('unsupported',p+'sync','LFO Freeze approximated by a zero-rate LFO; exact behavior unverified.')
            else:
                # Serum's ladder step k is Vital's tempo index minus one (1/4 note: 8 -> 7). With
                # the TRIP or DOT switch on, the ladder gains steps a third of the way between:
                # the triplet of step k sits at k + 2/3 and the dotted value at k - 2/3 (measured).
                step=float(tempo-1)
                params['kParamBeatSync']=1.0
                if sync==3:
                    step+=2.0/3.0; params['kParamTriplets']=1.0
                elif sync==2:
                    step-=2.0/3.0; params['kParamDotted']=1.0
                if step<0:
                    step=0.0; r.note('clipped',p+'tempo','Slower than Serum\'s slowest synced LFO rate.')
                params['kParamRate']=100*(step/SERUM_LFO_RATE_STEPS)**4
                params['kParamDelay']=synced_delay(delay,c.bps)
                if delay>0:
                    r.note('approximation',p+'delay_time','A synced Serum LFO delays by note lengths; the nearest one at the preset tempo is used.')
        # Serum's default mode retriggers per note (Vital "Trigger"); "Free" runs on like Vital "Sync".
        if mode==1: params['kParamMode']='Free'
        elif mode==2: params['kParamMode']='Envelope'
        elif mode>2: r.note('unsupported',p+'trigger','Advanced LFO trigger mode reduced to default retrigger.')
        # Stored inactive rate controls do not affect playback in this mode.
        v.used.update((p+'frequency',p+'tempo',p+'fade_time'))
        c.target(p+'phase','LFO',i-1,'kParamPhase')
        m=s.set_params(f'LFO{i-1}',{k:x for k,x in params.items() if not (k in ('kParamPhase','kParamDelay') and x==0)})
        shape=shapes[i-1] if i<=len(shapes) and shapes[i-1] else {
            'num_points':3,'points':[0,1,.5,0,1,1],'powers':[0,0,0],'smooth':False}
        points=shape.get('points',[]);n=shape.get('num_points',len(points)//2)
        powers=shape.get('powers',[])
        if not isinstance(n,int) or not 2<=n<=100 or not isinstance(points,list) or len(points)!=2*n:
            raise ConversionError(f'{p}: LFO must have 2..100 complete points')
        if not isinstance(powers,list) or len(powers)>n:
            raise ConversionError(f'{p}: malformed LFO powers')
        if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in points+powers):
            raise ConversionError(f'{p}: non-finite LFO coordinates/powers')
        xs=[c.unit(x,p+'x') for x in points[::2]]
        if any(a>b for a,b in zip(xs,xs[1:])) or xs[0]!=0 or xs[-1]!=1:
            raise ConversionError(f'{p}: LFO x coordinates must increase from 0 to 1')
        ys=[c.unit(y,p+'y') for y in points[1::2]]
        # Same axes and the same exponential bend in both synths; only the bend's number differs.
        points=list(zip(xs,ys));powers=[float(x) for x in (powers+[0]*n)[:n]]
        if terminal_hold:
            # The native Envelope freezes its smoothing state before the curve endpoint.
            # A held tail lets that state settle; scaling rate and points together preserves time.
            tau=SERUM_MIN_SMOOTH_S/(1.0-params['kParamSmooth']/100.0)**2 if params.get('kParamSmooth') else 0.0
            curve,q,tail=envelope_tail_curve(points,powers,bool(shape.get('smooth')),hz,tau)
            m['curveData']=curve
            m['plainParams']['kParamRate']*=q
            r.note('approximation',p+'terminal_hold',
                   'An Envelope smoothing tail is added at the endpoint, preserving the original point times. Periodic depth compensation is not applied; small terminal residuals remain.',
                   original_hz=hz,converted_hz=hz*q,point_scale=q,tail_seconds=tail,smoothing_constants=12)
        else:
            m['curveData']=lfo_curve(points,powers,bool(shape.get('smooth')))
        m['pathData']={}
        # Free frequency changes require a different control scale.
        c.target(p+'frequency','LFO',i-1,'kParamRate',nonlinear=True)
    _lfo_tempo_routes(c,mods)
    r.mapped.append('lfos:8')


def _global(c: Context):
    v,s,r=c.v,c.s,c.r
    volume=c.clip(v.get('volume'),0,7399.4404,'volume')
    # Serum master has cubic amplitude: 60*log10(knob). The reference's
    # Serum2->Vital path subtracts 3 dB of fitted synth-level compensation.
    # Vital stores (dB + 80)^2. Serum 2's master shows 20*log10(value) + 3 dB (0.25 = its -9 dB
    # default), and with those gains both synths measured equally loud (Serum 2.1.5 / Vital 1.0.7).
    master=0.0 if volume==0 else 10**((math.sqrt(volume)-80)/20)/math.sqrt(2)
    poly=v.integer('polyphony',1,32)
    port=c.clip(v.get('portamento_time'),-10,4,'portamento_time')
    # Serum's glide time is in seconds, up to 8 (measured); Vital's is 2^value, up to 16.
    glide=0.0 if port<=-10 else c.clip(2**port,0,8,'portamento_seconds')
    params=dict(kParamMasterVolume=c.unit(master*c.master_boost,'master_volume'),kParamPolyCount=float(poly),
                kParamMonoToggle=1.0 if poly==1 else 0.0,kParamLegato=float(v.integer('legato',0,1)),
                kParamPortamentoTime=glide,
                # Vital's "always glide" and Serum's behave alike: detached notes glide only with it on (measured).
                kParamPortaAlways=float(v.integer('portamento_force',0,1)),
                kParamBendRangeUp=float(v.integer('pitch_bend_range',0,48)),
                # The saved wheel control is a percentage; MIDI CC units are separate.
                kParamModWheel=100.0*c.unit(v.get('mod_wheel'),'mod_wheel'))
    if params['kParamBendRangeUp']!=2:
        r.note('unsupported','pitch_bend_down','Only the verified bend-up parameter is written; downward bend remains at template defaults.')
    # Vital's velocity tracking scales a voice by ((1 - t) + t v)^2 for velocity v = 0..1 (measured:
    # t = 1 at velocity 64 is -11.9 dB, t = 0.5 is -4.95 dB). Serum's voice amplitude control is
    # twice its value and takes velocity through a routing, so it starts at 0 and velocity lifts it
    # along that curve.
    track=c.clip(v.get('velocity_track'),-1,1,'velocity_track')
    if abs(track)>1e-6:
        top=max((1-track)**2,1.0)
        params['kParamVoiceAmp']=0.0
        curve=sampled_curve(lambda x: ((1-track)+track*x)**2/top)
        count=math.ceil(top/2-1e-9)
        for _ in range(count):
            c.fixed_routings.append(('velocity',Destination('Global',0,'kParamVoiceAmp'),50*top/count,False,curve))
    else:
        # Modulation of Vital's master volume goes to the voice amplitude, which is linear and
        # free: its stored value means (dB + 80)^2, so the routing gets the gain as a curve.
        c.target('volume','Global',0,'kParamVoiceAmp',
                 law=lambda raw: 0.0 if raw<=0 else min(1.0,0.5*10**((math.sqrt(raw)-math.sqrt(volume))/20)) if volume>0 else 0.5)
    s.set_params('Global0',params)
    # Serum's glide control sits at (seconds / 8)^(1/5) of its travel (measured).
    c.target('portamento_time','Global',0,'kParamPortamentoTime',
             law=lambda x: 0.0 if x<=-10 else min(1.0,2**((x-3)/5)))
    r.note('approximation','master_volume','Master and oscillator gains use laws measured on Serum 2.1.5 against Vital 1.0.7 (equal loudness at equal dB); unison, filter and effect gains can still differ.')
    for i in range(1,5):
        key=f'macro_control_{i}'
        m=s.set_params(f'Macro{i-1}',{'kParamValue':100*c.unit(v.get(key),key)})
        m['name']=v.doc.get(f'macro{i}',f'MACRO {i}')
        c.target(key,'Macro',i-1,'kParamValue')
    r.mapped.extend(['global','macros:4'])


def _remaining(c: Context):
    v,r=c.v,c.r
    # Non-audible/editor data, and generators whose unsupported routes are
    # reported by the modulation stage, are not emitted as spurious warnings.
    for prefix in ('random_','modulation_'): v.ignore_prefix(prefix)
    # MPE reception and the oversampling factor are host-side choices, not part of the sound being converted.
    v.used.update(['custom_warps','random_values','view_spectrogram','mpe_enabled','oversampling'])
    for key in sorted(v.settings):
        if key in v.used or key.endswith(('_view_2d',)) or not v.changed(key): continue
        r.note('unsupported',key,'Non-default setting has no implemented mapping; target keeps its default.',value=v.settings[key])
    tuning=v.doc.get('tuning')
    if isinstance(tuning,dict) and tuning.get('default') is False:
        r.note('unsupported','tuning','A custom tuning is not converted; Serum plays equal temperament.')
    if not v.doc.get('synth_version'):
        r.note('info','synth_version','No version in input; interpreted using Vital 1.x defaults.')


# Destinations the modulation stage can route once their module is converted. A stored control
# (macro, mod wheel) moving anything else is folded into the value instead of being dropped.
_ROUTED = re.compile(r'osc_[123]_(level|pan|tune|transpose|wave_frame|phase|random_phase|unison_detune)'
                     r'|filter_(1|2|fx)_(cutoff|resonance|drive|mix|blend)|env_[1-4]_(attack|hold|decay|sustain|release)'
                     r'|lfo_[1-8]_phase|macro_control_[1-4]|portamento_time|sample_(level|pan|transpose)|modulation_\d+_amount'
                     r'|distortion_filter_cutoff|phaser_(center|feedback|dry_wet)')
_VOICE_AMP = 'volume'             # routed unless velocity tracking already uses the voice amplitude
_NATIVE_WARPS = {1,3,5,6,7,8,9,10,11,12}


def _routed(v: VitalPreset, target: str) -> bool:
    hit=re.fullmatch(r'osc_([123])_(distortion|spectral_morph)_amount',target)
    if hit:
        # A warp or morph amount is routed through a sweep table, which needs the table position
        # for itself; with the position moving only Serum's own warps remain.
        p=f'osc_{hit.group(1)}_'
        moved={m.get('destination') for m in v.settings.get('modulations',[]) if isinstance(m,dict) and m.get('source')}
        warp=int(v.settings.get(p+'distortion_type',0))
        if hit.group(2)=='distortion':
            return warp in _NATIVE_WARPS or (warp in PHASE_WARPS and p+'wave_frame' not in moved)
        return (int(v.settings.get(p+'spectral_morph_type',0)) in BAKED_MORPHS and p+'wave_frame' not in moved
                and not (warp in PHASE_WARPS and p+'distortion_amount' in moved))
    if target=='sample_tune':
        return bool(v.settings.get('sample_keytrack',DEFAULTS.get('sample_keytrack',0)))
    if target==_VOICE_AMP:
        return abs(v.settings.get('velocity_track',0))<=1e-6
    if target=='distortion_filter_cutoff':
        return bool(v.settings.get('distortion_on',0)) and int(v.settings.get('distortion_filter_order',0)) in (1,2)
    if not _ROUTED.fullmatch(target):
        return False
    if target.startswith('phaser_'):
        return bool(v.settings.get('phaser_on',0))
    if target.startswith('filter_'):
        # A filter that is converted has modulation targets, as the voice filter or in the FX rack.
        prefix=target[:target.index('_',7)+1]
        get=lambda key: v.settings.get(prefix+key,DEFAULTS.get(prefix+key,0))
        model,style=int(get('model')),int(get('style'))
        # Resonator makeup and shape are fitted at the stored controls, so fixed macros
        # must be folded before fitting. Moving sources still reach the registered targets.
        if model in (6,7): return False
        return bool(get('on')) and model in FILTER_KINDS and style<=4
    return True


def fold_static_modulations(v: VitalPreset, r: Report, preserve: set[int] = frozenset()) -> None:
    """Bake macro/mod-wheel modulation of parameters Serum cannot receive into the stored value.

    Vital presets often park a macro at a fixed position and use it to set a unison count, an
    effect mix or an LFO rate. Those routings have no Serum counterpart, so the offset is applied
    instead of being dropped: amount * source * range, a bipolar routing reading the source from
    its centre (measured in Vital 1.0.7). A control that something else modulates is left alone,
    because its stored position says little about where it sits while playing.
    """
    mods=active_modulations(v);into=amount_modulators(mods)
    moved={m.target for m in mods.values()}
    for i,m in mods.items():
        if i in preserve: continue
        if m.source not in STATIC_SOURCES or m.source in moved or m.stereo: continue
        if m.target not in RANGES or _routed(v,m.target): continue
        amount=static_amount(v,mods,into,i,moved)
        if amount is None or abs(amount[0])<1e-12: continue
        lo,hi=RANGES[m.target]
        v.offsets[m.target]=v.offsets.get(m.target,0.0)+amount[0]*m.value(v.get(m.source))*(hi-lo)
        for j in amount[1]:
            v.folded.add(j);v.ignore_prefix(f'modulation_{j}_')
        r.note('approximation',f'modulation_{i}_','A macro/mod wheel position that Serum cannot receive was folded into the stored value.',
               source=m.source,destination=m.target,amount=amount[0],user_impact='control_fixed')


def _static_mix_routes(v: VitalPreset, ids: dict, disabled: frozenset, report=None) -> dict[int,str]:
    """Keep eligible stored Mix controls until their complete response can be reserved."""
    mods=active_modulations(v);into=amount_modulators(mods);moved={m.target for m in mods.values()}
    result={}
    for target,effect,module in (('distortion_mix','distortion','FXDistortion'),
                                 ('delay_dry_wet','delay','FXDelay')):
        if target in disabled or not v.get(effect+'_on'): continue
        routes=[(i,m) for i,m in mods.items() if m.target==target and (abs(m.amount)>1e-12 or into.get(i))]
        if len(routes)!=1: continue
        i,m=routes[0]
        if not mix_route_supported(v,mods,into,i,report): continue
        if m.source not in STATIC_SOURCES or m.source in moved or m.stereo: continue
        if any((module,parameter) not in ids for parameter in ('kParamWet','kParamLevelOut')): continue
        amount=static_amount(v,mods,into,i,moved,ignore_zero_leaves=True)
        if amount is None or abs(amount[0])<=1e-12: continue
        prefixes=('delay_','delay_aux_') if effect=='delay' else ('distortion_',)
        if any(n.source==m.source and n.target!=target and n.target.startswith(prefixes)
               and (abs(n.amount)>1e-12 or into.get(j)) for j,n in mods.items()): continue
        if any(j!=i and mods[j].source==m.source for j in amount[1]): continue
        result[i]=target
    return result


def convert(vital: VitalPreset, template: SerumPreset, *, allow_placeholder: bool=False,
            destination_ids: dict[tuple[str,str],int] | None=None, embed: bool=True,
            _disabled_trajectories: frozenset = frozenset(),
            _disabled_atomic_mix_targets: frozenset = frozenset(),
            _disabled_post_master: bool=False,
            _disabled_ring_controls: frozenset = frozenset(),
            _disabled_phase_controls: frozenset = frozenset()) -> Result:
    original=copy.deepcopy(vital)
    original.used=set(); original.offsets={}; original.folded=set()
    # Do not let reuse of the same parsed input hide unsupported settings.
    vital.used=set()
    target=clean_template(template)
    report=Report(vital.source,vital.name)
    vital.offsets={}; vital.folded=set()
    c=Context(vital,target,report,embed)
    c.disabled_phase_controls=_disabled_phase_controls
    c.ring_controls=[];c.failed_ring_controls=set();c.disabled_ring_controls=_disabled_ring_controls
    c.table_trajectories=[];c.failed_trajectories=set()
    c.disabled_trajectories=(_disabled_trajectories if not allow_placeholder else
                            _disabled_trajectories|frozenset(f'osc_{i}_' for i in range(1,4)))
    ids=destination_ids if destination_ids is not None else KNOWN_IDS
    c.preserved_static_mix_routes=_static_mix_routes(vital,ids,_disabled_atomic_mix_targets,report)
    preserve=set(c.preserved_static_mix_routes)
    preserve.update(post_master.preserve(vital,ids,_disabled_post_master))
    if not allow_placeholder:
        for i in range(1,4):
            p=f'osc_{i}_'
            if p in _disabled_trajectories or not vital.get(p+'on'): continue
            plan=trajectory.correlated_controls(vital,p)
            if plan is not None: preserve.update(m.slot for m in plan.routes)
    fold_static_modulations(vital,report,preserve)
    for store in (target.meta,target.state):
        store.update(presetName=vital.name,presetAuthor=vital.doc.get('author',''),
                     presetDescription=vital.doc.get('comments',''),
                     tags=[vital.doc['preset_style']] if vital.doc.get('preset_style') else [])
    _filters(c)
    _oscillators(c,allow_placeholder)
    _sample(c)
    _levels(c)
    _envelopes(c)
    _lfos(c)
    _global(c)
    convert_effects(c)
    post_master.register(c,ids,_disabled_post_master)
    ring_modulation.register(c)
    convert_modulations(c,ids)
    mapped={record['vital_slot'] for record in report.modulations if record.get('status')=='mapped'}
    failed_mix={target for slot,target in c.preserved_static_mix_routes.items() if slot not in mapped}
    if c.post_master_failed:
        result=convert(vital,template,allow_placeholder=allow_placeholder,destination_ids=destination_ids,embed=embed,
                       _disabled_trajectories=_disabled_trajectories,_disabled_atomic_mix_targets=_disabled_atomic_mix_targets,
                       _disabled_post_master=True,_disabled_ring_controls=_disabled_ring_controls,
                       _disabled_phase_controls=_disabled_phase_controls)
        result.report.note('approximation','master_volume',c.post_master_failure+' The original conversion is retained.',user_impact='control_partial')
        return result
    if c.failed_trajectories or failed_mix or c.failed_ring_controls:
        result=convert(vital,template,allow_placeholder=allow_placeholder,destination_ids=destination_ids,embed=embed,
                       _disabled_trajectories=_disabled_trajectories|frozenset(c.failed_trajectories),
                       _disabled_atomic_mix_targets=_disabled_atomic_mix_targets|frozenset(failed_mix),
                       _disabled_post_master=_disabled_post_master,
                       _disabled_ring_controls=_disabled_ring_controls|frozenset(c.failed_ring_controls),
                       _disabled_phase_controls=_disabled_phase_controls)
        for p in sorted(c.failed_trajectories):
            result.report.note('approximation',p+'trajectory',
                               'The joint table trajectory could not be assigned with its complete source response and slot requirements; the original conversion is retained.',user_impact='control_partial')
        for target in sorted(failed_mix):
            result.report.note('approximation',target,
                               'The complete mix response could not be assigned; the stored control is folded into the original conversion.',user_impact='control_fixed')
        for p in sorted(c.failed_ring_controls):
            result.report.note('approximation',p+'ring_gain',
                               'The complete ring-modulation response could not be assigned; the original conversion is retained.',user_impact='control_partial')
        return result
    _remaining(c)
    if filter_structure.apply(c,ids): target=c.s
    failed_phase=phase_modulation.apply(c,ids)
    if failed_phase:
        result=convert(vital,template,allow_placeholder=allow_placeholder,destination_ids=destination_ids,embed=embed,
                       _disabled_trajectories=_disabled_trajectories,_disabled_atomic_mix_targets=_disabled_atomic_mix_targets,
                       _disabled_post_master=_disabled_post_master,_disabled_ring_controls=_disabled_ring_controls,
                       _disabled_phase_controls=_disabled_phase_controls|frozenset(failed_phase))
        for p in sorted(failed_phase):
            result.report.note('approximation',p+'phase_response',
                               'The complete phase response could not be assigned; the original conversion is retained.',user_impact='control_partial')
        return result
    voice_gain.apply(c,original,ids)
    finalize_modulation_reporting(report)
    target=c.s
    report.note('info','template','Settings this conversion does not write keep the values of the template preset.',
                product_version=template.meta.get('productVersion'),schema_version=template.meta.get('version'))
    return Result(target,list(c.assets.values()),report)
