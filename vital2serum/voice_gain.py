"""Restore supported common source attenuation before voice processing.

The original source volumes retain their common attenuation. A constant voice
amplitude or a scaled, owned velocity curve restores that factor before the voice
filters, while its previous static output compensation is removed. Feature decisions
must finish before this operation; their original gain flags remain unchanged.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from .curves import remap_curve, sampled_curve
from .errors import ConversionError
from .modulation import KNOWN_IDS, MAX_SLOTS, SOURCES, active_modulations
from .post_master import gain_stages, master_value
from .serum import plain


@dataclass(frozen=True)
class Plan:
    factor: float
    branch: str
    master: float
    level_note: dict
    slot: int | None = None
    velocity_slot: str | None = None
    velocity_track: float = 0.0
    velocity_top: float = 1.0
    head_cleanup: HeadCleanup | None = None
    tail_cleanup: TailCleanup | None = None


def _velocity_slot(preset, ids, track):
    """Recognize only the complete single normalized velocity routing we own."""
    if not isinstance(ids,dict) or type(ids.get(('Global','kParamVoiceAmp'))) is not int:
        return None
    if ids[('Global','kParamVoiceAmp')]!=2 or SOURCES.get('velocity')!=16:
        return None
    slots=[(key,module) for key,module in preset.state.items()
           if key.startswith('ModSlot') and isinstance(module,dict)
           and module.get('destModuleTypeString')=='Global'
           and module.get('destModuleParamName')=='kParamVoiceAmp']
    if len(slots)!=1:return None
    key,module=slots[0]
    if key not in {f'ModSlot{i}' for i in range(MAX_SLOTS)}:return None
    if (module.get('source')!=[16,0] or any(type(x) is not int for x in module['source'])
            or type(module.get('destModuleID')) is not int or module['destModuleID']!=0
            or type(module.get('destModuleParamID')) is not int or module['destModuleParamID']!=2):
        return None
    top=max((1.0-track)**2,1.0)
    expected=dict(source=[16,0],destModuleTypeString='Global',destModuleID=0,
                  destModuleParamName='kParamVoiceAmp',destModuleParamID=2,
                  plainParams=dict(kParamAmount=50.0*top,kParamBipolar=0.0,kParamMainCurveData=1.0),
                  flex=[sampled_curve(lambda x:((1.0-track)+track*x)**2/top),{}])
    if module!=expected:return None
    index=int(key[7:])
    for other in preset.state.values():
        if not isinstance(other,dict):continue
        kind=other.get('destModuleTypeString')
        if isinstance(kind,str) and kind.startswith('ModSlot'):
            owner=other.get('destModuleID')
            if kind!='ModSlot' or type(owner) is not int or not 0<=owner<MAX_SLOTS:
                return None
            if owner==index:return None
    return key


def _constant_slot(preset, ids):
    """Reserve only a known destination and an entirely unowned native slot."""
    if not isinstance(ids,dict) or not isinstance(preset.state.get('Macro0'),dict):
        return None
    number=ids.get(('Global','kParamVoiceAmp'))
    if type(number) is not int or number!=2 or SOURCES.get('macro_control_1')!=25:
        return None
    incoming=set()
    for key,module in preset.state.items():
        if not key.startswith('ModSlot') or not isinstance(module,dict):continue
        kind=module.get('destModuleTypeString')
        if isinstance(kind,str) and kind.startswith('ModSlot'):
            index=module.get('destModuleID')
            if kind!='ModSlot' or type(index) is not int or not 0<=index<MAX_SLOTS:
                return None
            incoming.add(index)
    empty={'plainParams':'default','source':[0,0]}
    return next((i for i in range(MAX_SLOTS)
                 if i not in incoming and preset.state.get(f'ModSlot{i}')==empty),None)


# Rack stage identities, distinct from the FXRack container's own controls.
_FX_TYPES = {'FXDistortion':0, 'FXPhaser':2, 'FXChorus':3, 'FXDelay':4,
             'FXComp':5, 'FXReverb':6, 'FXEQ':7, 'FXFilter':8, 'FXUtils':12}
_NON_FX_RANGES = {'Global':range(1),'Oscillator':range(4),'WTOsc':range(3),
                  'NoiseOsc':range(3,4),'VoiceFilter':range(2),'Env':range(4),
                  'LFO':range(8),'Macro':range(8)}
_NON_FX = set(_NON_FX_RANGES)


def _non_fx_owner(state,kind,index):
    if kind not in _NON_FX or type(index) is not int or index not in _NON_FX_RANGES[kind]:return False
    if kind in ('WTOsc','NoiseOsc'):
        owner=state.get(f'Oscillator{index}',{})
        return isinstance(owner,dict) and isinstance(owner.get(f'{kind}{index}'),dict)
    return isinstance(state.get(f'{kind}{index}'),dict)


@dataclass(frozen=True)
class HeadCleanup:
    slots: tuple[str, ...]
    report_fields: tuple[tuple, ...]


@dataclass(frozen=True)
class TailCleanup:
    start: int
    gain: float
    note: int


def _tail_cleanup(preset, report, ids, desired, original_master):
    """Recognize a complete static suffix before shortening its output gain."""
    if not isinstance(ids,dict) or not isinstance(report.mapped,list):return None
    if report.mapped.count('master_output_gain')!=1:return None
    if not isinstance(report.notes,list) or not isinstance(report.modulations,list):return None
    owned=[(i,n) for i,n in enumerate(report.notes) if isinstance(n,dict)
           and n.get('feature')=='master_volume' and ('gain' in n or 'output_stage' in n)]
    if len(owned)!=1:return None
    note_index,note=owned[0]
    if (set(note)!={'kind','feature','message','gain','output_stage'}
            or note['kind']!='approximation' or not isinstance(note['message'],str)
            or type(note['gain']) not in (int,float) or not math.isfinite(note['gain'])
            or note['gain']!=desired or type(note['output_stage']) is not int):return None
    start=note['output_stage'];state=preset.state;module=state.get('FXRack0')
    if not isinstance(module,dict) or not isinstance(module.get('FX'),list):return None
    rack=module['FX']
    if not 0<=start<len(rack):return None
    expected=gain_stages(desired,10-start)
    if expected is None or len(expected)>2 or len(rack)!=start+len(expected):return None
    for stage,gain in zip(rack[start:],expected):
        wanted={'FXEQ':{'plainParams':{'kParamEnable':1.0,'kParamLevelOut':.5*math.sqrt(gain)}},
                'kUIParamMixOrGain':0.0,'type':7}
        if not isinstance(stage,dict) or stage!=wanted:return None
        params=stage['FXEQ']['plainParams']
        if (type(stage['type']) is not int
                or type(stage['kUIParamMixOrGain']) not in (int,float)
                or any(type(value) not in (int,float) or not math.isfinite(value)
                       for value in params.values())):return None
    for name,other in state.items():
        if name.startswith('FXRack') and name!='FXRack0':
            if (name not in ('FXRack1','FXRack2') or not isinstance(other,dict)
                    or not isinstance(other.get('FX',[]),list) or other.get('FX',[])):return None
    kinds=[]
    for stage in rack:
        if not isinstance(stage,dict):return None
        names=[key for key in stage if key.startswith('FX')]
        if (len(names)!=1 or names[0] not in _FX_TYPES
                or type(stage.get('type')) is not int or stage['type']!=_FX_TYPES[names[0]]
                or not isinstance(stage[names[0]],dict)):return None
        kinds.append(names[0])
    def reference(kind,index,parameter,number=None):
        if not isinstance(kind,str) or type(index) is not int or not isinstance(parameter,str):return False
        learned=ids.get((kind,parameter))
        if type(learned) is not int or (number is not None and (type(number) is not int or number!=learned)):return False
        if kind in _FX_TYPES:return 0<=index<start and kinds[index]==kind
        if kind=='ModSlot':return 0<=index<MAX_SLOTS and isinstance(state.get(f'ModSlot{index}'),dict)
        return _non_fx_owner(state,kind,index)
    active={}
    for name,slot in state.items():
        if not name.startswith('ModSlot'):continue
        suffix=name[7:]
        if (not suffix.isdigit() or str(int(suffix))!=suffix or not 0<=int(suffix)<MAX_SLOTS
                or not isinstance(slot,dict)):return None
        allowed={'source','destModuleTypeString','destModuleID','destModuleParamName','destModuleParamID','plainParams','flex'}
        if not set(slot).issubset(allowed):return None
        source=slot.get('source')
        if not isinstance(source,list) or len(source)!=2 or any(type(x) is not int for x in source):return None
        destination=any(k.startswith('destModule') for k in slot)
        if not destination:
            if any(source):return None
            continue
        # Destination ownership matters even when its source or stored amount is zero.
        if any(x not in {0,*SOURCES.values(),*range(25,33)} for x in source):return None
        if not reference(slot.get('destModuleTypeString'),slot.get('destModuleID'),
                         slot.get('destModuleParamName'),slot.get('destModuleParamID')):return None
        if type(slot.get('destModuleParamID')) is not int:return None
        active[int(suffix)]=slot
    recognized={('notes',note_index,'output_stage')}
    def record_ref(record,path,module_key,index_key,param_key,number_key,slot_key):
        if not isinstance(record,dict):return False
        if module_key not in record and index_key not in record:return True
        if module_key not in record or index_key not in record:return False
        if number_key in record and type(record[number_key]) is not int:return False
        if not reference(record[module_key],record[index_key],record.get(param_key),record.get(number_key)):return False
        recognized.add(path+(index_key,))
        if slot_key in record:
            slot=record[slot_key]
            if type(slot) is not int or slot not in active:return False
            actual=active[slot]
            if ((actual['destModuleTypeString'],actual['destModuleID'],actual['destModuleParamName'])
                    !=(record[module_key],record[index_key],record.get(param_key))):return False
            if number_key in record and record[number_key]!=actual['destModuleParamID']:return False
        elif record.get('status')=='mapped':return False
        return True
    for i,record in enumerate(report.modulations):
        path=('modulations',i)
        if not record_ref(record,path,'serum_module','serum_module_id','serum_parameter','serum_parameter_id','serum_slot'):return None
        for field in ('atomic_law_destinations','supplementary_destinations'):
            if field not in record:continue
            if not isinstance(record[field],list):return None
            for j,item in enumerate(record[field]):
                if not record_ref(item,path+(field,j),'module','module_id','parameter','parameter_id','slot'):return None
                if 'slot' not in item or type(item.get('parameter_id')) is not int:return None
    def known_fields(value,path=()):
        if isinstance(value,dict):
            for key,child in value.items():
                target=path+(key,)
                if key in ('module_id','serum_module_id','output_stage') and target not in recognized:return False
                if not known_fields(child,target):return False
        elif isinstance(value,list):
            for i,child in enumerate(value):
                if not known_fields(child,path+(i,)):return False
        return True
    if not known_fields({'notes':report.notes,'modulations':report.modulations}):return None
    return TailCleanup(start,original_master,note_index)


def _head_cleanup(preset, report, ids):
    """Validate every positional reference before removing an owned rack head."""
    if not isinstance(ids,dict):return None
    state=preset.state
    rack=state['FXRack0']['FX']
    for name,module in state.items():
        if name.startswith('FXRack') and name!='FXRack0':
            if name not in ('FXRack1','FXRack2') or not isinstance(module,dict):return None
            if not isinstance(module.get('FX',[]),list) or module.get('FX',[]):return None
    kinds=[]
    for stage in rack:
        if not isinstance(stage,dict):return None
        names=[name for name in stage if name.startswith('FX')]
        if (len(names)!=1 or names[0] not in _FX_TYPES
                or type(stage.get('type')) is not int or stage['type']!=_FX_TYPES[names[0]]
                or not isinstance(stage[names[0]],dict)):return None
        kinds.append(names[0])
    slots=[];active={}
    def stage_ref(kind,index,parameter,number=None):
        if kind not in _FX_TYPES or type(index) is not int or not 1<=index<len(rack):return False
        if kinds[index]!=kind or not isinstance(parameter,str):return False
        learned=ids.get((kind,parameter))
        return (type(learned) is int and 0<=learned<=65535
                and (number is None or type(number) is int and number==learned))
    for name,module in state.items():
        if not name.startswith('ModSlot'):continue
        suffix=name[7:]
        if (not suffix.isdigit() or str(int(suffix))!=suffix or not 0<=int(suffix)<MAX_SLOTS
                or not isinstance(module,dict)):return None
        allowed={'source','destModuleTypeString','destModuleID','destModuleParamName','destModuleParamID','plainParams','flex'}
        if not set(module).issubset(allowed):return None
        source=module.get('source')
        if not isinstance(source,list) or len(source)!=2 or any(type(x) is not int for x in source):return None
        destination=any(k.startswith('destModule') for k in module)
        if not any(source):
            if destination:return None
            continue
        if any(x not in {0,*SOURCES.values(),*range(25,33)} for x in source):return None
        kind=module.get('destModuleTypeString');index=module.get('destModuleID')
        parameter=module.get('destModuleParamName');number=module.get('destModuleParamID')
        if type(index) is not int or type(number) is not int or not 0<=number<=65535:return None
        if not isinstance(kind,str) or not isinstance(parameter,str):return None
        if kind in _FX_TYPES:
            if not stage_ref(kind,index,parameter,number):return None
            slots.append(name)
        elif kind in _NON_FX:
            if not _non_fx_owner(state,kind,index):return None
            learned=ids.get((kind,parameter))
            if type(learned) is not int or learned!=number:return None
        else:
            # FXRack and ModSlot controls are containers, not positions in FXRack0.FX.
            return None
        active[int(suffix)]=module
    fields=[];recognized=set()
    def record_ref(record,path,module_key,index_key,param_key,number_key,slot_key):
        if not isinstance(record,dict):return False
        present=module_key in record or index_key in record
        if not present:return True
        if module_key not in record or index_key not in record:return False
        kind,index=record[module_key],record[index_key]
        if number_key in record and type(record[number_key]) is not int:return False
        recognized.add(path+(index_key,))
        parameter=record.get(param_key);number=record.get(number_key)
        if kind in _FX_TYPES:
            if not stage_ref(kind,index,parameter,number):return False
        else:
            if not _non_fx_owner(state,kind,index):return False
            learned=ids.get((kind,parameter))
            if type(learned) is not int or (number_key in record and number!=learned):return False
        if slot_key in record:
            slot=record[slot_key]
            if type(slot) is not int or slot not in active:return False
            module=active[slot]
            if ((module['destModuleTypeString'],module['destModuleID'],module['destModuleParamName'])
                    !=(kind,index,parameter)):return False
            if number is not None and number!=module['destModuleParamID']:return False
        elif record.get('status')=='mapped':return False
        if kind in _FX_TYPES:fields.append(path+(index_key,))
        return True
    if not isinstance(report.modulations,list) or not isinstance(report.notes,list):return None
    for i,record in enumerate(report.modulations):
        path=('modulations',i)
        if not record_ref(record,path,'serum_module','serum_module_id','serum_parameter',
                          'serum_parameter_id','serum_slot'):return None
        for name in ('atomic_law_destinations','supplementary_destinations'):
            if name not in record:continue
            if not isinstance(record[name],list):return None
            for j,item in enumerate(record[name]):
                if not record_ref(item,path+(name,j),'module','module_id','parameter',
                                  'parameter_id','slot'):return None
                if 'slot' not in item or type(item.get('parameter_id')) is not int:return None
    for i,note in enumerate(report.notes):
        if not isinstance(note,dict):return None
        if 'output_stage' not in note:continue
        index=note['output_stage'];path=('notes',i,'output_stage')
        if (note.get('feature')!='master_volume' or type(index) is not int
                or not 1<=index<len(kinds) or kinds[index]!='FXEQ'):return None
        recognized.add(path);fields.append(path)
    def known_fields(value,path=()):
        if isinstance(value,dict):
            for key,child in value.items():
                target=path+(key,)
                if key in ('module_id','serum_module_id','output_stage') and target not in recognized:return False
                if not known_fields(child,target):return False
        elif isinstance(value,list):
            for i,child in enumerate(value):
                if not known_fields(child,path+(i,)):return False
        return True
    if not known_fields({'modulations':report.modulations,'notes':report.notes}):return None
    return HeadCleanup(tuple(slots),tuple(fields))


def _remove_head(out, notes, modulations, plan):
    out.state['FXRack0']['FX'].pop(0)
    for slot in plan.slots:out.state[slot]['destModuleID']-=1
    root={'notes':notes,'modulations':modulations}
    shifted=set()
    for path in plan.report_fields:
        node=root
        for key in path[:-1]:node=node[key]
        field=(id(node),path[-1])
        if field not in shifted:
            node[path[-1]]-=1
            shifted.add(field)


def prepare(original, preset, report, *, master_boost=1.0, fx_gain=1.0,
            destination_ids=None):
    """Require a complete, constant gain replacement and its optional slot."""
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (master_boost,fx_gain)):
        return None
    if master_boost>1.0 and fx_gain==1.0:
        factor,branch=master_boost,'master'
    elif fx_gain>1.0 and master_boost==1.0:
        factor,branch=fx_gain,'effects'
    else:return None
    if factor>4.0:return None
    global_module=preset.state.get('Global0',{})
    if not isinstance(global_module,dict):return None
    global_params=plain(global_module)
    track=original.settings.get('velocity_track',original.get('velocity_track'))
    if not isinstance(track,(int,float)) or not math.isfinite(track):return None
    velocity=abs(track)>1e-6
    if velocity and not -1.0<=track<=1.0:return None
    top=max((1.0-track)**2,1.0)
    if velocity and not (math.ceil(top/2.0-1e-9)==1 and factor*top<=2.0):return None
    if global_params.get('kParamVoiceAmp',.5)!=(0.0 if velocity else .5):return None
    master=global_params.get('kParamMasterVolume')
    if not isinstance(master,(int,float)) or not math.isfinite(master) or master<=0.0:
        return None
    mods=active_modulations(original)
    moved={m.target for m in mods.values()}
    moving_topology=('sample_on','sample_destination')+tuple(f'osc_{i}_destination' for i in (1,2,3))
    if moved.intersection(('volume','velocity_track')+moving_topology):
        return None
    # Stored zero amounts still own their destinations: auxiliary amount controls
    # and unsupported routes must not become an unexamined gain path.
    prefixes={f'osc_{i}_' for i in (1,2,3)
              if original.get(f'osc_{i}_on') or f'osc_{i}_on' in moved}
    if original.settings.get('sample_on',original.get('sample_on')) not in (0.0,1.0):return None
    if any(original.get(p+'destination') not in (0.,1.,2.,3.,4.)
           for p in ('osc_1_','osc_2_','osc_3_','sample_')):
        return None
    if any(p+'distortion_type' in moved for p in prefixes):return None
    source_modules=[preset.state.get(f'Oscillator{i}',{}) for i in range(4)]
    if any(not isinstance(module,dict) for module in source_modules):return None
    sources=[i for i,module in enumerate(source_modules) if plain(module).get('kParamEnable',0)]
    if not sources:return None
    routes=[d for k,d in preset.state.items() if k.startswith('ModSlot')
            and isinstance(d,dict) and d.get('destModuleParamName')]
    if any(d.get('destModuleTypeString')=='Global' and d.get('destModuleParamName')
           =='kParamMasterVolume' for d in routes):return None
    velocity_slot=None
    ids=KNOWN_IDS if destination_ids is None else destination_ids
    if velocity:
        velocity_slot=_velocity_slot(preset,ids,track)
        if velocity_slot is None:return None
        if plain(preset.state[velocity_slot])['kParamAmount']*factor>100.0:return None
    elif any(d.get('destModuleTypeString')=='Global' and d.get('destModuleParamName')
             =='kParamVoiceAmp' for d in routes):return None
    if not isinstance(report.notes,list) or any(not isinstance(n,dict) for n in report.notes):return None
    notes=[n for n in report.notes if n.get('feature')=='levels' and n.get('gain_db',0)>0]
    if len(notes)!=1:return None
    cleanup=None;tail=None
    if branch=='master':
        desired=master_value(original.get('volume'),factor)
        if not math.isfinite(desired) or desired<=0.0 or abs(master-min(1.0,desired))>2e-6:
            return None
        if desired>1.0 and 'master_output_gain' not in report.mapped:return None
        if any(n.get('kind')=='clipped' and n.get('feature')=='master_volume' for n in report.notes):
            return None
        if desired>1.0:
            original_master=master_value(original.get('volume'))
            tail=_tail_cleanup(preset,report,ids,desired,original_master)
            if tail is None:return None
            master=min(1.0,original_master)
        else:master/=factor
    else:
        rack_module=preset.state.get('FXRack0',{})
        if not isinstance(rack_module,dict):return None
        rack=rack_module.get('FX',[])
        first=rack[0] if isinstance(rack,list) and rack else {}
        if not isinstance(first,dict):return None
        body=first.get('FXEQ',{})
        if not isinstance(body,dict):return None
        params=body.get('plainParams',{})
        if not isinstance(params,dict):return None
        level=params.get('kParamLevelOut')
        if (set(first)!={'FXEQ','kUIParamMixOrGain','type'} or type(first.get('type')) is not int or first['type']!=7
                or type(first.get('kUIParamMixOrGain')) not in (int,float) or first['kUIParamMixOrGain']!=0.0
                or set(body)!={'plainParams'} or set(params)!={'kParamEnable','kParamLevelOut'}
                or type(params.get('kParamEnable')) not in (int,float) or params['kParamEnable']!=1.0
                or type(level) not in (int,float)
                or not math.isfinite(level) or abs(level-.5*math.sqrt(factor))>2e-7):return None
        if any(d.get('destModuleTypeString')=='FXEQ' and d.get('destModuleID')==0 for d in routes):
            return None
        cleanup=_head_cleanup(preset,report,ids)
        if cleanup is None:return None
    slot=None
    if factor>2.0:
        slot=_constant_slot(preset,ids)
        if slot is None:return None
    return Plan(factor,branch,master,notes[0],slot,velocity_slot,track if velocity else 0.0,top if velocity else 1.0,cleanup,tail)


def apply(c, original, destination_ids=None):
    """Apply both gain changes atomically after all normal conversion retries."""
    try:
        plan=prepare(original,c.s,c.r,master_boost=c.master_boost,fx_gain=c.fx_gain,
                     destination_ids=destination_ids)
    except (ConversionError,KeyError,TypeError,ValueError,OverflowError):
        return False
    if plan is None:return False
    out=copy.deepcopy(c.s)
    if plan.velocity_slot is not None:
        # The owned normalized curve remains in 0..1. The complete coefficient
        # bound P*top <= 2 keeps both slot depth and voice amplitude in range.
        out.state[plan.velocity_slot]['plainParams']['kParamAmount']*=plan.factor
    else:out.state['Global0']['plainParams']['kParamVoiceAmp']=min(1.0,plan.factor/2.0)
    if plan.slot is not None:
        out.state[f'ModSlot{plan.slot}']=dict(
            source=[25,0],destModuleTypeString='Global',destModuleID=0,
            destModuleParamName='kParamVoiceAmp',destModuleParamID=2,
            plainParams=dict(kParamAmount=100.0*(plan.factor/2.0-1.0),
                             kParamBipolar=0.0,kParamMainCurveData=1.0),
            flex=[remap_curve([(0.0,1.0),(1.0,1.0)]),{}])
    if plan.branch=='master':
        out.state['Global0']['plainParams']['kParamMasterVolume']=plan.master
    notes=copy.deepcopy(c.r.notes)
    modulations=copy.deepcopy(c.r.modulations)
    mapped=c.r.mapped
    if plan.tail_cleanup is not None:
        tail=plan.tail_cleanup;del out.state['FXRack0']['FX'][tail.start:]
        if tail.gain>1.0:
            out.state['FXRack0']['FX'].append({'FXEQ':{'plainParams':{'kParamEnable':1.0,'kParamLevelOut':.5*math.sqrt(tail.gain)}},'kUIParamMixOrGain':0.0,'type':7})
            notes[tail.note]['gain']=tail.gain
        else:
            notes.pop(tail.note)
            mapped=[value for value in mapped if value!='master_output_gain']
    if plan.head_cleanup is not None:
        _remove_head(out,notes,modulations,plan.head_cleanup)
    note_index=next(i for i,n in enumerate(c.r.notes) if n is plan.level_note)
    if plan.tail_cleanup is not None and plan.tail_cleanup.gain<=1.0 and plan.tail_cleanup.note<note_index:
        note_index-=1
    notes.pop(note_index)
    note=dict(kind='approximation',feature='levels',
                      message='Common source attenuation is retained. A constant voice amplitude restores the input before voice filtering, and the previous static output gain compensation is removed. Existing source and filter approximations remain.',
                      gain_factor=plan.factor,voice_amplitude=plan.factor/2.0,
                      previous_gain_placement=plan.branch)
    if plan.slot is not None:
        note.update(voice_amplitude=1.0,voice_amplitude_sum=plan.factor/2.0,
                    constant_amplitude_depth=plan.factor/2.0-1.0,
                    amplitude_slot=f'ModSlot{plan.slot}')
    if plan.velocity_slot is not None:
        note=dict(kind='approximation',feature='levels',
                  message='Common source attenuation is retained. The coefficient of the preserved velocity curve restores the input before voice filtering, and the previous static output gain compensation is removed. Existing source and filter approximations remain.',
                  gain_factor=plan.factor,velocity_track=plan.velocity_track,
                  velocity_amplitude_depth=plan.factor/2.0,
                  amplitude_slot=plan.velocity_slot,previous_gain_placement=plan.branch)
        if plan.velocity_track<0.0:
            note.update(message='Common source attenuation is retained. The coefficient of the preserved normalized velocity curve restores the input before voice filtering, and the previous static output gain compensation is removed. Existing source and filter approximations remain.',
                        velocity_amplitude_depth=out.state[plan.velocity_slot]['plainParams']['kParamAmount']/100.0,
                        velocity_normalization_top=plan.velocity_top)
    notes.append(note)
    c.s=out
    c.r.notes=notes
    c.r.modulations=modulations
    c.r.mapped=mapped
    return True
