"""Preserve the linear and driven sections of a serial voice filter.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import copy, math, re
from . import fm
from .effects import filter_fit
from .modulation import KNOWN_IDS, MAX_SLOTS, active_modulations, amount_modulators
from .serum import plain


def signal_scope(v, selected):
    """Sources and control generators reaching the selected voice input.

    FM/RM reads a modulator before its output level and routing, so silent output
    sources remain in the dependency closure while they supply an active warp.
    """
    mods=active_modulations(v)
    into=amount_modulators(mods)
    live={i:m for i,m in mods.items() if i not in v.folded}

    def enabled(slot, seen=frozenset()):
        if slot not in live:return False
        if slot in seen:return True
        return (abs(live[slot].amount)>1e-12 or
                any(enabled(child,seen|{slot}) for child in into.get(slot,())))

    def moving(target):
        return any(m.target==target and enabled(i) for i,m in live.items())

    roots=set()
    for p in ('osc_1_','osc_2_','osc_3_','sample_'):
        if ((v.get(p+'on') or moving(p+'on')) and
                (v.get(p+'destination')==selected-1 or moving(p+'destination')) and
                (v.get(p+'level')>0 or moving(p+'level'))):
            roots.add(p)
    sources=set(roots)
    todo=list(roots)
    while todo:
        p=todo.pop()
        if p=='sample_':continue
        kind=int(v.get(p+'distortion_type'))
        if not (v.get(p+'distortion_amount')>0 or moving(p+'distortion_amount')):continue
        dependent=fm.modulator(int(p[4]),kind)
        other=f'osc_{dependent}_' if dependent else 'sample_' if kind in (9,12) else None
        if other and (v.get(other+'on') or moving(other+'on')) and other not in sources:
            sources.add(other)
            todo.append(other)

    prefixes=set(sources)|{f'filter_{selected}_'}

    def affected(target, seen=frozenset()):
        dependency=re.fullmatch(r'modulation_(\d+)_amount',target)
        if dependency:
            parent=int(dependency[1])
            if parent in seen or parent not in live:return False
            return affected(live[parent].target,seen|{parent})
        for p in prefixes:
            if target.startswith(p):
                # A supplying oscillator is read before its output volume/pan/route.
                return p in roots or p not in sources or target[len(p):] not in ('level','pan','destination')
        return False

    # A generator's unsupported controls matter when it drives this signal chain.
    changed=True
    while changed:
        changed=False
        for i,m in live.items():
            if enabled(i) and affected(m.target) and m.source.startswith(('env_','lfo_','random_')):
                p=m.source+'_'
                if p not in prefixes:
                    prefixes.add(p)
                    changed=True
    return roots,sources,live,enabled,affected


def unresolved_signal(v, report, selected, scope=None):
    """Reject unresolved controls on the signal dependency graph, not unrelated FX."""
    roots,sources,mods,enabled,affected=scope or signal_scope(v,selected)
    for route in report.modulations:
        slot=route.get('vital_slot')
        if (route.get('status')!='mapped' and slot in mods and enabled(slot)
                and affected(mods[slot].target)):
            return True
    for note in report.notes:
        if note.get('kind') not in ('unsupported','clipped'):continue
        feature=note.get('feature','')
        if affected(feature) or affected(note.get('destination','')):
            return True
        match=re.match(r'modulation_(\d+)_',feature)
        if match:
            slot=int(match[1])
            if slot in mods and enabled(slot) and affected(mods[slot].target):return True
        if (feature=='tuning' and roots) or (feature=='sample' and 'sample_' in sources):
            return True
    return False


def post_filter_volume(report, scope):
    """Moving output volume must not have been relocated into the voice input."""
    _,_,mods,enabled,_=scope
    for slot,m in mods.items():
        if m.target!='volume' or not enabled(slot):continue
        record=next((r for r in report.modulations if r.get('vital_slot')==slot),None)
        if not record or record.get('status')!='mapped':return False
        destinations=record.get('atomic_law_destinations') or [dict(
            module=record.get('serum_module'),parameter=record.get('serum_parameter'))]
        if not destinations or any(d.get('module')!='FXEQ' or d.get('parameter')!='kParamLevelOut'
                                   for d in destinations):
            return False
    return True


def apply(c, destination_ids=None):
    """Commit the complete voice structure only after conversion and its fallbacks."""
    active=[i for i in (1,2) if c.v.get(f'filter_{i}_on')]
    if len(active)!=1:return False
    scope=signal_scope(c.v,active[0])
    if (not scope[0] or unresolved_signal(c.v,c.r,active[0],scope) or
            not post_filter_volume(c.r,scope)):
        return False
    replacement,details=prepare(c.v,c.s,destination_ids=destination_ids,
                                source_lowered=c.master_boost>1+1e-9 or c.fx_gain>1+1e-9)
    if replacement is c.s:return False
    c.s=replacement
    selected=details['selected_filter']
    prefix=f'filter_{selected}_'
    c.r.notes[:]=[note for note in c.r.notes if not (
        note.get('kind')=='approximation' and
        ((note.get('feature')==prefix+'blend' and note.get('target_type')=='BPN24') or
         (note.get('feature')==prefix+'drive' and note.get('message')==
          "Vital's filter saturates at the level of one oscillator; Serum's drive and output level are set to clip at the same place.")))]
    c.r.mapped.append(f'filter_{selected}->VoiceFilter1')
    c.r.note('approximation',f'filter_{selected}_structure',
             'The linear and driven sections use two serial voice filters. The clean section retains finite-input nonlinear residuals; source, feedback and envelope approximations remain.',
             stages=['clean two-pole','driven two-pole'],
             copied_cutoff_slots=details['cutoff_copies'])
    return True


def prepare(v, current, *, destination_ids=None, max_slots=MAX_SLOTS, max_points=480,
            source_lowered=False):
    """Prepare the complete serial voice replacement or return the exact original object."""
    ids=KNOWN_IDS if destination_ids is None else destination_ids
    if source_lowered:return current,dict(status='fallback',reason='Original voice input was lowered before filtering')
    active=[i for i in (1,2) if v.get(f'filter_{i}_on')]
    if len(active)!=1:return current,dict(status='fallback',reason='Two active filters need both original voice resources')
    p=f'filter_{active[0]}_';model=int(v.get(p+'model'));style=int(v.get(p+'style'));blend=v.get(p+'blend')
    if model!=0:return current,dict(status='fallback',reason='Not the source Analog structure')
    if style==1 and blend in (0,1,2):weight=1.;var=int(blend*50)
    elif style==4 and blend==0:weight=math.sqrt(2);var=50
    else:return current,dict(status='fallback',reason='Unmeasured response combination')
    if abs(v.get(p+'mix')-1)>1e-12:return current,dict(status='fallback',reason='Partial wet requires an independent dry branch')
    if v.get(p+'filter_input'):return current,dict(status='fallback',reason='An independent source filter input needs its original branch')
    for source in ('osc_1_','osc_2_','osc_3_','sample_'):
        if v.get(source+'on') and v.get(source+'destination') not in (0,1,3,4):
            return current,dict(status='fallback',reason='Shared or advanced input branches require additional voice resources')
    mods=active_modulations(v);into=amount_modulators(mods)
    cutoff_roots=[i for i,m in mods.items() if m.target==p+'cutoff']
    if any(into.get(i) for i in cutoff_roots):return current,dict(status='fallback',reason='Cutoff depth dependencies are not carried as a complete slot group')
    moving=[m.target for i,m in mods.items() if m.target.startswith(p) and (abs(m.amount)>1e-12 or into.get(i))]
    if any(target!=p+'cutoff' for target in moving):return current,dict(status='fallback',reason='Moving transfer controls need additional correlated responses')
    resonance=v.get(p+'resonance');drive=v.get(p+'drive')
    if resonance!=0 or drive<0 or drive>20:return current,dict(status='fallback',reason='Outside proposed source transfer scope')
    if plain(current.state['VoiceFilter1']).get('kParamEnable',0):return current,dict(status='fallback',reason='Second native voice stage already used')
    if (set(current.state['VoiceFilter1'])-{'plainParams'} or
            set(plain(current.state['VoiceFilter1']))-{'kParamEnable'}):
        return current,dict(status='fallback',reason='Unmeasured fields on the spare native voice stage')
    matched=filter_fit(('analog',12),0)[0]
    pre_res=filter_fit(('analog',12),(1/2.15)**2)[0]
    gain=10**(drive/20);ceiling=3.31;small_gain=16.
    pre_gain=weight*gain*ceiling/small_gain
    post_gain=weight/(math.sqrt(gain)*ceiling)
    if pre_gain>4 or post_gain>4:return current,dict(status='fallback',reason='Native output level headroom insufficient')
    owned={'kParamEnable','kParamType','kParamFreq','kParamVar','kParamWet','kParamReso','kParamDrive','kParamPad','kParamLevelOut'}
    old_filter=current.state['VoiceFilter0'];old_params=plain(old_filter)
    if set(old_filter)-{'plainParams'} or set(old_params)-owned-{'kParamKeyTrack'}:
        return current,dict(status='fallback',reason='Unmeasured native filter fields or assets')
    output=plain(current.state['RoutingSlot5'])
    if (set(output)-{'kParamRoutingDest','kParamFXBus1Level','kParamFXBus2Level'} or output.get('kParamRoutingDest','kRoutingDestMaster')!='kRoutingDestMaster'
            or output.get('kParamFXBus1Level',0)!=0 or output.get('kParamFXBus2Level',0)!=0):
        return current,dict(status='fallback',reason='Independent filter output branch needs its own voice resource')
    spare_output=plain(current.state['RoutingSlot6'])
    if (set(spare_output)-{'kParamRoutingDest','kParamFXBus1Level','kParamFXBus2Level'} or
            spare_output.get('kParamRoutingDest','kRoutingDestMaster')!='kRoutingDestMaster' or
            spare_output.get('kParamFXBus1Level',0)!=0 or spare_output.get('kParamFXBus2Level',0)!=0):
        return current,dict(status='fallback',reason='Unmeasured output routing on the spare voice stage')
    native_filter_slots=[(k,d) for k,d in current.state.items() if k.startswith('ModSlot') and d.get('destModuleTypeString')=='VoiceFilter' and d.get('destModuleID')==0]
    if any(d.get('destModuleParamName')!='kParamFreq' for _,d in native_filter_slots):
        return current,dict(status='fallback',reason='Moving native transfer control is outside the fixed two-stage structure')
    if any(str(d.get('destModuleTypeString','')).startswith('ModSlot') for k,d in current.state.items() if k.startswith('ModSlot') and d.get('destModuleParamName')):
        return current,dict(status='fallback',reason='Native slot depth dependencies are not duplicated')
    used={int(k[7:]) for k,d in current.state.items() if k.startswith('ModSlot') and d.get('destModuleParamName')}
    frequency=[(k,d) for k,d in current.state.items() if k.startswith('ModSlot') and d.get('destModuleTypeString')=='VoiceFilter'
               and d.get('destModuleID')==0 and d.get('destModuleParamName')=='kParamFreq']
    free=[i for i in range(max_slots) if i not in used]
    number=ids.get(('VoiceFilter','kParamFreq'))
    if frequency and (number is None or any(d.get('destModuleParamID')!=number for _,d in frequency)):
        return current,dict(status='fallback',reason='No matching complete cutoff destination identifier')
    if len(free)<len(frequency):return current,dict(status='fallback',reason='No complete cutoff slot group')
    for _,d in frequency:
        if any(len(curve.get('xVals',[]))>max_points for curve in d.get('flex',[]) if isinstance(curve,dict)):
            return current,dict(status='fallback',reason='Cutoff curve cannot be carried completely')
    # Every resource check precedes the first mutation. No retained partial architecture.
    result=copy.deepcopy(current)
    cutoff=plain(current.state['VoiceFilter0'])['kParamFreq']
    common=dict(kParamEnable=1,kParamType='LBH12',kParamFreq=cutoff,kParamVar=var,kParamWet=100)
    common.update({k:copy.deepcopy(value) for k,value in old_filter['plainParams'].items() if k not in owned})
    result.state['VoiceFilter0']['plainParams']={**common,'kParamReso':pre_res,'kParamDrive':0,'kParamPad':1,'kParamLevelOut':.5*math.sqrt(pre_gain)}
    result.state['VoiceFilter1']['plainParams']={**common,'kParamReso':matched,'kParamDrive':100,'kParamPad':0,'kParamLevelOut':.5*math.sqrt(post_gain)}
    # The original source routes and bypass branches are retained byte-for-byte. Filter output
    # routes follow the native serial convention used by the frozen converter.
    result.state['RoutingSlot5']['plainParams']=dict(kParamRoutingDest='kRoutingDestFilter',kParamFXBus1Level=0,kParamFXBus2Level=0)
    result.state['RoutingSlot6']['plainParams']=dict(kParamRoutingDest='kRoutingDestMaster',kParamFXBus1Level=0,kParamFXBus2Level=0)
    copied=[]
    for (old,d),new in zip(frequency,free):
        slot=copy.deepcopy(d);slot['destModuleID']=1;slot['destModuleParamID']=number
        result.state[f'ModSlot{new}']=slot;copied.append(dict(old=old,new=f'ModSlot{new}'))
    return result,dict(status='candidate',selected_filter=active[0],pre_gain=pre_gain,post_gain=post_gain,
        source_drive_gain=gain,source_resonance=resonance,effective_feedback=0.,
        effective_raw_resonance=0.,native_post_resonance=matched,native_pre_resonance=pre_res,
        cutoff_copies=copied,transfer_scope='Fixed zero resonance; finite input approximation',
        preserved_keytrack=old_params.get('kParamKeyTrack','native default'),
        scope='The native clean stage retains finite-input residuals')
