import copy

import pytest

from vital2serum import modulation
from vital2serum import mapping
from vital2serum.mapping import Context, convert
from vital2serum.modulation import (
    Destination, KNOWN_IDS, SOURCES, active_modulations, amount_modulators,
    convert_modulations, finalize_modulation_reporting,
)
from vital2serum.report import Report
from vital2serum.serum import clean_template, plain
from vital2serum.vital import VitalPreset


def context(doc, template, routes, amounts, *, target='osc_1_pan', law=None):
    doc['settings']['modulations']=[dict(source=source,destination=destination) for source,destination in routes]
    doc['settings'].update({f'modulation_{i}_amount':value for i,value in enumerate(amounts,1)})
    ctx=Context(VitalPreset(copy.deepcopy(doc)),clean_template(template),Report('test'))
    ctx.target(target,'Oscillator',0,'kParamPan',law=law)
    return ctx


def slots(ctx):
    return [module for name,module in ctx.s.state.items()
            if name.startswith('ModSlot') and module.get('destModuleParamName')]


def record(ctx, slot):
    return next(item for item in ctx.r.modulations if item['vital_slot']==slot)


def finish(ctx):
    state=copy.deepcopy(ctx.s.state)
    notes=copy.deepcopy(ctx.r.notes)
    mapped=list(ctx.r.mapped)
    assets=copy.deepcopy(ctx.r.assets)
    finalize_modulation_reporting(ctx.r)
    assert ctx.s.state==state
    assert ctx.r.notes==notes and ctx.r.mapped==mapped and ctx.r.assets==assets
    result=copy.deepcopy(ctx.r.as_dict())
    finalize_modulation_reporting(ctx.r)
    assert ctx.r.as_dict()==result
    assert not hasattr(ctx.r,'_modulation_reporting_updates')


def test_unknown_aux_is_omitted_without_a_fictitious_native_slot(doc,template):
    ctx=context(doc,template,[('env_2','osc_1_pan'),('random_1','modulation_1_amount')],[.1,.1])
    convert_modulations(ctx,KNOWN_IDS)
    # Internal decisions retain their established view until the terminal step.
    assert record(ctx,2)['status']=='mapped'
    assert len(slots(ctx))==1
    assert slots(ctx)[0]['source']==[SOURCES['env_2'],0]
    assert plain(slots(ctx)[0])['kParamAmount']==10
    finish(ctx)
    assert record(ctx,1)['status']=='mapped' and record(ctx,1)['control_status']=='partial'
    aux=record(ctx,2)
    assert aux['status']=='omitted'
    assert not {'serum_slot','serum_slots','serum_amount','serum_parameter_id'}&aux.keys()
    note=next(n for n in ctx.r.notes if n['kind']=='unsupported')
    assert note['sources']==['random_1'] and note['destination']=='osc_1_pan'


def test_supported_aux_keeps_its_operation_and_consumer_group(doc,template):
    ctx=context(doc,template,[('env_2','osc_1_pan'),('lfo_1','modulation_1_amount')],[.1,.1])
    convert_modulations(ctx,KNOWN_IDS)
    before=copy.deepcopy(ctx.r.modulations)
    assert [(s['source'],plain(s)['kParamAmount']) for s in slots(ctx)]==[
        ([SOURCES['env_2'],0],10),([SOURCES['env_2'],SOURCES['lfo_1']],20)]
    finish(ctx)
    assert ctx.r.modulations==before
    assert record(ctx,2)['status']=='mapped'
    assert record(ctx,2)['serum_slots']==[0,1]
    assert not any(n['kind']=='unsupported' for n in ctx.r.notes)


def test_independent_supported_aux_survives_an_omitted_term(doc,template):
    ctx=context(doc,template,[('env_2','osc_1_pan'),('lfo_1','modulation_1_amount'),
                             ('random_1','modulation_1_amount')],[.1,.1,.2])
    convert_modulations(ctx,KNOWN_IDS)
    assert len(slots(ctx))==2 and slots(ctx)[1]['source'][1]==SOURCES['lfo_1']
    finish(ctx)
    assert record(ctx,2)['status']=='mapped' and 'control_status' not in record(ctx,2)
    assert record(ctx,3)['status']=='omitted'
    assert record(ctx,1)['control_status']=='partial'


def test_aux_preserved_in_only_some_terms_is_reported_as_partial(doc,template):
    ctx=context(doc,template,[('env_2','osc_1_pan'),('lfo_1','modulation_1_amount'),
                             ('random_1','modulation_2_amount')],[.1,.1,.2])
    convert_modulations(ctx,KNOWN_IDS)
    assert len(slots(ctx))==2
    finish(ctx)
    assert record(ctx,2)['status']=='mapped' and record(ctx,2)['control_status']=='partial'
    assert record(ctx,3)['status']=='omitted'


def test_slot_exhaustion_does_not_claim_the_unwritten_aux(doc,template,monkeypatch):
    monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    ctx=context(doc,template,[('env_2','osc_1_pan'),('lfo_1','modulation_1_amount')],[.1,.1])
    convert_modulations(ctx,KNOWN_IDS)
    assert len(slots(ctx))==1 and record(ctx,2)['status']=='mapped'
    finish(ctx)
    assert record(ctx,2)['status']=='omitted' and 'serum_slot' not in record(ctx,2)
    assert record(ctx,1)['status']=='mapped' and record(ctx,1)['control_status']=='partial'
    assert any(n['kind']=='unsupported' and n['feature']=='modulations' for n in ctx.r.notes)


def test_resource_fold_keeps_saved_contributions_without_claiming_live_controls(doc,template,monkeypatch):
    monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    doc['settings'].update(macro_control_1=.5,macro_control_2=.25)
    ctx=context(doc,template,[('env_2','osc_1_pan'),('macro_control_1','modulation_1_amount'),
                             ('macro_control_2','modulation_2_amount')],[.1,.2,.3])
    convert_modulations(ctx,KNOWN_IDS)
    assert len(slots(ctx))==1 and slots(ctx)[0]['source']==[SOURCES['env_2'],0]
    assert plain(slots(ctx)[0])['kParamAmount']==pytest.approx(45)
    finish(ctx)
    for slot in (2,3):
        assert record(ctx,slot)['status']=='fixed' and record(ctx,slot)['control_status']=='fixed'
        assert 'serum_slot' not in record(ctx,slot)
    note=next(n for n in ctx.r.notes if n.get('user_impact')=='control_fixed')
    assert note['sources']==['macro_control_1','macro_control_2']
    assert note['destination']=='osc_1_pan'


def test_optional_plan_provenance_preserves_all_generated_state(doc,template):
    doc['settings'].update(macro_control_1=.5,macro_control_2=.25)
    routes=[('env_2','osc_1_pan'),('macro_control_1','modulation_1_amount'),
            ('macro_control_2','modulation_2_amount')]
    left=context(copy.deepcopy(doc),template,routes,[.1,.2,.3])
    right=context(copy.deepcopy(doc),template,routes,[.1,.2,.3])
    outputs=[]
    for ctx,tracking in ((left,False),(right,True)):
        mods=active_modulations(ctx.v);into=amount_modulators(mods)
        terms=modulation._terms(mods,into,1,ctx.r)
        writer=modulation._Writer(ctx,KNOWN_IDS)
        options={'provenance':[],'omitted':set(),'destination':'osc_1_pan'} if tracking else {}
        outputs.append(modulation._plan(ctx.v,mods,terms,{m.target for m in mods.values()},writer,ctx.r,'modulation_1_',**options))
    assert outputs[0]==outputs[1] and left.s.state==right.s.state


def test_dynamic_aux_nonlinear_response_reports_the_operation_limit(doc,template):
    ctx=context(doc,template,[('env_2','osc_1_pan'),('lfo_1','modulation_1_amount')],
                [.1,.1],law=lambda x:x*x)
    convert_modulations(ctx,KNOWN_IDS)
    finish(ctx)
    assert record(ctx,1)['status']=='mapped' and record(ctx,2)['status']=='mapped'
    assert any(slot['source'][1]==SOURCES['lfo_1'] for slot in slots(ctx))
    notice=next(n for n in ctx.r.user_notices() if n['feature']=='modulation_1_')
    assert notice['category']=='control_partial' and notice['destination']=='osc_1_pan'


@pytest.mark.parametrize('atomic',[False,True])
def test_static_aux_response_keeps_the_curve_and_labels_the_control_fixed(doc,template,atomic):
    doc['settings'].update(macro_control_1=.5,filter_1_resonance=.2)
    ctx=context(doc,template,[('env_2','filter_1_resonance'),('macro_control_1','modulation_1_amount')],
                [.1,.2],target='filter_1_resonance',law=lambda x:x*x)
    ctx.static_aux_law_targets={'filter_1_resonance'}
    events=[]
    if atomic:
        dest=Destination('Oscillator',0,'kParamPan',law=lambda x:x*x)
        ctx.atomic_law_destinations={1:([dest],lambda:events.append('written'),lambda:events.append('omitted'))}
    convert_modulations(ctx,KNOWN_IDS)
    route=slots(ctx)[0]
    for index in (0,16,32):
        position=.04+plain(route)['kParamAmount']/100*(1-route['flex'][0]['yVals'][index])
        assert position==pytest.approx((.2+.3*index/32)**2)
    finish(ctx)
    assert record(ctx,1)['status']=='mapped' and record(ctx,2)['status']=='fixed'
    assert 'atomic_law_destinations' not in record(ctx,2)
    note=next(n for n in ctx.r.notes if n['feature']=='modulation_1_static_aux')
    assert note['user_impact']=='control_fixed' and note['sources']==['macro_control_1']
    assert not atomic or events==['written']


@pytest.mark.parametrize('empty',[False,True])
def test_atomic_resource_or_empty_amount_retains_stored_control_notice(doc,template,monkeypatch,empty):
    ctx=context(doc,template,[('env_2','osc_1_pan')],[0 if empty else .1])
    events=[]
    ctx.atomic_law_destinations={1:([Destination('Oscillator',0,'kParamPan',law=lambda x:(x+1)/2)],
                                  lambda:events.append('written'),lambda:events.append('omitted'))}
    if not empty:monkeypatch.setattr(modulation,'MAX_SLOTS',0)
    before=copy.deepcopy(ctx.s.state)
    convert_modulations(ctx,KNOWN_IDS)
    finish(ctx)
    assert events==['omitted'] and ctx.s.state==before and record(ctx,1)['status']=='omitted'
    note=next(n for n in ctx.r.notes if n['feature']=='modulation_1_atomic_law')
    assert note['user_impact']=='control_partial'
    assert (note['source'],note['destination'])==('env_2','osc_1_pan')


def test_correlated_destination_failure_is_reported_without_changing_primary(doc,template,monkeypatch):
    monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    ctx=context(doc,template,[('env_2','osc_1_pan')],[.1])
    ctx.supplementary_destinations['osc_1_pan']=[Destination('VoiceFilter',0,'kParamLevelOut',law=lambda x:(x+1)/2)]
    convert_modulations(ctx,KNOWN_IDS)
    finish(ctx)
    assert len(slots(ctx))==1 and record(ctx,1)['status']=='mapped'
    assert 'supplementary_destinations' not in record(ctx,1)
    note=next(n for n in ctx.r.notes if n['feature']=='modulation_1_makeup')
    assert note['user_impact']=='control_partial'


def test_complete_control_resource_failure_is_not_a_live_control(doc,template,monkeypatch):
    monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    ctx=context(doc,template,[('env_2','osc_1_pan')],[.1])
    events=[]
    ctx.complete_control_buses={1:(.2,'env_2','lfo_1',None,None,
        Destination('Oscillator',0,'kParamPan'),20,None,
        lambda:events.append('written'),lambda:events.append('omitted'))}
    before=copy.deepcopy(ctx.s.state)
    convert_modulations(ctx,KNOWN_IDS)
    finish(ctx)
    assert ctx.s.state==before and events==['omitted']
    assert record(ctx,1)['status']=='omitted'
    note=next(n for n in ctx.r.notes if n['feature']=='modulation_1_control_bus')
    assert note['user_impact']=='control_partial'


def test_terminal_reporting_survives_report_copy_and_keeps_current_refs(doc,template):
    ctx=context(doc,template,[('env_2','osc_1_pan'),('random_1','modulation_1_amount')],[.1,.1])
    convert_modulations(ctx,KNOWN_IDS)
    ctx.r.modulations=copy.deepcopy(ctx.r.modulations)
    current=record(ctx,1);current.update(serum_slot=4,serum_slots=[4,7])
    finish(ctx)
    assert record(ctx,1)['serum_slot']==4 and record(ctx,1)['serum_slots']==[4,7]
    assert record(ctx,2)['status']=='omitted' and 'serum_slot' not in record(ctx,2)


def test_normal_convert_returns_finalized_reporting_after_all_state_decisions(doc,template):
    doc['settings'].update(modulations=[{'source':'env_2','destination':'osc_1_pan'},
                                       {'source':'random_1','destination':'modulation_1_amount'}],
                           modulation_1_amount=.1,modulation_2_amount=.1)
    result=convert(VitalPreset(doc),template)
    aux=next(r for r in result.report.modulations if r['vital_slot']==2)
    assert aux['status']=='omitted' and 'serum_slot' not in aux
    assert not hasattr(result.report,'_modulation_reporting_updates')


@pytest.mark.parametrize('background',['plain','filter','head','tail'])
def test_final_notice_correction_cannot_change_state_or_upstream_decisions(doc,template,monkeypatch,background):
    from conftest import wave
    s=doc['settings']
    s.update(modulations=[{'source':'env_2','destination':'osc_1_pan'},
                          {'source':'lfo_1','destination':'modulation_1_amount'}],
             modulation_1_amount=.1,modulation_2_amount=.1)
    monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    if background=='filter':
        s.update(filter_1_on=1,filter_1_model=0,filter_1_style=1,filter_1_blend=0,
                 filter_1_resonance=0,filter_1_drive=20,filter_1_mix=1)
    if background in ('head','tail'):
        import math
        s.update(wavetables=[wave([3*math.sin(2*math.pi*i/2048) for i in range(2048)]),None,None],osc_1_level=1)
    if background=='head':s.update(distortion_on=1,distortion_type=0,distortion_drive=6)
    current=convert(VitalPreset(copy.deepcopy(doc)),template)
    with monkeypatch.context() as patch:
        patch.setattr(mapping,'finalize_modulation_reporting',lambda report:None)
        unfinalized=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert current.preset.to_bytes()==unfinalized.preset.to_bytes()
    assert current.preset.state==unfinalized.preset.state
    assert [(a.data,a.gain) for a in current.assets]==[(a.data,a.gain) for a in unfinalized.assets]
    assert current.report.mapped==unfinalized.report.mapped
    assert current.report.notes==unfinalized.report.notes
    aux=next(r for r in current.report.modulations if r['vital_slot']==2)
    previous=next(r for r in unfinalized.report.modulations if r['vital_slot']==2)
    assert aux['status']=='omitted' and previous['status']=='mapped'


def test_actionable_fixed_notice_keeps_aux_identity_without_claiming_all_approximations(doc,template,monkeypatch):
    monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    doc['settings'].update(macro_control_1=.5,macro_control_2=.25)
    ctx=context(doc,template,[('env_2','osc_1_pan'),('macro_control_1','modulation_1_amount'),
                             ('macro_control_2','modulation_2_amount')],[.1,.2,.3])
    convert_modulations(ctx,KNOWN_IDS)
    finish(ctx)
    notice=next(n for n in ctx.r.user_notices() if n['category']=='control_fixed')
    assert notice['sources']==['macro_control_1','macro_control_2']
    assert notice['destination']=='osc_1_pan'
    assert not any(n['feature']=='modulations' for n in ctx.r.user_notices())
