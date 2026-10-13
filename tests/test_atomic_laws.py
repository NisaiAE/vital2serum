import copy
import math
from types import SimpleNamespace
import pytest
from vital2serum.mapping import Context
from vital2serum.modulation import Destination, KNOWN_IDS, convert_modulations, _order_phase_buses
from vital2serum.report import Report
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def context(doc,template,filler=0):
    s=doc['settings']
    s.update(distortion_drive=-30,modulations=[{'source':'lfo_1','destination':'distortion_drive'}],modulation_1_amount=1)
    for i in range(filler):
        s['modulations'].append({'source':'env_2','destination':'osc_1_pan'})
        s[f'modulation_{i+2}_amount']=.01
    ctx=Context(VitalPreset(doc),copy.deepcopy(template),Report('test','test'))
    ctx.target('osc_1_pan','Oscillator',0,'kParamPan')
    law=lambda raw:.25+.5*(raw+30)/60
    destinations=[Destination('FXEQ',0,'kParamLevelOut',law=law),
                  Destination('FXDistortion',1,'kParamWet',law=law),
                  Destination('FXDistortion',1,'kParamLevelOut',law=law)]
    destinations += [Destination('FXEQ',i,'kParamLevelOut',law=law) for i in range(2,6)]
    events=[]
    ctx.atomic_law_destinations={1:(destinations,lambda:events.append('written'),lambda:events.append('omitted'))}
    return ctx,events,destinations


@pytest.mark.parametrize('filler,expected',[(0,'written'),(57,'written'),(58,'omitted'),(63,'omitted')])
def test_all_destinations_reserve_capacity_after_existing_routes(doc,template,filler,expected):
    ctx,events,destinations=context(doc,template,filler)
    convert_modulations(ctx,KNOWN_IDS)
    assert events==[expected]
    routes=[m for k,m in ctx.s.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')]
    assert sum(m['destModuleParamName']=='kParamPan' for m in routes)==filler
    assert len(routes)==filler+(7 if expected=='written' else 0)
    record=ctx.r.modulations[0]
    assert record['status']==('mapped' if expected=='written' else 'omitted')
    if expected=='written':
        assert [item['slot'] for item in record['atomic_law_destinations']]==list(range(filler,filler+7))


@pytest.mark.parametrize('failure',['id','stereo','bipolar','source_gain','moving_aux','fixed_aux_without_optin','nan','range','missing_law','empty','zero'])
def test_invalid_group_preserves_all_slots_and_calls_recovery(doc,template,failure):
    ctx,events,destinations=context(doc,template,3)
    ids=dict(KNOWN_IDS)
    if failure=='id':ids.pop(('FXDistortion','kParamWet'))
    elif failure=='stereo':ctx.v.settings['modulation_1_stereo']=1
    elif failure=='bipolar':ctx.v.settings['modulation_1_bipolar']=1
    elif failure=='source_gain':ctx.source_gain['lfo_1']=1.01
    elif failure in ('moving_aux','fixed_aux_without_optin'):
        ctx.v.settings['modulations'].append({'source':'lfo_2' if failure=='moving_aux' else 'macro_control_1','destination':'modulation_1_amount'})
        ctx.v.settings['modulation_5_amount']=.25
        if failure=='moving_aux':ctx.static_aux_law_targets={'distortion_drive'}
    elif failure=='nan':destinations[-1].law=lambda raw:math.nan
    elif failure=='range':destinations[-1].law=lambda raw:1.25
    elif failure=='missing_law':destinations[-1].law=None
    elif failure=='empty':destinations.clear()
    elif failure=='zero':ctx.v.settings['modulation_1_amount']=0
    convert_modulations(ctx,ids)
    assert events==['omitted']
    routes=[m for k,m in ctx.s.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')]
    assert len(routes)==3 and all(m['destModuleParamName']=='kParamPan' for m in routes)
    assert not any('atomic_law_destinations' in record for record in ctx.r.modulations)


def test_fixed_aux_requires_optin_and_reconstructs_saved_position(doc,template):
    ctx,events,destinations=context(doc,template)
    ctx.v.settings.update(macro_control_1=.5,modulation_1_amount=.1,modulation_2_amount=.3)
    ctx.v.settings['modulations'].append({'source':'macro_control_1','destination':'modulation_1_amount'})
    ctx.static_aux_law_targets={'distortion_drive'}
    convert_modulations(ctx,KNOWN_IDS)
    assert events==['written']
    route=ctx.s.state['ModSlot0'];amount=plain(route)['kParamAmount']/100
    assert amount==pytest.approx(.2)
    for i,y in enumerate(route['flex'][0]['yVals']):
        assert .25+amount*(1-y)==pytest.approx(.25+.5*.4*i/32)
    assert any(note['feature']=='modulation_1_static_aux' for note in ctx.r.notes)
    assert ctx.r.modulations[1]['aux_for_slot']==1


@pytest.mark.parametrize('filler,expected',[(56,'written'),(57,'omitted')])
def test_internal_extrema_count_every_signed_piece(doc,template,filler,expected):
    ctx,events,destinations=context(doc,template,filler)
    destinations[-1].law=lambda raw:.5+.3*math.sin(2*math.pi*(raw+30)/60)
    convert_modulations(ctx,KNOWN_IDS)
    assert events==[expected]
    if expected=='written':
        record=ctx.r.modulations[0]
        assert len(record['atomic_law_destinations'])==8
        signed=[ctx.s.state[f'ModSlot{i}'] for i in [62,63]]
        for i in range(33):
            value=.5+sum(plain(route)['kParamAmount']/100*(1-route['flex'][0]['yVals'][i]) for route in signed)
            assert value==pytest.approx(.5+.3*math.sin(2*math.pi*i/32))


def test_phase_reordering_renumbers_shared_atomic_records_once(template):
    state=copy.deepcopy(template.state)
    state['ModSlot0']={'source':[6,0],'destModuleTypeString':'Macro','destModuleID':4,'destModuleParamName':'kParamValue'}
    state['ModSlot1']={'source':[3,0],'destModuleTypeString':'LFO','destModuleID':0,'destModuleParamName':'kParamPhase'}
    state['ModSlot2']={'source':[29,0],'destModuleTypeString':'FXEQ','destModuleID':0,'destModuleParamName':'kParamLevelOut'}
    items=[dict(slot=0),dict(slot=2)]
    report=Report('test','test');report.modulations=[dict(serum_slot=0,serum_slots=[0,2],atomic_law_destinations=items),dict(atomic_law_destinations=items)]
    writer=SimpleNamespace(next=3,ctx=SimpleNamespace(s=SimpleNamespace(state=state),r=report))
    _order_phase_buses(writer)
    assert state['ModSlot0']['destModuleParamName']=='kParamPhase'
    assert report.modulations[0]['serum_slot']==1
    assert [item['slot'] for item in items]==[1,2]


def test_complete_and_atomic_controls_precede_optional_clock(doc,template):
    ctx,events,destinations=context(doc,template,56)
    prior=[]
    output=Destination('Oscillator',0,'kParamPan')
    ctx.optional_routings=[('env_2',output,.01,False,lambda:prior.append('optional'),lambda:prior.append('optional_omitted'))]
    ctx.complete_control_buses={2:(.25,'env_2','env_3',None,None,output,30,None,
                                   lambda:prior.append('complete'),lambda:prior.append('complete_omitted'))}
    convert_modulations(ctx,KNOWN_IDS)
    assert prior==['complete','optional_omitted'] and events==['written']
    assert ctx.r.modulations[1]['complete_control_bus']['output_slot']==56
    assert [item['slot'] for item in ctx.r.modulations[0]['atomic_law_destinations']]==list(range(57,64))


def test_complete_control_uses_last_two_slots_before_optional_clock(doc,template):
    ctx,events,_=context(doc,template,62)
    ctx.atomic_law_destinations.clear()
    prior=[]
    output=Destination('Oscillator',0,'kParamPan')
    ctx.optional_routings=[('env_2',output,.01,False,lambda:prior.append('optional'),lambda:prior.append('optional_omitted'))]
    ctx.complete_control_buses={1:(.25,'env_2','env_3',None,None,output,30,None,
                                   lambda:prior.append('complete'),lambda:prior.append('complete_omitted'))}
    convert_modulations(ctx,KNOWN_IDS)
    assert prior==['complete','optional_omitted'] and events==[]
    assert ctx.r.modulations[0]['complete_control_bus']['output_slot']==63
