import copy
import pytest
from conftest import wave
from vital2serum.effects import source_level
from vital2serum.mapping import Context, convert
from vital2serum.report import Report
from vital2serum.serum import SerumPreset, clean_template, plain
from vital2serum.vital import VitalPreset


def dual(doc,**settings):
    doc['settings'].update(filter_1_on=1,filter_2_on=1,filter_1_model=0,filter_2_model=3,
                           filter_1_cutoff=84,filter_2_cutoff=60,filter_1_keytrack=0,
                           filter_2_keytrack=0,filter_1_resonance=.2,filter_2_resonance=.6)
    doc['settings'].update(settings)
    return doc


@pytest.mark.parametrize('inputs,order,serial',[
    ((0,0),(1,2),False),((0,1),(1,2),True),
    ((1,0),(2,1),True),((1,1),(2,1),True),
])
def test_two_voice_filters_keep_processing_order(template,doc,inputs,order,serial):
    dual(doc,filter_1_filter_input=inputs[0],filter_2_filter_input=inputs[1])
    result=convert(VitalPreset(doc),template)
    state=result.preset.state
    for index,selected in enumerate(order):
        assert plain(state[f'VoiceFilter{index}'])['kParamEnable']==1
        assert f'filter_{selected}->VoiceFilter{index}' in result.report.mapped
    assert plain(state['RoutingSlot5'])['kParamRoutingDest']==('kRoutingDestFilter' if serial else 'kRoutingDestMaster')
    assert plain(state['RoutingSlot6'])['kParamRoutingDest']=='kRoutingDestMaster'
    assert plain(state['RoutingSlot0'])['kParamFilterBalance']==(-100 if order[0]==1 else 100)
    assert not state['FXRack0']['FX']
    assert not any(n['feature']=='filter_routing' for n in result.report.notes)
    assert SerumPreset.from_bytes(result.preset.to_bytes()).state==state


@pytest.mark.parametrize('reverse',[False,True])
def test_filter_inputs_fx_and_direct_have_separate_routes(template,doc,reverse):
    dual(doc,filter_1_filter_input=int(reverse),filter_2_filter_input=int(not reverse),
         osc_2_on=1,osc_3_on=1,osc_2_destination=1,osc_3_destination=3,
         wavetables=[wave(),wave(),wave()])
    result=convert(VitalPreset(doc),template)
    state=result.preset.state
    assert plain(state['RoutingSlot0'])['kParamFilterBalance']==(100 if reverse else -100)
    assert plain(state['RoutingSlot1'])['kParamFilterBalance']==(-100 if reverse else 100)
    assert plain(state['RoutingSlot2'])['kParamRoutingDest']=='kRoutingDestMaster'
    assert not any('FXFilter' in e for e in state['FXRack0']['FX'])
    doc['settings']['osc_3_destination']=4
    direct=convert(VitalPreset(doc),template)
    assert plain(direct.preset.state['RoutingSlot2'])['kParamRoutingDest']=='kRoutingDestDirect'
    context=Context(VitalPreset(doc),clean_template(template),Report('',''))
    context.native_filter_routing=True
    context.voice_filter_slots={1:int(reverse),2:int(not reverse)}
    context.routing(3,1,'sample_destination')
    assert plain(context.s.state['RoutingSlot3'])['kParamFilterBalance']==(-100 if reverse else 100)


def test_each_stage_keeps_output_level_and_key_tracking(template,doc):
    dual(doc,filter_2_filter_input=1,filter_1_keytrack=.5,filter_2_keytrack=1,
         filter_1_mix=.3,filter_2_drive=10)
    result=convert(VitalPreset(doc),template)
    first,second=(plain(result.preset.state[f'VoiceFilter{i}']) for i in (0,1))
    assert second['kParamKeyTrack']==1
    note=next(m for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('source')==[17,0])
    assert note['destModuleTypeString']=='VoiceFilter' and note['destModuleID']==0
    for selected,actual in ((1,first),(2,second)):
        single=copy.deepcopy(doc)
        single['settings'][f'filter_{3-selected}_on']=0
        single['settings']['osc_1_destination']=selected-1
        expected=plain(convert(VitalPreset(single),template).preset.state['VoiceFilter0'])
        assert actual['kParamLevelOut']==pytest.approx(expected['kParamLevelOut'])


def test_both_stages_receive_their_own_cutoff_and_resonance_modulation(template,doc):
    dual(doc,filter_2_filter_input=1)
    routes=[('env_2','filter_1_cutoff'),('env_3','filter_2_cutoff'),
            ('lfo_1','filter_1_resonance'),('lfo_2','filter_2_resonance')]
    doc['settings']['modulations']=[dict(source=source,destination=dest) for source,dest in routes]
    for index in range(1,5):doc['settings'][f'modulation_{index}_amount']=.1
    result=convert(VitalPreset(doc),template)
    slots=[m for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('destModuleTypeString')]
    assert {(m['destModuleTypeString'],m['destModuleID'],m['destModuleParamName']) for m in slots}>={
        ('VoiceFilter',0,'kParamFreq'),('VoiceFilter',1,'kParamFreq'),
        ('VoiceFilter',0,'kParamReso'),('VoiceFilter',1,'kParamReso')}
    assert all(m['status']=='mapped' for m in result.report.modulations)


@pytest.mark.parametrize('destination',[2,5,14])
def test_shared_or_advanced_source_routes_keep_reported_fallback(template,doc,destination):
    dual(doc,filter_2_filter_input=1,osc_1_destination=destination)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['VoiceFilter1'])['kParamEnable']==0
    assert any('FXFilter' in e for e in result.preset.state['FXRack0']['FX'])
    assert any(n['feature']=='filter_routing' for n in result.report.notes)


def test_unconverted_resonator_does_not_expand_dual_filter_support(template,doc):
    dual(doc,filter_1_model=6,filter_1_style=1,filter_2_filter_input=1)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['VoiceFilter1'])['kParamEnable']==0
    assert any(n['kind']=='unsupported' and n['feature']=='filter_1_' for n in result.report.notes)


@pytest.mark.parametrize('primary,inputs,expected',[
    (1,(0,0),('kRoutingDestFilter','kRoutingDestMaster')),
    (2,(0,0),('kRoutingDestMaster','kRoutingDestFilter')),
    (1,(0,1),('kRoutingDestFilter','kRoutingDestMaster')),
    (2,(0,1),('kRoutingDestFilter','kRoutingDestFilter')),
    (1,(1,0),('kRoutingDestFilter','kRoutingDestFilter')),
    (2,(1,0),('kRoutingDestMaster','kRoutingDestFilter')),
    (1,(1,1),('kRoutingDestFilter','kRoutingDestFilter')),
    (2,(1,1),('kRoutingDestMaster','kRoutingDestFilter')),
])
def test_unsupported_branch_only_passes_a_reachable_surviving_filter(template,doc,primary,inputs,expected):
    dual(doc,filter_1_model=0 if primary==1 else 5,filter_2_model=0 if primary==2 else 5,
         filter_1_filter_input=inputs[0],filter_2_filter_input=inputs[1],
         osc_2_on=1,osc_2_destination=1,wavetables=[wave(),wave(),None])
    result=convert(VitalPreset(doc),template)
    for source,destination in enumerate(expected):
        assert plain(result.preset.state[f'RoutingSlot{source}'])['kParamRoutingDest']==destination
    assert plain(result.preset.state['VoiceFilter1'])['kParamEnable']==0


@pytest.mark.parametrize('destination,expected',[
    (0,'kRoutingDestMaster'),(1,'kRoutingDestMaster'),(2,'kRoutingDestFilter'),
    (3,'kRoutingDestMaster'),(4,'kRoutingDestDirect'),
])
def test_missing_or_shared_filter_routes_keep_explicit_fallback(template,doc,destination,expected):
    dual(doc,filter_1_model=5,filter_2_model=5 if destination!=2 else 0,osc_1_destination=destination)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['RoutingSlot0'])['kParamRoutingDest']==expected


def test_source_level_follows_direct_input_routes(doc):
    dual(doc,osc_1_level=.5,osc_2_on=1,osc_2_destination=1,osc_2_level=.7,
         osc_3_on=1,osc_3_destination=3,osc_3_level=1,sample_on=1,sample_destination=4,sample_level=1)
    vital=VitalPreset(doc)
    assert source_level(vital,'filter_1_')==pytest.approx(.25)
    assert source_level(vital,'filter_2_')==pytest.approx(.49)
    doc['settings']['filter_2_filter_input']=1
    assert source_level(VitalPreset(doc),'filter_2_') is None
    doc['settings']['filter_1_filter_input']=1
    assert source_level(VitalPreset(doc),'filter_1_') is None
    assert source_level(VitalPreset(doc),'filter_2_')==pytest.approx(.49)


def test_bypassing_source_does_not_change_voice_filter_drive_makeup(template,doc):
    doc['settings'].update(filter_1_on=1,filter_1_drive=20,filter_1_resonance=.5,osc_1_level=.7)
    first=plain(convert(VitalPreset(copy.deepcopy(doc)),template).preset.state['VoiceFilter0'])
    doc['settings'].update(osc_2_on=1,osc_2_destination=3,osc_2_level=1,wavetables=[wave(),wave(),None])
    second=plain(convert(VitalPreset(copy.deepcopy(doc)),template).preset.state['VoiceFilter0'])
    assert first['kParamLevelOut']==pytest.approx(second['kParamLevelOut'])
    doc['settings']['osc_2_destination']=0
    hot=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])
    assert hot['kParamLevelOut']!=pytest.approx(first['kParamLevelOut'])
