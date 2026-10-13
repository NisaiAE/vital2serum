import copy
import math
from unittest.mock import patch

import pytest
from conftest import wave
from vital2serum import filter_structure, voice_gain
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def source(doc, **settings):
    doc['settings'].update(filter_1_on=1,filter_2_on=0,filter_1_model=0,
                           filter_1_style=1,filter_1_blend=0,filter_1_resonance=0,
                           filter_1_drive=20,filter_1_keytrack=0,filter_1_mix=1,
                           osc_1_level=math.sqrt(.5))
    doc['settings'].update(settings)
    return doc


def previous(doc, template):
    vital=VitalPreset(copy.deepcopy(doc))
    with patch.object(filter_structure,'apply',return_value=False):
        result=convert(vital,template)
    return vital,result


def assert_pure_fallback(vital,result,**options):
    before=result.preset.to_bytes()
    report=copy.deepcopy(result.report.as_dict())
    assets=copy.deepcopy(result.assets)
    replacement,details=filter_structure.prepare(vital,result.preset,**options)
    assert details['status']=='fallback'
    assert replacement is result.preset
    assert replacement.to_bytes()==before
    assert result.report.as_dict()==report and result.assets==assets


@pytest.mark.parametrize('style,blend,weight,var',[(1,0,1,0),(1,1,1,50),(1,2,1,100),(4,0,math.sqrt(2),50)])
@pytest.mark.parametrize('drive',[0,10,20])
def test_fixed_structure_keeps_source_envelope_fx_and_output(template,doc,style,blend,weight,var,drive):
    source(doc,filter_1_style=style,filter_1_blend=blend,filter_1_drive=drive)
    vital,old=previous(doc,template)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    first,second=(plain(result.preset.state[f'VoiceFilter{i}']) for i in (0,1))
    assert first['kParamType']==second['kParamType']=='LBH12'
    assert first['kParamPad']==1 and first['kParamDrive']==0
    assert second['kParamPad']==0 and second['kParamDrive']==100
    assert first['kParamVar']==second['kParamVar']==var
    # The input bend and saturation ceiling follow the same source drive law.
    gain=10**(drive/20)
    assert (2*first['kParamLevelOut'])**2==pytest.approx(weight*gain*3.31/16)
    assert (2*second['kParamLevelOut'])**2==pytest.approx(weight/(math.sqrt(gain)*3.31))
    for module,value in old.preset.state.items():
        if module not in ('VoiceFilter0','VoiceFilter1','RoutingSlot5','RoutingSlot6'):
            assert result.preset.state[module]==value
    assert result.assets==old.assets and result.preset.meta==old.preset.meta
    assert plain(result.preset.state['RoutingSlot5'])['kParamRoutingDest']=='kRoutingDestFilter'
    assert plain(result.preset.state['RoutingSlot6'])['kParamRoutingDest']=='kRoutingDestMaster'
    assert any(n['feature']=='filter_1_structure' for n in result.report.notes)
    assert any(n['feature']=='envelopes' for n in result.report.notes)
    assert not any(n.get('target_type')=='BPN24' for n in result.report.notes)


@pytest.mark.parametrize('keytrack',[-1,.5,1])
@pytest.mark.parametrize('selected',[1,2])
def test_keytrack_and_all_cutoff_slots_are_carried_to_both_stages(template,doc,keytrack,selected):
    source(doc)
    doc['settings'].update(filter_1_on=int(selected==1),filter_2_on=int(selected==2),
                           osc_1_destination=selected-1)
    p=f'filter_{selected}_'
    doc['settings'].update({p+'model':0,p+'style':1,p+'blend':0,p+'mix':1,
                            p+'drive':20,p+'resonance':0,p+'keytrack':keytrack})
    doc['settings']['modulations']=[dict(source='env_2',destination=p+'cutoff'),
                                    dict(source='lfo_1',destination=p+'cutoff')]
    doc['settings'].update(modulation_1_amount=.1,modulation_2_amount=-.05,
                           modulation_1_power=2,modulation_2_bipolar=1)
    _,old=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    original=[d for k,d in old.preset.state.items() if k.startswith('ModSlot') and d.get('destModuleTypeString')=='VoiceFilter']
    first=[d for k,d in result.preset.state.items() if k.startswith('ModSlot') and d.get('destModuleTypeString')=='VoiceFilter' and d.get('destModuleID')==0]
    second=[d for k,d in result.preset.state.items() if k.startswith('ModSlot') and d.get('destModuleTypeString')=='VoiceFilter' and d.get('destModuleID')==1]
    assert first==original and len(second)==len(first)
    for left,right in zip(first,second):
        normalized=copy.deepcopy(right);normalized['destModuleID']=0
        assert normalized==left
        assert left is not right
    if keytrack==1:
        assert all(plain(result.preset.state[f'VoiceFilter{i}'])['kParamKeyTrack']==1 for i in (0,1))
    else:assert any(d['source']==[17,0] for d in first)


@pytest.mark.parametrize('change',[
    {'filter_2_on':1}, {'filter_1_mix':.5}, {'filter_1_resonance':.1},
    {'filter_1_style':0}, {'filter_1_blend':.5}, {'filter_1_model':3},
    {'filter_1_filter_input':1}, {'osc_1_destination':2},
])
def test_unmeasured_structure_keeps_previous_payload_and_report(template,doc,change):
    source(doc,**change)
    vital,old=previous(doc,template)
    assert_pure_fallback(vital,old)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()


@pytest.mark.parametrize('control',['mix','drive','resonance','blend'])
def test_moving_transfer_keeps_previous_conversion(template,doc,control):
    source(doc)
    doc['settings'].update(modulations=[dict(source='env_2',destination='filter_1_'+control)],modulation_1_amount=.1)
    vital,old=previous(doc,template)
    assert_pure_fallback(vital,old)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()


@pytest.mark.parametrize('depth',[1,2])
def test_cutoff_depth_dependency_retains_the_whole_previous_group(template,doc,depth):
    source(doc)
    routes=[dict(source='env_2',destination='filter_1_cutoff'),dict(source='lfo_1',destination='modulation_1_amount')]
    if depth==2:routes.append(dict(source='env_3',destination='modulation_2_amount'))
    doc['settings'].update(modulations=routes,modulation_1_amount=.1,modulation_2_amount=.25,modulation_3_amount=.2)
    vital,old=previous(doc,template)
    assert_pure_fallback(vital,old)


@pytest.mark.parametrize('resource',['identifier','slot','curve','source_gain'])
def test_resource_failure_has_no_partial_stage_or_cutoff_group(template,doc,resource):
    source(doc)
    doc['settings'].update(modulations=[dict(source='env_2',destination='filter_1_cutoff')],modulation_1_amount=.1)
    vital,old=previous(doc,template)
    if resource=='curve':
        slot=next(d for k,d in old.preset.state.items() if k.startswith('ModSlot') and d.get('destModuleTypeString')=='VoiceFilter')
        slot['flex']=[dict(xVals=[0,.5,1],yVals=[0,.2,1]),{}]
    options={'identifier':{'destination_ids':{}},'slot':{'max_slots':0},
             'curve':{'max_points':2},'source_gain':{'source_lowered':True}}[resource]
    assert_pure_fallback(vital,old,**options)


@pytest.mark.parametrize('field',['unknown_filter','independent_output','second_stage','native_depth','spare_field','spare_routing'])
def test_unowned_native_state_is_not_overwritten(template,doc,field):
    source(doc)
    vital,old=previous(doc,template)
    if field=='unknown_filter':old.preset.state['VoiceFilter0']['plainParams']['kParamStereo']=.25
    elif field=='independent_output':old.preset.state['RoutingSlot5']['plainParams']={**plain(old.preset.state['RoutingSlot5']),'kParamFXBus1Level':1}
    elif field=='second_stage':old.preset.state['VoiceFilter1']['plainParams']['kParamEnable']=1
    elif field=='spare_field':old.preset.state['VoiceFilter1']['plainParams']['kParamStereo']=.25
    elif field=='spare_routing':old.preset.state['RoutingSlot6']['plainParams']={'kParamFXBus2Level':1}
    else:old.preset.state['ModSlot63']=dict(source=[6,0],destModuleTypeString='ModSlot',destModuleID=0,destModuleParamName='kParamAmount')
    assert_pure_fallback(vital,old)


@pytest.mark.parametrize('kind',[7,8,10,11])
def test_zero_output_modulator_remains_in_the_signal_closure(template,doc,kind):
    dependent=2 if kind in (7,10) else 3
    source(doc,osc_1_distortion_type=kind,osc_1_distortion_amount=.2)
    doc['settings'].update({f'osc_{dependent}_on':1,f'osc_{dependent}_level':0,
                            'wavetables':[wave(),wave(),wave()],
                            'modulations':[dict(source='random_1',destination=f'osc_{dependent}_transpose')],
                            'modulation_1_amount':.1})
    _,old=previous(doc,template)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()
    assert any(m['status']=='omitted' for m in result.report.modulations)


def test_transitive_fm_controls_and_remaining_fields_guard_before_commit(template,doc):
    source(doc,osc_1_distortion_type=7,osc_1_distortion_amount=.2,
           osc_2_on=1,osc_2_level=0,osc_2_distortion_type=8,osc_2_distortion_amount=.2,
           osc_3_on=1,osc_3_level=0,osc_3_transpose_quantize=145,wavetables=[wave(),wave(),wave()])
    _,old=previous(doc,template)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert any(n['feature']=='osc_3_transpose_quantize' for n in result.report.notes)


def test_supported_moving_fm_is_not_rejected_by_its_type(template,doc):
    source(doc,osc_1_distortion_type=7,osc_2_on=1,osc_2_level=0,wavetables=[wave(),wave(),None],
           modulations=[dict(source='env_2',destination='osc_1_distortion_amount')],modulation_1_amount=.1)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['VoiceFilter1'])['kParamEnable']==1
    assert any(n['feature']=='osc_1_warp' for n in result.report.notes)


def test_unrelated_omitted_route_does_not_disable_the_voice_structure(template,doc):
    source(doc,modulations=[dict(source='random_1',destination='reverb_dry_wet')],modulation_1_amount=.1)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['VoiceFilter1'])['kParamEnable']==1
    assert any(m['status']=='omitted' for m in result.report.modulations)


@pytest.mark.parametrize('target,on,destination',[('on',0,0),('destination',1,3)])
def test_unconverted_dynamic_source_enable_or_routing_is_guarded(template,doc,target,on,destination):
    source(doc,osc_1_on=on,osc_1_destination=destination,
           modulations=[dict(source='env_2',destination='osc_1_'+target)],modulation_1_amount=.1)
    _,old=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()


def test_unconverted_dynamic_modulator_enable_is_guarded(template,doc):
    source(doc,osc_1_distortion_type=7,osc_1_distortion_amount=.2,
           osc_2_on=0,wavetables=[wave(),wave(),None],
           modulations=[dict(source='env_2',destination='osc_2_on')],modulation_1_amount=.1)
    _,old=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()


def test_unconverted_generator_control_is_part_of_the_signal_scope(template,doc):
    source(doc,lfo_1_keytrack_transpose=12,
           modulations=[dict(source='lfo_1',destination='osc_1_tune')],modulation_1_amount=.1)
    _,old=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()


def test_omitted_auxiliary_signal_control_is_traced_to_its_root(template,doc):
    source(doc,modulations=[dict(source='env_2',destination='osc_1_tune'),
                            dict(source='random_1',destination='modulation_1_amount')],
           modulation_1_amount=0,modulation_2_amount=.1)
    _,old=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()


def test_supply_output_controls_do_not_change_its_pre_level_signal(template,doc):
    source(doc,osc_1_distortion_type=7,osc_1_distortion_amount=.2,
           osc_2_on=1,osc_2_level=0,wavetables=[wave(),wave(),None],
           modulations=[dict(source='random_1',destination='osc_2_pan')],modulation_1_amount=.1)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['VoiceFilter1'])['kParamEnable']==1
    assert any(m['status']=='omitted' for m in result.report.modulations)


def assert_constant_gain_keeps_the_original_filter_structure(result,attenuated,effects=False):
    note=next(n for n in result.report.notes if n['feature']=='levels')
    factor=note['gain_factor']
    assert factor>2 and note['voice_amplitude_sum']==factor/2
    slot_name=note['amplitude_slot'];slot=result.preset.state[slot_name]
    assert attenuated.preset.state[slot_name]=={'plainParams':'default','source':[0,0]}
    assert slot['source']==[25,0] and slot['flex'][0]['yVals']==[0.,0.]
    amplitude=plain(result.preset.state['Global0'])['kParamVoiceAmp']+plain(slot)['kParamAmount']/100
    volume=plain(result.preset.state['Oscillator0'])['kParamVolume']
    assert volume*2*amplitude==pytest.approx(plain(attenuated.preset.state['Oscillator0'])['kParamVolume']*factor)
    expected=copy.deepcopy(attenuated.preset.state)
    expected['Global0']['plainParams']['kParamVoiceAmp']=1.
    expected[slot_name]=copy.deepcopy(slot)
    if effects:expected['FXRack0']['FX'].pop(0)
    else:expected['Global0']['plainParams']['kParamMasterVolume']/=factor
    assert result.preset.state==expected
    assert result.assets==attenuated.assets
    assert result.report.mapped==attenuated.report.mapped
    assert result.report.modulations==attenuated.report.modulations
    assert not any(n['feature']=='filter_1_structure' for n in result.report.notes)
    assert plain(result.preset.state['VoiceFilter1']).get('kParamEnable',0)==0


def test_source_lowering_guard_precedes_the_terminal_voice_gain(template,doc):
    source(doc,osc_1_level=1,osc_1_unison_voices=3,osc_1_stereo_spread=0)
    _,old=previous(doc,template)
    with patch.object(voice_gain,'apply',return_value=False):
        _,attenuated=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert_constant_gain_keeps_the_original_filter_structure(result,attenuated)
    assert result.preset.to_bytes()==old.preset.to_bytes()


@pytest.mark.parametrize('settings',[{'osc_1_level':0},{'osc_1_destination':1},{'osc_1_destination':3}])
def test_no_reachable_source_keeps_the_original_dark_filter(template,doc,settings):
    source(doc,**settings)
    _,old=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()


@pytest.mark.parametrize('source_name',['env_2','lfo_1','random_1'])
def test_moving_volume_before_filter_or_omitted_is_not_split(template,doc,source_name):
    source(doc,modulations=[dict(source=source_name,destination='volume')],modulation_1_amount=.1)
    _,old=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()


def test_atomic_post_filter_volume_is_distinct_from_voice_amp(template,doc):
    source(doc,modulations=[dict(source='lfo_1',destination='volume')],modulation_1_amount=.1)
    vital,old=previous(doc,template)
    scope=filter_structure.signal_scope(vital,1)
    assert not filter_structure.post_filter_volume(old.report,scope)
    record=next(r for r in old.report.modulations if r['vital_slot']==1)
    record.update(serum_module='FXEQ',serum_parameter='kParamLevelOut',
                  atomic_law_destinations=[dict(module='FXEQ',parameter='kParamLevelOut')])
    assert filter_structure.post_filter_volume(old.report,scope)
    record['atomic_law_destinations'].append(dict(module='Global',parameter='kParamVoiceAmp'))
    assert not filter_structure.post_filter_volume(old.report,scope)


def test_mix_and_trajectory_groups_survive_success_and_resource_fallback(template,doc):
    source(doc,osc_1_level=math.sqrt(.2),osc_1_distortion_type=4,
           osc_1_distortion_amount=.2,osc_1_wave_frame=10,
           distortion_on=1,distortion_mix=.25,distortion_drive=10,
           delay_on=1,delay_dry_wet=.3,
           modulations=[dict(source='env_2',destination='filter_1_cutoff'),
                        dict(source='env_3',destination='osc_1_wave_frame'),
                        dict(source='env_3',destination='osc_1_distortion_amount'),
                        dict(source='lfo_2',destination='distortion_mix'),
                        dict(source='lfo_3',destination='delay_dry_wet')],
           modulation_1_amount=.1,modulation_2_amount=.25,modulation_3_amount=.1,
           modulation_4_amount=.3,modulation_5_amount=.3)
    vital,old=previous(doc,template)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert any(r.get('joint_trajectory') for r in old.report.modulations)
    assert any(r.get('atomic_law_destinations') for r in old.report.modulations)
    assert plain(result.preset.state['VoiceFilter1'])['kParamEnable']==1
    for key,value in old.preset.state.items():
        if key not in ('VoiceFilter0','VoiceFilter1','RoutingSlot5','RoutingSlot6') and (
                not key.startswith('ModSlot') or value.get('destModuleParamName')):
            assert result.preset.state[key]==value
    assert result.assets==old.assets
    assert result.report.modulations==old.report.modulations
    assert_pure_fallback(vital,old,max_slots=0)


def test_effects_gain_recovery_keeps_the_original_filter_structure_guard(template,doc):
    source(doc,osc_1_level=1,osc_1_unison_voices=3,osc_1_stereo_spread=0,distortion_on=1)
    _,old=previous(doc,template)
    with patch.object(voice_gain,'apply',return_value=False):
        _,attenuated=previous(doc,template)
    result=convert(VitalPreset(doc),template)
    assert_constant_gain_keeps_the_original_filter_structure(result,attenuated,effects=True)
    assert result.preset.to_bytes()==old.preset.to_bytes()
