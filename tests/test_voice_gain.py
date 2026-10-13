import copy
import base64
import math
import struct
from unittest.mock import patch

import pytest
from conftest import wave
from vital2serum import filter_structure, phase_modulation, voice_gain
from vital2serum.mapping import Context, convert
from vital2serum.serum import plain
from vital2serum.modulation import KNOWN_IDS
from vital2serum.vital import VitalPreset


def source(doc, factor=1.5, effects=False):
    doc['settings'].update(
        wavetables=[wave([factor*math.sin(2*math.pi*i/2048) for i in range(2048)]),None,None],
        osc_1_level=1,volume=4000,velocity_track=0)
    if effects:doc['settings'].update(distortion_on=1,distortion_drive=0)
    return doc


def previous(doc, template, **kwargs):
    original=VitalPreset(copy.deepcopy(doc))
    contexts=[]
    def retain(c, v, *args):
        contexts.append(c)
        assert v.settings==doc['settings'] and not v.offsets and not v.folded
        return False
    with patch.object(voice_gain,'apply',side_effect=retain):
        result=convert(VitalPreset(copy.deepcopy(doc)),template,**kwargs)
    return original,result,contexts[-1]


@pytest.mark.parametrize('factor',[1.25,1.5,2.0])
@pytest.mark.parametrize('effects',[False,True])
@pytest.mark.parametrize('embed',[False,True])
def test_complete_conversion_restores_voice_gain_and_removes_only_the_owned_head(template,doc,factor,effects,embed):
    source(doc,factor,effects)
    original,old,ctx=previous(doc,template,embed=embed)
    result=convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    expected=copy.deepcopy(old.preset.state)
    expected['Global0']['plainParams']['kParamVoiceAmp']=factor/2
    if effects:expected['FXRack0']['FX'].pop(0)
    else:expected['Global0']['plainParams']['kParamMasterVolume']/=factor
    assert result.preset.state==expected
    assert result.assets==old.assets and result.preset.meta==old.preset.meta
    assert result.report.assets==old.report.assets
    assert result.report.mapped==old.report.mapped and result.report.modulations==old.report.modulations
    assert [n for n in result.report.notes if n['feature']!='levels']==[
        n for n in old.report.notes if n['feature']!='levels']
    note=next(n for n in result.report.notes if n['feature']=='levels')
    assert note['gain_factor']==factor and note['voice_amplitude']==factor/2
    assert ctx.master_boost==factor if not effects else ctx.fx_gain==factor
    assert voice_gain.prepare(original,old.preset,old.report,
                              master_boost=ctx.master_boost,fx_gain=ctx.fx_gain)
    reused=VitalPreset(copy.deepcopy(doc))
    first=convert(reused,template,embed=embed)
    second=convert(reused,template,embed=embed)
    assert first.preset.to_bytes()==second.preset.to_bytes()
    assert first.report.as_dict()==second.report.as_dict()
    assert reused.doc==doc


@pytest.mark.parametrize('effects',[False,True])
def test_original_feature_guards_run_with_the_original_lowering_flags(template,doc,effects):
    source(doc,effects=effects)
    seen=[]
    def record(c,*args):
        seen.append((c.master_boost,c.fx_gain,plain(c.s.state['Global0']).get('kParamVoiceAmp',.5)))
        return False
    with patch.object(filter_structure,'apply',side_effect=record),patch.object(phase_modulation,'apply',side_effect=record):
        result=convert(VitalPreset(doc),template)
    assert len(seen)==2 and all(x==(1.,1.5,.5) if effects else x==(1.5,1.,.5) for x in seen)
    assert plain(result.preset.state['Global0'])['kParamVoiceAmp']==.75


def test_context_factor_and_branch_do_not_depend_on_report_wording(template,doc):
    source(doc)
    original,_,ctx=previous(doc,template)
    note=next(n for n in ctx.r.notes if n['feature']=='levels')
    note['message']='A different explanation.'
    note['gain_db']=999
    assert voice_gain.apply(ctx,original)
    assert plain(ctx.s.state['Global0'])['kParamVoiceAmp']==.75


def assert_fallback(ctx, original, **kwargs):
    preset,report=ctx.s,ctx.r
    before=preset.to_bytes()
    details=copy.deepcopy(report.as_dict())
    assets=copy.deepcopy(ctx.assets)
    assert not voice_gain.apply(ctx,original,**kwargs)
    assert ctx.s is preset and ctx.r is report
    assert ctx.s.to_bytes()==before and ctx.r.as_dict()==details and ctx.assets==assets


@pytest.mark.parametrize('boost,gain',[
    (1,1),(4.01,1),(1,4.01),(1.5,1.5),(0,1.5),(-1,1.5),
    (math.nan,1),(1,math.nan),(math.inf,1),(1,math.inf),
])
def test_invalid_or_ambiguous_context_factor_preserves_whole_conversion(template,doc,boost,gain):
    source(doc)
    original,_,ctx=previous(doc,template)
    ctx.master_boost,ctx.fx_gain=boost,gain
    assert_fallback(ctx,original)


@pytest.mark.parametrize('field,value',[
    ('kParamVoiceAmp',.6),('kParamVoiceAmp',None),('kParamVoiceAmp',math.nan),
    ('kParamMasterVolume',0),('kParamMasterVolume',None),('kParamMasterVolume',math.nan),
    ('kParamMasterVolume',math.inf),('kParamMasterVolume',.123),
])
def test_invalid_global_gain_preserves_whole_conversion(template,doc,field,value):
    source(doc)
    original,_,ctx=previous(doc,template)
    ctx.s.state['Global0']['plainParams'][field]=value
    # Invalid native values cannot be serialized; guard identity is still required.
    preset,report=ctx.s,ctx.r
    assert not voice_gain.apply(ctx,original)
    assert ctx.s is preset and ctx.r is report


@pytest.mark.parametrize('field,value',[
    ('kParamLevelOut',math.nan),('kParamLevelOut',math.inf),('kParamLevelOut',None),
    ('kParamLevelOut',.5),('kParamEnable',0),('kParamGain1',1),
])
def test_unknown_head_stage_has_no_partial_voice_gain(template,doc,field,value):
    source(doc,effects=True)
    original,_,ctx=previous(doc,template)
    ctx.s.state['FXRack0']['FX'][0]['FXEQ']['plainParams'][field]=value
    preset,report=ctx.s,ctx.r
    assert not voice_gain.apply(ctx,original)
    assert ctx.s is preset and ctx.r is report
    assert plain(ctx.s.state['Global0']).get('kParamVoiceAmp',.5)==.5


@pytest.mark.parametrize('location',['global','source','rack','first','body','params'])
@pytest.mark.parametrize('value',[None,[],42])
def test_unknown_module_shapes_fall_back_without_replacing_objects(template,doc,location,value):
    source(doc,effects=True)
    original,_,ctx=previous(doc,template)
    if location=='global':ctx.s.state['Global0']=value
    elif location=='source':ctx.s.state['Oscillator0']=value
    elif location=='rack':ctx.s.state['FXRack0']=value
    elif location=='first':ctx.s.state['FXRack0']['FX'][0]=value
    elif location=='body':ctx.s.state['FXRack0']['FX'][0]['FXEQ']=value
    else:ctx.s.state['FXRack0']['FX'][0]['FXEQ']['plainParams']=value
    preset,report=ctx.s,ctx.r
    details=copy.deepcopy(report.as_dict())
    assert not voice_gain.apply(ctx,original)
    assert ctx.s is preset and ctx.r is report and report.as_dict()==details


@pytest.mark.parametrize('module,param',[('Global','kParamVoiceAmp'),('Global','kParamMasterVolume'),('FXEQ','kParamLevelOut')])
def test_native_owned_gain_destinations_are_not_replaced(template,doc,module,param):
    source(doc,effects=True)
    original,_,ctx=previous(doc,template)
    ctx.s.state['ModSlot63']={'source':[6,16],'destModuleTypeString':module,
                             'destModuleID':0,'destModuleParamName':param,
                             'plainParams':{'kParamAmount':0}}
    assert_fallback(ctx,original)


@pytest.mark.parametrize('target',['volume','velocity_track','osc_1_destination','osc_1_distortion_type','sample_on','sample_destination'])
@pytest.mark.parametrize('aux',[False,True])
def test_original_unsupported_and_zero_depth_aux_routes_guard_the_gain(template,doc,target,aux):
    source(doc)
    doc['settings'].update(modulations=[dict(source='env_2',destination=target)],modulation_1_amount=0)
    if aux:
        doc['settings']['modulations'].append(dict(source='lfo_1',destination='modulation_1_amount'))
        doc['settings']['modulation_2_amount']=.5
    original,old,ctx=previous(doc,template)
    assert_fallback(ctx,original)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes()
    assert result.report.as_dict()==old.report.as_dict()


@pytest.mark.parametrize('settings',[
    {'velocity_track':.5},{'osc_1_destination':5},{'sample_on':2},
    {'sample_destination':5},{'osc_3_destination':-1},
])
def test_unmeasured_source_paths_retain_old_state_report_and_assets(template,doc,settings):
    source(doc)
    original,_,ctx=previous(doc,template)
    original.settings.update(settings)
    assert_fallback(ctx,original)


def test_potentially_enabled_oscillator_type_route_is_in_the_original_guard(template,doc):
    source(doc)
    original,_,ctx=previous(doc,template)
    original.settings.update(osc_2_on=0,modulations=[
        dict(source='env_2',destination='osc_2_on'),
        dict(source='lfo_1',destination='osc_2_distortion_type'),
        dict(source='env_3',destination='modulation_2_amount')],
        modulation_1_amount=.5,modulation_2_amount=0,modulation_3_amount=.5)
    assert_fallback(ctx,original)


def test_unresolved_clamp_is_retained_but_owned_tail_is_minimized(template,doc):
    source(doc)
    doc['settings']['volume']=6400
    original,old,ctx=previous(doc,template)
    assert 'master_output_gain' in old.report.mapped
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    original_master=voice_gain.master_value(original.get('volume'))
    assert result.preset.state['FXRack0']['FX']==[]
    assert plain(result.preset.state['Global0'])['kParamMasterVolume']==original_master
    before=copy.deepcopy(old.preset.state)
    before['Global0']['plainParams'].update(kParamVoiceAmp=.75,kParamMasterVolume=original_master)
    before['FXRack0']['FX']=[]
    assert result.preset.state==before and result.assets==old.assets
    assert result.report.modulations==old.report.modulations
    assert 'master_output_gain' not in result.report.mapped
    ctx.r.mapped.remove('master_output_gain')
    assert_fallback(ctx,original)
    original,_,ctx=previous(doc,template)
    ctx.r.note('clipped','master_volume','An unresolved output clamp.')
    assert_fallback(ctx,original)


def test_wrong_restoration_branch_preserves_all_state(template,doc):
    source(doc)
    original,_,ctx=previous(doc,template)
    ctx.master_boost,ctx.fx_gain=1,1.5
    assert_fallback(ctx,original)


def test_missing_source_group_is_not_given_a_new_gain(template,doc):
    source(doc)
    original,_,ctx=previous(doc,template)
    ctx.s.state['Oscillator0']['plainParams']['kParamEnable']=0
    assert_fallback(ctx,original)


def test_retry_finishes_before_recovery_and_keeps_unrelated_fallback_notes(template,doc):
    source(doc)
    _,old,_=previous(doc,template,_disabled_post_master=True)
    from vital2serum import post_master
    original_register=post_master.register
    def fail(c,ids,disabled=False):
        if not disabled:
            c.post_master_failed=True;c.post_master_failure='Output resources unavailable.'
            return False
        return original_register(c,ids,disabled)
    with patch.object(post_master,'register',side_effect=fail):
        result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['Global0'])['kParamVoiceAmp']==.75
    assert any('Output resources unavailable.' in n['message'] for n in result.report.notes)
    assert result.assets==old.assets


@pytest.mark.parametrize('factor',[2.0,2.5,3.0,4.0])
@pytest.mark.parametrize('effects',[False,True])
@pytest.mark.parametrize('embed',[False,True])
def test_extended_gain_is_constant_without_changing_original_routes_or_assets(template,doc,factor,effects,embed):
    source(doc,factor,effects)
    doc['settings'].update(filter_1_on=1,filter_1_model=0,filter_1_style=0,
                           filter_1_resonance=0,filter_1_drive=20,macro_control_1=.37,
                           modulations=[dict(source='env_2',destination='filter_1_cutoff')],
                           modulation_1_amount=.1)
    _,old,ctx=previous(doc,template,embed=embed)
    reused=VitalPreset(copy.deepcopy(doc))
    result=convert(reused,template,embed=embed)
    if effects and factor==4:
        # The ordinary head stage clips this endpoint below gain four. Its
        # incomplete compensation must remain, rather than being removed.
        assert result.preset.to_bytes()==old.preset.to_bytes()
        assert result.report.as_dict()==old.report.as_dict()
        return
    expected=copy.deepcopy(old.preset.state)
    expected['Global0']['plainParams']['kParamVoiceAmp']=min(1,factor/2)
    if effects:expected['FXRack0']['FX'].pop(0)
    else:expected['Global0']['plainParams']['kParamMasterVolume']/=factor
    if factor>2:
        free=next(i for i in range(64) if expected[f'ModSlot{i}']=={'plainParams':'default','source':[0,0]})
        slot=result.preset.state[f'ModSlot{free}']
        assert slot['source']==[25,0] and slot['destModuleTypeString']=='Global'
        assert slot['destModuleID']==0 and slot['destModuleParamID']==2
        assert slot['destModuleParamName']=='kParamVoiceAmp'
        assert plain(slot)==dict(kParamAmount=100*(factor/2-1),kParamBipolar=0.,kParamMainCurveData=1.)
        curve,aux=slot['flex']
        assert aux=={} and curve==dict(xVals=[0.,1.],yVals=[0.,0.],curveVals=[.5,.5],numPoints=1)
        for position in (0.,.17,.5,.93,1.):
            # The serialized main curve is one at every Macro position.
            value=1-(curve['yVals'][0]+position*(curve['yVals'][1]-curve['yVals'][0]))
            assert 2*(1+plain(slot)['kParamAmount']/100*value)==factor
        expected[f'ModSlot{free}']=copy.deepcopy(slot)
        note=next(n for n in result.report.notes if n['feature']=='levels')
        assert note['voice_amplitude']==1 and note['voice_amplitude_sum']==factor/2
    assert result.preset.state==expected
    assert result.assets==old.assets and result.preset.meta==old.preset.meta
    assert result.report.assets==old.report.assets
    assert result.report.modulations==old.report.modulations and result.report.mapped==old.report.mapped
    assert [n for n in result.report.notes if n['feature']!='levels']==[
        n for n in old.report.notes if n['feature']!='levels']
    assert (ctx.master_boost,ctx.fx_gain)==((1,factor) if effects else (factor,1))
    again=convert(reused,template,embed=embed)
    assert again.preset.to_bytes()==result.preset.to_bytes() and again.report.as_dict()==result.report.as_dict()
    assert reused.doc==doc


@pytest.mark.parametrize('number',[None,0,3,2.0,True,'2',math.nan,math.inf])
def test_constant_gain_requires_the_exact_known_integer_destination(template,doc,number):
    source(doc,3,effects=True)
    original,_,ctx=previous(doc,template)
    ids=dict(KNOWN_IDS)
    if number is None:ids.pop(('Global','kParamVoiceAmp'))
    else:ids[('Global','kParamVoiceAmp')]=number
    assert_fallback(ctx,original,destination_ids=ids)
    _,old,_=previous(doc,template,destination_ids=ids)
    actual=convert(VitalPreset(copy.deepcopy(doc)),template,destination_ids=ids)
    assert actual.preset.to_bytes()==old.preset.to_bytes() and actual.report.as_dict()==old.report.as_dict()


@pytest.mark.parametrize('number',[None,3,True])
def test_existing_static_amplitude_path_does_not_require_a_new_id_or_slot(template,doc,number):
    source(doc,2,effects=True)
    original,_,ctx=previous(doc,template)
    ids={} if number is None else {('Global','kParamVoiceAmp'):number}
    for i in range(64):ctx.s.state[f'ModSlot{i}']={'plainParams':{},'source':[0,0]}
    old_slots={k:copy.deepcopy(v) for k,v in ctx.s.state.items() if k.startswith('ModSlot')}
    assert voice_gain.apply(ctx,original,ids)
    assert plain(ctx.s.state['Global0'])['kParamVoiceAmp']==1
    assert {k:v for k,v in ctx.s.state.items() if k.startswith('ModSlot')}==old_slots


@pytest.mark.parametrize('occupied',[
    {'plainParams':'default','source':[6,0]},
    {'plainParams':{},'source':[0,0]},
    {'plainParams':'default','source':[0,0],'flex':[{},{}]},
    None,[],
])
def test_no_wholly_empty_slot_retains_all_objects_and_assets(template,doc,occupied):
    source(doc,3,effects=True)
    original,_,ctx=previous(doc,template)
    for i in range(64):ctx.s.state[f'ModSlot{i}']=copy.deepcopy(occupied)
    assert_fallback(ctx,original)


@pytest.mark.parametrize('module',[None,[],42])
def test_missing_native_constant_source_is_not_replaced_by_a_guess(template,doc,module):
    source(doc,3,effects=True)
    original,_,ctx=previous(doc,template)
    ctx.s.state['Macro0']=module
    assert_fallback(ctx,original)


@pytest.mark.parametrize('effects',[False,True])
def test_slot_depth_destinations_protect_even_an_empty_slot(template,doc,effects):
    source(doc,3,effects=effects)
    original,_,ctx=previous(doc,template)
    ctx.s.state['ModSlot1']=dict(source=[6,0],destModuleTypeString='ModSlot',
                                destModuleID=0,destModuleParamName='kParamAmount',
                                plainParams=dict(kParamAmount=0.))
    old=copy.deepcopy(ctx.s.state)
    if effects:
        # A container destination has no verified rack-stage schema: retain the whole conversion.
        assert_fallback(ctx,original)
        return
    assert voice_gain.apply(ctx,original)
    assert ctx.s.state['ModSlot0']==old['ModSlot0'] and ctx.s.state['ModSlot1']==old['ModSlot1']
    assert ctx.s.state['ModSlot2']['destModuleParamName']=='kParamVoiceAmp'
    for k,v in old.items():
        if k not in ('Global0','FXRack0','ModSlot2'):assert ctx.s.state[k]==v


@pytest.mark.parametrize('kind,index',[('ModSlot',None),('ModSlot',True),('ModSlot',64),('ModSlot0',0)])
def test_unknown_slot_owner_does_not_reserve_a_partial_gain(template,doc,kind,index):
    source(doc,3,effects=True)
    original,_,ctx=previous(doc,template)
    ctx.s.state['ModSlot1']=dict(source=[6,0],destModuleTypeString=kind,
                                destModuleID=index,destModuleParamName='kParamAmount')
    assert_fallback(ctx,original)


@pytest.mark.parametrize('factor',[3.,4.,math.nextafter(4.,math.inf)])
def test_extended_unrecovered_master_clamp_keeps_the_old_gain(template,doc,factor):
    source(doc,factor)
    doc['settings']['volume']=6400
    original,_,ctx=previous(doc,template)
    ctx.r.mapped.remove('master_output_gain')
    assert_fallback(ctx,original)


@pytest.mark.parametrize('target',['volume','velocity_track','osc_1_distortion_type'])
def test_extended_gain_rejects_zero_primary_with_active_aux(template,doc,target):
    source(doc,3,effects=True)
    doc['settings'].update(modulations=[dict(source='env_2',destination=target),
                                       dict(source='lfo_1',destination='modulation_1_amount')],
                           modulation_1_amount=0,modulation_2_amount=.5)
    original,_,ctx=previous(doc,template)
    assert_fallback(ctx,original)


def test_original_gain_guard_decisions_precede_extended_amplitude(template,doc):
    source(doc,3,effects=True)
    seen=[]
    def record(c,*args):
        seen.append((c.master_boost,c.fx_gain,plain(c.s.state['Global0']).get('kParamVoiceAmp',.5)))
        return False
    with patch.object(filter_structure,'apply',side_effect=record),patch.object(phase_modulation,'apply',side_effect=record):
        result=convert(VitalPreset(doc),template)
    assert seen==[(1.,3.,.5),(1.,3.,.5)]
    assert plain(result.preset.state['Global0'])['kParamVoiceAmp']==1


def test_extended_failed_slot_budget_preserves_all_existing_features(template,doc):
    source(doc,3,effects=True)
    original,_,ctx=previous(doc,template)
    ctx.s.state['ModSlot0']=dict(source=[25,0],destModuleTypeString='WTOsc',
                                destModuleID=0,destModuleParamName='kParamWarp',
                                plainParams=dict(kParamAmount=10.))
    for i in range(1,64):
        ctx.s.state[f'ModSlot{i}']=dict(source=[3,0],destModuleTypeString='VoiceFilter',
                                      destModuleID=0,destModuleParamName='kParamFreq',
                                      plainParams=dict(kParamAmount=1.))
    assert_fallback(ctx,original)


def test_exact_upper_endpoint_with_a_complete_head_stage_is_supported(template,doc):
    source(doc,4,effects=True)
    original,old,ctx=previous(doc,template)
    ctx.s.state['FXRack0']['FX'][0]['FXEQ']['plainParams']['kParamLevelOut']=1.
    assert voice_gain.apply(ctx,original)
    assert plain(ctx.s.state['Global0'])['kParamVoiceAmp']==1.
    slot=next(d for k,d in ctx.s.state.items() if k.startswith('ModSlot') and d.get('destModuleParamName')=='kParamVoiceAmp')
    assert plain(slot)['kParamAmount']==100.
    assert ctx.s.state['FXRack0']['FX']==old.preset.state['FXRack0']['FX'][1:]


def enable_noise(doc):
    pcm=[int(16000*math.sin(2*math.pi*i/44)) for i in range(882)]
    doc['settings'].update(sample_on=1,sample_level=.4,sample_destination=0,
                           sample_random_phase=0,sample_loop=1,
                           sample=dict(length=len(pcm),sample_rate=44100,
                                       samples=base64.b64encode(struct.pack('<882h',*pcm)).decode()))


@pytest.mark.parametrize('kind',[7,8,9,10,11,12])
@pytest.mark.parametrize('factor',[1.5,3.])
def test_fixed_cross_operator_and_supply_graph_keep_their_original_source_law(template,doc,kind,factor):
    source(doc,factor)
    doc['settings'].update(osc_1_distortion_type=kind,osc_1_distortion_amount=.2,
                           osc_2_on=1,osc_3_on=1,osc_2_level=.2,osc_3_level=0,
                           wavetables=[doc['settings']['wavetables'][0],wave(),wave()])
    if kind in (9,12):enable_noise(doc)
    _,old,ctx=previous(doc,template)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert plain(result.preset.state['Global0'])['kParamVoiceAmp']==min(1.,ctx.master_boost/2)
    expected=copy.deepcopy(old.preset.state)
    expected['Global0']['plainParams']['kParamVoiceAmp']=min(1.,ctx.master_boost/2)
    expected['Global0']['plainParams']['kParamMasterVolume']/=ctx.master_boost
    for k,v in result.preset.state.items():
        if k.startswith('ModSlot') and v.get('destModuleParamName')=='kParamVoiceAmp':expected[k]=copy.deepcopy(v)
    assert result.preset.state==expected
    assert result.assets==old.assets and result.report.mapped==old.report.mapped
    assert result.report.modulations==old.report.modulations


@pytest.mark.parametrize('destination',[0,1,2,3,4])
@pytest.mark.parametrize('factor',[1.5,3.])
def test_fixed_noise_and_wt_destinations_restore_only_the_common_gain(template,doc,destination,factor):
    source(doc,factor)
    enable_noise(doc)
    doc['settings'].update(osc_1_destination=destination,sample_destination=destination)
    original,old,ctx=previous(doc,template)
    assets=ctx.assets
    assert voice_gain.apply(ctx,original)
    assert ctx.s.state['Oscillator0']==old.preset.state['Oscillator0']
    assert ctx.s.state['Oscillator3']==old.preset.state['Oscillator3']
    assert {k:v for k,v in ctx.s.state.items() if k.startswith('RoutingSlot')}=={
        k:v for k,v in old.preset.state.items() if k.startswith('RoutingSlot')}
    assert ctx.s.state['VoiceFilter0']==old.preset.state['VoiceFilter0']
    assert ctx.s.state['VoiceFilter1']==old.preset.state['VoiceFilter1']
    assert ctx.r.mapped==old.report.mapped and ctx.r.modulations==old.report.modulations
    assert ctx.assets is assets and ctx.s.state['Oscillator3']['NoiseOsc3']['embeddedNoiseData']


@pytest.mark.parametrize('target',['osc_1_destination','osc_3_destination','sample_destination','sample_on'])
def test_moving_source_topology_is_not_given_a_new_shared_gain(template,doc,target):
    source(doc,3)
    enable_noise(doc)
    doc['settings'].update(modulations=[dict(source='env_2',destination=target),
                                       dict(source='lfo_1',destination='modulation_1_amount')],
                           modulation_1_amount=0,modulation_2_amount=.5)
    original,_,ctx=previous(doc,template)
    assert_fallback(ctx,original)


@pytest.mark.parametrize('target',['osc_1_destination','osc_2_destination','osc_3_destination','sample_destination'])
@pytest.mark.parametrize('value',[-1,.5,5,math.nan,math.inf])
def test_unknown_source_destination_does_not_change_the_gain_group(template,doc,target,value):
    source(doc,3)
    original,_,ctx=previous(doc,template)
    original.settings[target]=value
    assert_fallback(ctx,original)


def test_direct_master_clamp_is_not_ignored_when_source_gain_is_recovered(template,doc):
    source(doc,3)
    doc['settings'].update(osc_1_destination=4,volume=6400)
    original,_,ctx=previous(doc,template)
    assert 'master_output_gain' not in ctx.r.mapped
    assert_fallback(ctx,original)


def velocity_route(preset):
    return next((key,module) for key,module in preset.state.items()
                if key.startswith('ModSlot') and module.get('destModuleTypeString')=='Global'
                and module.get('destModuleParamName')=='kParamVoiceAmp')


@pytest.mark.parametrize('track',[math.nextafter(1e-6,1.),.01,.5,1.])
@pytest.mark.parametrize('factor',[1.25,1.5,2.])
@pytest.mark.parametrize('effects',[False,True])
@pytest.mark.parametrize('embed',[False,True])
def test_positive_velocity_keeps_the_owned_curve_and_scales_only_its_depth(template,doc,track,factor,effects,embed):
    source(doc,factor,effects)
    doc['settings'].update(velocity_track=track,macro_control_1=.37,
                           filter_1_on=1,filter_1_model=0,filter_1_style=0,
                           filter_1_resonance=0,filter_1_drive=20,
                           modulations=[dict(source='env_2',destination='filter_1_cutoff')],
                           modulation_1_amount=.1)
    _,old,ctx=previous(doc,template,embed=embed)
    result=convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    key,old_route=velocity_route(old.preset)
    _,new_route=velocity_route(result.preset)
    expected=copy.deepcopy(old.preset.state)
    expected[key]['plainParams']['kParamAmount']=50*factor
    if effects:expected['FXRack0']['FX'].pop(0)
    else:expected['Global0']['plainParams']['kParamMasterVolume']/=factor
    assert result.preset.state==expected
    assert plain(result.preset.state['Global0'])['kParamVoiceAmp']==0
    assert new_route['flex']==old_route['flex']
    assert old_route['source']==new_route['source']==[16,0]
    assert result.assets==old.assets and result.preset.meta==old.preset.meta
    assert result.report.assets==old.report.assets
    assert result.report.mapped==old.report.mapped and result.report.modulations==old.report.modulations
    assert [n for n in result.report.notes if n['feature']!='levels']==[
        n for n in old.report.notes if n['feature']!='levels']
    note=next(n for n in result.report.notes if n['feature']=='levels')
    assert note['gain_factor']==factor and note['velocity_track']==track
    assert note['velocity_amplitude_depth']==factor/2 and note['amplitude_slot']==key
    assert 'constant' not in note['message'].lower() and 'preserved velocity curve' in note['message']
    curve=old_route['flex'][0]
    # Every linear segment is bounded by its endpoint values. Scaling the
    # coefficient preserves the same interpolation error and pre-filter input.
    for index in range(1025):
        velocity=index/1024
        segment=min(31,int(velocity*32))
        fraction=velocity*32-segment
        value=1-((1-fraction)*curve['yVals'][segment]+fraction*curve['yVals'][segment+1])
        assert (1-track)**2-1e-15<=value<=1
        old_amp=plain(old_route)['kParamAmount']/100*value
        new_amp=plain(new_route)['kParamAmount']/100*value
        assert 0<=new_amp<=1 and new_amp/factor==pytest.approx(old_amp,abs=2e-16)
        old_input=plain(old.preset.state['Oscillator0'])['kParamVolume']*2*old_amp
        new_input=plain(result.preset.state['Oscillator0'])['kParamVolume']*2*new_amp
        assert new_input/factor==pytest.approx(old_input,abs=2e-16)
    assert (ctx.master_boost,ctx.fx_gain)==((1,factor) if effects else (factor,1))
    reused=VitalPreset(copy.deepcopy(doc))
    again=convert(reused,template,embed=embed)
    second=convert(reused,template,embed=embed)
    assert again.preset.to_bytes()==second.preset.to_bytes()==result.preset.to_bytes()
    assert again.report.as_dict()==second.report.as_dict()==result.report.as_dict()
    assert reused.doc==doc


@pytest.mark.parametrize('track',[math.nextafter(1.,math.inf),math.nextafter(-1.,-math.inf),math.nan,math.inf])
def test_velocity_outside_the_fixed_tracking_domain_keeps_the_whole_conversion(template,doc,track):
    source(doc)
    doc['settings']['velocity_track']=1
    original,_,ctx=previous(doc,template)
    original.settings['velocity_track']=track
    assert_fallback(ctx,original)


@pytest.mark.parametrize('track',[-1.,-.5,-.2])
def test_negative_velocity_above_the_complete_amplitude_bound_preserves_the_whole_conversion(template,doc,track):
    source(doc)
    doc['settings']['velocity_track']=track
    original,old,ctx=previous(doc,template)
    assert ctx.master_boost*max((1-track)**2,1)>2
    assert_fallback(ctx,original)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes() and result.report.as_dict()==old.report.as_dict()


@pytest.mark.parametrize('factor',[math.nextafter(2.,math.inf),2.5,3.,4.])
def test_velocity_cannot_exceed_the_owned_depth_budget(template,doc,factor):
    source(doc,factor)
    doc['settings']['velocity_track']=1
    original,_,ctx=previous(doc,template)
    # PCM serialization rounds the nextafter endpoint to two. Exercise the
    # Context boundary directly rather than assuming binary64 PCM storage.
    ctx.master_boost=factor
    assert_fallback(ctx,original)


@pytest.mark.parametrize('number',[None,0,3,2.0,True,'2',math.nan,math.inf])
def test_velocity_requires_an_exact_known_destination_id(template,doc,number):
    source(doc)
    doc['settings']['velocity_track']=1
    original,_,ctx=previous(doc,template)
    ids={} if number is None else {('Global','kParamVoiceAmp'):number}
    assert_fallback(ctx,original,destination_ids=ids)


@pytest.mark.parametrize('alteration',[
    'source','aux_source','source_bool','module_id','module_id_bool','param_id','param_id_bool',
    'bipolar','depth','main_curve_disabled','main_power','aux_curve','curve','curve_count',
    'curve_power','extra_field','extra_route','incoming','unknown_incoming','base','missing',
])
def test_noncanonical_or_shared_velocity_routes_keep_objects_report_and_assets(template,doc,alteration):
    source(doc,effects=True)
    doc['settings']['velocity_track']=.5
    original,_,ctx=previous(doc,template)
    key,route=velocity_route(ctx.s)
    params=route['plainParams']
    if alteration=='source':route['source'][0]=25
    elif alteration=='aux_source':route['source'][1]=6
    elif alteration=='source_bool':route['source'][1]=False
    elif alteration=='module_id':route['destModuleID']=1
    elif alteration=='module_id_bool':route['destModuleID']=False
    elif alteration=='param_id':route['destModuleParamID']=3
    elif alteration=='param_id_bool':route['destModuleParamID']=True
    elif alteration=='bipolar':params['kParamBipolar']=1.
    elif alteration=='depth':params['kParamAmount']=49.9
    elif alteration=='main_curve_disabled':params['kParamMainCurveData']=0.
    elif alteration=='main_power':params['kParamMainCurvePower']=1.
    elif alteration=='aux_curve':route['flex'][1]={'numPoints':1}
    elif alteration=='curve':route['flex'][0]['yVals'][1]+=.0001
    elif alteration=='curve_count':route['flex'][0]['numPoints']=31
    elif alteration=='curve_power':route['flex'][0]['curveVals'][1]=.51
    elif alteration=='extra_field':route['unknown']=0
    elif alteration=='extra_route':ctx.s.state['ModSlot63']=copy.deepcopy(route)
    elif alteration in ('incoming','unknown_incoming'):
        ctx.s.state['ModSlot63']=dict(source=[6,0],destModuleTypeString='ModSlot' if alteration=='incoming' else 'ModSlot0',
                                      destModuleID=int(key[7:]),destModuleParamName='kParamAmount',
                                      plainParams={'kParamAmount':0.})
    elif alteration=='base':plain(ctx.s.state['Global0'])['kParamVoiceAmp']=.01
    else:del ctx.s.state[key]
    assert_fallback(ctx,original)


@pytest.mark.parametrize('target',['volume','velocity_track'])
def test_velocity_rejects_original_zero_depth_with_active_aux_or_unsupported_source(template,doc,target):
    source(doc)
    doc['settings'].update(velocity_track=.5,modulations=[
        dict(source='unsupported_source',destination=target),
        dict(source='lfo_1',destination='modulation_1_amount')],
        modulation_1_amount=0,modulation_2_amount=.5)
    original,old,ctx=previous(doc,template)
    assert_fallback(ctx,original)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes() and result.report.as_dict()==old.report.as_dict()


def test_velocity_uses_its_existing_slot_when_no_new_slots_are_free(template,doc):
    source(doc)
    doc['settings']['velocity_track']=1
    original,_,ctx=previous(doc,template)
    key,route=velocity_route(ctx.s)
    for index in range(64):
        slot=f'ModSlot{index}'
        if slot!=key:ctx.s.state[slot]={'plainParams':'default','source':[6,0]}
    before=copy.deepcopy(ctx.s.state)
    assert voice_gain.apply(ctx,original)
    expected=copy.deepcopy(before)
    expected[key]['plainParams']['kParamAmount']=75.
    expected['Global0']['plainParams']['kParamMasterVolume']/=1.5
    assert ctx.s.state==expected


@pytest.mark.parametrize('track',[0.,1e-6,-1e-6])
def test_tracking_below_the_existing_activation_boundary_keeps_the_constant_path(template,doc,track):
    source(doc)
    doc['settings']['velocity_track']=track
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert plain(result.preset.state['Global0'])['kParamVoiceAmp']==.75
    assert 'velocity_track' not in next(n for n in result.report.notes if n['feature']=='levels')
