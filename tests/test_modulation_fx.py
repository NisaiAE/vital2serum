import copy
import math
import pytest
from vital2serum.defaults import DEFAULTS
from vital2serum.mapping import Context, convert
from vital2serum.modulation_fx import PHASER_FREQ_SPAN, convert as convert_fx, frequency_position, mix_position
from vital2serum.report import Report
from vital2serum.serum import clean_template, plain
from vital2serum.vital import VitalPreset


def phaser(result):
    return next(plain(e['FXPhaser']) for e in result.preset.state['FXRack0']['FX'] if 'FXPhaser' in e)


def context(doc,template):
    return Context(VitalPreset(doc),clean_template(template),Report('<memory>'))


def test_enabled_phaser_accepts_omitted_controls(doc,template):
    doc['settings']['phaser_on']=1
    result=convert(VitalPreset(doc),template)
    p=phaser(result)
    assert p['kParamFreq']==pytest.approx(440)
    assert p['kParamWet']==pytest.approx(100)
    assert p['kParamFeedback']==pytest.approx(50)
    assert p['kParamNumPoles']==4
    assert 'phaser' in result.report.mapped
    assert DEFAULTS['flanger_dry_wet']==.5
    assert DEFAULTS['phaser_mod_depth']==24


@pytest.mark.parametrize('sync,tempo,expected,flag',[
    (1,8,100*(7/14)**4,None),
    (2,8,100*((7-2/3)/14)**4,'kParamDotted'),
    (3,8,100*((7+2/3)/14)**4,'kParamTriplets'),
])
def test_triangle_clock_preserves_note_divisions(doc,template,sync,tempo,expected,flag):
    doc['settings'].update(phaser_on=1,phaser_sync=sync,phaser_tempo=tempo,phaser_phase_offset=0)
    result=convert(VitalPreset(doc),template)
    clock=plain(result.preset.state['LFO7'])
    assert clock['kParamBeatSync']==1
    assert clock['kParamRate']==pytest.approx(expected)
    assert clock['kParamMode']=='Free'
    if flag:assert clock[flag]==1
    assert 'kParamType' not in clock
    slots=[m for n,m in result.preset.state.items() if n.startswith('ModSlot') and m.get('source')==[13,0]]
    assert len(slots)==1
    assert slots[0]['destModuleTypeString']=='FXPhaser'
    assert slots[0]['destModuleParamID']==6
    assert plain(slots[0])['kParamAmount']==pytest.approx(200*24/PHASER_FREQ_SPAN)
    assert plain(slots[0])['kParamBipolar']==1


@pytest.mark.parametrize('settings,rate',[
    ({'phaser_sync':0,'phaser_frequency':1},2),
    ({'phaser_sync':0,'phaser_frequency':-5,'phaser_tempo':0},1/32),
])
def test_free_clock_ignores_the_stored_tempo_index(doc,template,settings,rate):
    doc['settings'].update(phaser_on=1,phaser_phase_offset=0,**settings)
    result=convert(VitalPreset(doc),template)
    clock=plain(result.preset.state['LFO7'])
    assert clock['kParamBeatSync']==0
    assert clock['kParamRate']==rate
    curve=result.preset.state['LFO7']['curveData']
    assert curve['yVals'][0]==pytest.approx(.5)


@pytest.mark.parametrize('sync',(1,2,3))
def test_frozen_phaser_with_depth_is_omitted_without_targets_or_clock(doc,template,sync):
    doc['settings'].update(phaser_on=1,phaser_sync=sync,phaser_tempo=0,phaser_mod_depth=24)
    c=context(doc,template)
    before=copy.deepcopy(c.s.state)
    assert convert_fx(c,'phaser',0) is None
    assert not c.optional_routings
    assert not c.fixed_routings
    assert not c.destinations
    assert c.s.state==before
    result=convert(VitalPreset(doc),template)
    assert not any('FXPhaser' in e for e in result.preset.state['FXRack0']['FX'])
    assert any(n['feature']=='phaser_freeze' and n['kind']=='unsupported' for n in result.report.notes)
    assert 'phaser' not in result.report.mapped


def test_static_phaser_is_supported_when_tempo_is_frozen(doc,template):
    doc['settings'].update(phaser_on=1,phaser_sync=1,phaser_tempo=0,phaser_mod_depth=0)
    result=convert(VitalPreset(doc),template)
    assert phaser(result)['kParamDepth']==0
    assert not any(n['feature']=='phaser_freeze' for n in result.report.notes)
    assert not any(m.get('destModuleTypeString')=='FXPhaser' for k,m in result.preset.state.items() if k.startswith('ModSlot'))


def test_clock_preserves_sources_aux_inputs_and_lfo_destinations(doc,template):
    doc['settings'].update(phaser_on=1,phaser_phase_offset=0,
        modulations=[{'source':'lfo_8','destination':'osc_1_level'},
                     {'source':'lfo_7','destination':'modulation_1_amount'},
                     {'source':'env_2','destination':'lfo_6_phase'}],
        modulation_1_amount=.2,modulation_2_amount=.2,modulation_3_amount=.2)
    c=context(doc,template)
    original=copy.deepcopy(c.s.state)
    c.source_gain.update(lfo_8=.75,lfo_7=.8)
    convert_fx(c,'phaser',3)
    assert c.optional_routings[0][0]=='lfo_5'
    assert c.optional_routings[0][1].index==3
    for i in (5,6,7):assert c.s.state[f'LFO{i}']==original[f'LFO{i}']
    assert c.source_gain=={'lfo_8':.75,'lfo_7':.8}


def test_full_lfo_allocation_uses_reported_native_fallback(doc,template):
    doc['settings'].update(phaser_on=1,phaser_sync=1,phaser_tempo=8,
        modulations=[{'source':f'lfo_{i}','destination':'osc_1_level'} for i in range(1,9)])
    c=context(doc,template)
    original=copy.deepcopy(c.s.state)
    name,kind,p=convert_fx(c,'phaser',2)
    assert (name,kind)==('FXPhaser',2)
    assert p['kParamRate']==pytest.approx(2)
    assert p['kParamDepth']>0
    assert not c.fixed_routings
    assert not c.optional_routings
    assert all(c.s.state[f'LFO{i}']==original[f'LFO{i}'] for i in range(8))
    assert any(n['feature']=='phaser_clock' and n['kind']=='approximation' for n in c.r.notes)


def test_bypassed_source_does_not_reserve_clock(doc,template):
    doc['settings'].update(phaser_on=1,modulations=[{'source':'lfo_8','destination':'osc_1_level'}],modulation_1_bypass=1)
    c=context(doc,template)
    convert_fx(c,'phaser',0)
    assert c.optional_routings[0][0]=='lfo_8'


@pytest.mark.parametrize('control',('phaser_dry_wet','phaser_mod_depth'))
def test_dry_or_static_phaser_does_not_allocate_clock(doc,template,control):
    doc['settings'].update(phaser_on=1,**{control:0})
    c=context(doc,template)
    convert_fx(c,'phaser',0)
    assert not c.fixed_routings
    assert not c.optional_routings
    assert not any(n['feature']=='phaser_phase_offset' for n in c.r.notes)


def test_initially_dry_phaser_keeps_its_clock_when_mix_is_modulated(doc,template):
    doc['settings'].update(phaser_on=1,phaser_dry_wet=0,phaser_mod_depth=24,
        modulations=[{'source':'env_2','destination':'phaser_dry_wet'}],modulation_1_amount=1)
    result=convert(VitalPreset(doc),template)
    slots=[m for n,m in result.preset.state.items() if n.startswith('ModSlot') and m.get('destModuleTypeString')=='FXPhaser']
    assert {m['destModuleParamName'] for m in slots}=={'kParamFreq','kParamWet'}
    clock=next(m for m in slots if m['destModuleParamName']=='kParamFreq')
    assert clock['source']==[13,0]
    assert plain(clock)['kParamAmount']>0


@pytest.mark.parametrize('center,limit',[(8,'low'),(136,'high')])
def test_sweep_clipping_is_reported(doc,template,center,limit):
    doc['settings'].update(phaser_on=1,phaser_center=center,phaser_mod_depth=48,phaser_phase_offset=0)
    result=convert(VitalPreset(doc),template)
    p=phaser(result)
    assert 20<=p['kParamFreq']<=20000
    if limit=='low':assert p['kParamFreq']==20
    sweep=next(n for n in result.report.notes if n['feature']=='phaser_sweep')
    assert sweep['kind']=='clipped'
    assert sweep['low_hz']<sweep['target_min_hz'] if limit=='low' else sweep['high_hz']>sweep['target_max_hz']
    slot=next(m for n,m in result.preset.state.items() if n.startswith('ModSlot') and m.get('destModuleTypeString')=='FXPhaser')
    assert abs(plain(slot)['kParamAmount'])<=100


def test_slowest_dotted_clock_reports_input_tempo_approximation(doc,template):
    doc['settings'].update(phaser_on=1,phaser_sync=2,phaser_tempo=1)
    c=context(doc,template)
    c.bps=1.5
    convert_fx(c,'phaser',0)
    clock=plain(c.s.state['LFO7'])
    assert clock['kParamBeatSync']==0
    assert clock['kParamRate']==pytest.approx(1.5/128/1.5)
    assert any(n['feature']=='phaser_sync' and n['kind']=='approximation' for n in c.r.notes)


def test_unmatched_flanger_is_reported_without_rack_entry(doc,template):
    doc['settings']['flanger_on']=1
    result=convert(VitalPreset(doc),template)
    assert not any('FXFlanger' in e for e in result.preset.state['FXRack0']['FX'])
    assert any(n['feature']=='flanger' and n['kind']=='unsupported' for n in result.report.notes)


@pytest.mark.parametrize('target',('phaser_center','phaser_dry_wet','phaser_feedback'))
def test_omitted_control_and_macro_routing_share_the_same_default(doc,template,target):
    doc['settings'].update(phaser_on=1,phaser_mod_depth=0,macro_control_1=.75,
        modulations=[{'source':'macro_control_1','destination':target}],modulation_1_amount=-.1)
    implicit=convert(VitalPreset(copy.deepcopy(doc)),template)
    explicit=copy.deepcopy(doc);explicit['settings'][target]=DEFAULTS[target]
    expected=convert(VitalPreset(explicit),template)
    assert implicit.preset.state==expected.preset.state
    assert implicit.report.modulations[0]['status']=='mapped'


@pytest.mark.parametrize('source',('macro_control_1','lfo_1'))
def test_feedback_routing_targets_phaser_at_its_rack_position(doc,template,source):
    doc['settings'].update(phaser_on=1,phaser_mod_depth=0,
        modulations=[{'source':source,'destination':'phaser_feedback'}],modulation_1_amount=.2)
    result=convert(VitalPreset(doc),template)
    slot=next(m for n,m in result.preset.state.items() if n.startswith('ModSlot') and m.get('destModuleTypeString')=='FXPhaser')
    assert slot['destModuleParamName']=='kParamFeedback'
    assert slot['destModuleParamID']==7
    assert slot['destModuleID']==0


def test_parameter_positions_match_written_controls():
    assert frequency_position(80)==pytest.approx(math.log(440/20)/math.log(1000))
    assert mix_position(0)==0
    assert mix_position(.5)==pytest.approx(.5)
    assert mix_position(1)==1


@pytest.mark.parametrize('count,native',[(63,False),(64,True)])
def test_clock_does_not_displace_existing_modulation_slots(doc,template,count,native):
    doc['settings'].update(phaser_on=1,phaser_sync=0,phaser_frequency=-1,phaser_phase_offset=0,
        modulations=[{'source':'env_2','destination':'osc_1_level'} for _ in range(count)])
    for i in range(1,count+1):doc['settings'][f'modulation_{i}_amount']=.1
    expected=copy.deepcopy(doc);expected['settings']['phaser_on']=0
    baseline=convert(VitalPreset(expected),template)
    result=convert(VitalPreset(doc),template)
    for i in range(count):assert result.preset.state[f'ModSlot{i}']==baseline.preset.state[f'ModSlot{i}']
    assert all(m['status']=='mapped' for m in result.report.modulations)
    assert not any(n['kind']=='unsupported' and n['feature']=='modulations' for n in result.report.notes)
    p=phaser(result)
    assert (p['kParamDepth']>0)==native
    if native:
        assert p['kParamRate']==pytest.approx(.5)
        assert result.preset.state['LFO7']==baseline.preset.state['LFO7']
        assert any(n['feature']=='phaser_clock' and 'No modulation slot' in n['message'] for n in result.report.notes)
    else:assert result.preset.state['ModSlot63']['destModuleTypeString']=='FXPhaser'


def test_clock_capacity_uses_expanded_nonlinear_routes(doc,template):
    doc['settings'].update(phaser_on=1,phaser_phase_offset=0,
        modulations=[{'source':'env_2','destination':'osc_1_level'} for _ in range(62)]+
                    [{'source':'env_2','destination':'phaser_center'}])
    for i in range(1,64):doc['settings'][f'modulation_{i}_amount']=.1
    doc['settings']['modulation_63_bipolar']=1
    expected=copy.deepcopy(doc);expected['settings']['phaser_mod_depth']=0
    baseline=convert(VitalPreset(expected),template)
    result=convert(VitalPreset(doc),template)
    assert len(result.report.modulations[-1]['serum_slots'])==2
    for i in range(64):assert result.preset.state[f'ModSlot{i}']==baseline.preset.state[f'ModSlot{i}']
    assert phaser(result)['kParamDepth']>0
    assert result.preset.state['LFO7']==baseline.preset.state['LFO7']
