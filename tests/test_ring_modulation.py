import copy
import math
from unittest.mock import patch
import numpy as np
import pytest
from conftest import wave
from vital2serum import fm, modulation, ring_modulation, voice_gain
from vital2serum.mapping import convert
from vital2serum.modulation import KNOWN_IDS, SOURCE_LAWS
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def input_doc(doc, *, enabled=True, source='env_2', level=.4, amplitude=4.):
    settings=doc['settings']
    x=np.arange(2048)/2048
    settings.update(osc_1_on=1,osc_1_level=level,osc_1_distortion_type=10,
        osc_1_distortion_amount=.1,osc_1_unison_voices=1,osc_1_stereo_spread=1,
        osc_2_on=int(enabled),osc_2_level=0,osc_2_unison_voices=1,
        osc_2_stereo_spread=1,osc_2_stack_style=0,osc_2_distortion_type=0,
        wavetables=[wave(),wave(list(amplitude*np.sin(2*np.pi*x))),None],
        modulations=[dict(source=source,destination='osc_1_distortion_amount')],
        modulation_1_amount=.8,modulation_1_power=0,modulation_1_bipolar=0,
        modulation_1_stereo=0,modulation_1_ramp_up=-10,modulation_1_ramp_down=-10)
    return doc


def previous(doc,template):
    return convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_ring_controls=frozenset({'osc_1_','osc_2_','osc_3_'}))


def pure_equal(a,b):
    assert a.preset.to_bytes()==b.preset.to_bytes()
    assert [(asset.data,asset.gain) for asset in a.assets]==[(asset.data,asset.gain) for asset in b.assets]
    left=a.report.as_dict();right=b.report.as_dict()
    left['notes']=[n for n in left['notes'] if 'complete ring-modulation response could not' not in n['message']]
    assert left['modulations']==right['modulations']
    assert left['mapped']==right['mapped'] and left['assets']==right['assets'] and left['notes']==right['notes']


def native_position(result,module,index,parameter,raw):
    body=result.preset.state[f'{module}{index}'] if module=='Oscillator' else result.preset.state[f'Oscillator{index}'][f'WTOsc{index}']
    stored=plain(body)[parameter]
    base=math.sqrt(stored) if parameter=='kParamVolume' else stored
    value=np.full_like(raw,base)
    for key,slot in result.preset.state.items():
        if not key.startswith('ModSlot') or slot.get('destModuleTypeString')!=module or slot.get('destModuleID')!=index or slot.get('destModuleParamName')!=parameter:continue
        curve=slot['flex'][0]
        value+=plain(slot)['kParamAmount']/100*np.interp(raw,curve['xVals'],1-np.asarray(curve['yVals']))
    return value


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('source',['env_1','env_2','macro_control_1'])
@pytest.mark.parametrize('amount',[.8,-.8])
def test_serialized_gain_and_warp_follow_one_clamped_amount(template,doc,enabled,source,amount):
    input_doc(doc,enabled=enabled,source=source)
    doc['settings'].update(osc_1_distortion_amount=.5,modulation_1_amount=amount)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    route=result.report.modulations[0]
    assert route['status']=='mapped' and route.get('atomic_law_destinations')
    raw=np.linspace(0,1,200001)
    source_values=np.sqrt(raw) if source=='env_1' else raw
    depth=np.clip(.5+amount*source_values,0,1)
    ratio=4 if enabled else None
    expected=np.array([.4*math.sqrt(fm.rm_gain(a,ratio)) for a in depth])
    volume=native_position(result,'Oscillator',0,'kParamVolume',raw)
    assert np.max(np.abs(volume-expected))<3e-5
    if enabled:
        warp=native_position(result,'WTOsc',0,'kParamWarp',raw)
        expected=np.array([fm.rm_warp(a,ratio) for a in depth])
        assert np.max(np.abs(warp-expected))<3e-5
    assert not any(n['kind']=='clipped' for n in result.report.notes)
    assert all(len(slot['flex'][0]['xVals'])<=480 for key,slot in result.preset.state.items() if key.startswith('ModSlot') and slot.get('flex'))


@pytest.mark.parametrize('source',['env_2','macro_control_1'])
def test_unity_rm_gain_keeps_previous_bytes_and_report(template,doc,source):
    input_doc(doc,source=source,amplitude=1.)
    a=convert(VitalPreset(copy.deepcopy(doc)),template);b=previous(doc,template)
    assert a.preset.to_bytes()==b.preset.to_bytes()
    assert a.report.as_dict()==b.report.as_dict()


@pytest.mark.parametrize('changed',[
    ('osc_1_level','env_3'),('osc_1_unison_voices','env_3'),
    ('osc_2_wave_frame','env_3'),('osc_2_unison_voices','env_3'),
    ('osc_2_distortion_amount','env_3'),('osc_2_transpose','env_3'),
])
def test_moving_gain_or_supplier_structure_keeps_previous_result(template,doc,changed):
    input_doc(doc)
    doc['settings']['modulations'].append(dict(source=changed[1],destination=changed[0]))
    doc['settings']['modulation_2_amount']=.2
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


@pytest.mark.parametrize('guard',['bipolar','stereo','aux','ramp','multi','unison','native_source_warp','unconverted_source'])
def test_unrepresented_or_multidimensional_scope_is_unchanged(template,doc,guard):
    input_doc(doc)
    settings=doc['settings']
    if guard in ('bipolar','stereo'):settings['modulation_1_'+guard]=1
    elif guard=='ramp':settings['modulation_1_ramp_up']=-5
    elif guard in ('aux','multi'):
        settings['modulations'].append(dict(source='env_3',destination='modulation_1_amount' if guard=='aux' else 'osc_1_distortion_amount'))
        settings['modulation_2_amount']=.1
    elif guard=='unison':settings['osc_2_unison_voices']=2
    elif guard=='native_source_warp':settings.update(osc_2_distortion_type=11,osc_2_distortion_amount=.2)
    else:settings['osc_2_transpose_quantize']=7
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


@pytest.mark.parametrize('failure',['volume_id','warp_id','slots','curve','source_gain'])
def test_complete_failure_restores_assets_headroom_routes_and_input_reuse(template,doc,failure):
    input_doc(doc,level=.99)
    doc['settings'].update(osc_3_on=1,osc_3_spectral_morph_type=9,osc_3_spectral_morph_amount=.2,
                            wavetables=doc['settings']['wavetables'][:2]+[wave()])
    options={}
    if failure.endswith('_id'):
        ids=dict(KNOWN_IDS);del ids[('Oscillator','kParamVolume') if failure=='volume_id' else ('WTOsc','kParamWarp')]
        options['destination_ids']=ids
    if failure=='source_gain':
        doc['settings']['modulations'][0]['source']='lfo_1'
        doc['settings'].update(lfo_1_sync=0,lfo_1_frequency=6,lfo_1_smooth_mode=1,lfo_1_smooth_time=-8)
    with patch.object(modulation,'MAX_SLOTS',0 if failure=='slots' else modulation.MAX_SLOTS), \
         patch.object(ring_modulation,'CURVE_POLICY',(2e-5,33) if failure=='curve' else ring_modulation.CURVE_POLICY):
        vital=VitalPreset(copy.deepcopy(doc))
        a=convert(vital,template,**options)
        again=convert(vital,template,**options)
        b=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_ring_controls=frozenset({'osc_1_'}),**options)
        pure_equal(a,b)
        assert a.preset.to_bytes()==again.preset.to_bytes() and a.report.as_dict()==again.report.as_dict()
        assert any('complete ring-modulation response could not' in n['message'] for n in a.report.notes)


def test_mixed_rm_success_and_failure_preserves_other_oscillator(template,doc):
    input_doc(doc,source='lfo_1')
    settings=doc['settings'];x=np.arange(2048)/2048
    settings.update(osc_1_distortion_type=11,osc_2_level=.2,osc_2_distortion_type=11,
        osc_2_distortion_amount=.1,osc_3_on=1,osc_3_level=0,
        wavetables=[wave(),wave(),wave(list(4*np.sin(2*np.pi*x)))],
        lfo_1_sync=0,lfo_1_frequency=6,lfo_1_smooth_mode=1,lfo_1_smooth_time=-8,
        modulations=[dict(source='lfo_1',destination='osc_1_distortion_amount'),
                     dict(source='env_2',destination='osc_2_distortion_amount')],modulation_2_amount=.8)
    a=convert(VitalPreset(copy.deepcopy(doc)),template)
    b=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_ring_controls=frozenset({'osc_1_'}))
    pure_equal(a,b)
    assert any(n['feature']=='osc_2_ring_gain' and 'follows the native balance' in n['message'] for n in a.report.notes)


def test_most_reachable_gain_is_reserved_before_global_headroom(template,doc):
    input_doc(doc,level=.99)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    level_note=next(n for n in result.report.notes if n['feature']=='levels')
    lowered=level_note['gain_factor']
    maximum=.99**2*fm.rm_gain(.9,4.)
    assert lowered==pytest.approx(maximum)
    raw=np.linspace(0,1,10001)
    positions=native_position(result,'Oscillator',0,'kParamVolume',raw)
    assert positions.max()<=1+1e-12
    amplitude=plain(result.preset.state['Global0']).get('kParamVoiceAmp',.5)
    amplitude_slots={key:slot for key,slot in result.preset.state.items()
                     if key.startswith('ModSlot') and slot.get('destModuleTypeString')=='Global'
                     and slot.get('destModuleParamName')=='kParamVoiceAmp'}
    for slot in amplitude_slots.values():
        assert slot['source']==[25,0]
        assert slot['flex'][0]['yVals']==[0.,0.] and slot['flex'][1]=={}
        amplitude+=plain(slot)['kParamAmount']/100
    assert 2*amplitude==pytest.approx(lowered)
    actual=positions**2*(2*amplitude)
    expected=.99**2*((1-(.1+.8*raw))+(.1+.8*raw)*4/fm.SERUM_RM_GAIN)
    assert np.max(np.abs(actual-expected))<1e-4
    with patch.object(voice_gain,'apply',return_value=False):
        old=convert(VitalPreset(copy.deepcopy(doc)),template)
    expected_state=copy.deepcopy(old.preset.state)
    expected_state['Global0']['plainParams']['kParamVoiceAmp']=plain(result.preset.state['Global0'])['kParamVoiceAmp']
    expected_state.update(copy.deepcopy(amplitude_slots))
    if level_note['previous_gain_placement']=='master':
        expected_state['Global0']['plainParams']['kParamMasterVolume']/=lowered
    else:expected_state['FXRack0']['FX'][0]['FXEQ']['plainParams']['kParamLevelOut']=.5
    assert result.preset.state==expected_state
    assert result.assets==old.assets and result.report.assets==old.report.assets
    assert result.report.mapped==old.report.mapped and result.report.modulations==old.report.modulations
    assert result.preset.state['FXRack0']==old.preset.state['FXRack0']
    assert [n for n in result.report.notes if n['feature']!='levels']==[
        n for n in old.report.notes if n['feature']!='levels']


@pytest.mark.parametrize('amount',[.8,-.8])
@pytest.mark.parametrize('power',[-3.,0.,3.])
def test_headroom_bounds_include_each_remap_knot_and_target_clamp(amount,power):
    route=modulation.Modulation(1,'env_2','osc_1_distortion_amount',amount,False,power,False,
        ((0.,.9),(.25,.05),(.65,.8),(1.,.2)),(3.,-2.,1.))
    bounds=ring_modulation._amount_bounds(.3,route)
    dense=np.array([np.clip(.3+amount*route.value(float(x)),0,1) for x in np.linspace(0,1,20001)])
    assert bounds[0]<=dense.min()+1e-12 and bounds[1]>=dense.max()-1e-12
    assert bounds[0]==pytest.approx(dense.min(),abs=1e-8)
    assert bounds[1]==pytest.approx(dense.max(),abs=1e-8)


@pytest.mark.parametrize('value',[0.,.5,1.])
def test_nonzero_saved_macro_keeps_the_complete_native_response(template,doc,value):
    input_doc(doc,source='macro_control_1')
    doc['settings']['macro_control_1']=value
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.report.modulations[0]['status']=='mapped'
    assert result.report.modulations[0]['atomic_law_destinations']
    assert not any(n['feature']=='modulation_1_' and 'folded' in n['message'] for n in result.report.notes)
    raw=np.array([0.,.5,1.])
    volume=native_position(result,'Oscillator',0,'kParamVolume',raw)
    expected=np.array([.4*math.sqrt(fm.rm_gain(.1+.8*x,4.)) for x in raw])
    assert np.max(np.abs(volume-expected))<3e-5


@pytest.mark.parametrize('failure',['ring_curve','master_curve','both'])
def test_failure_preserves_independent_master_filter_and_spectral_results(template,doc,failure):
    input_doc(doc)
    settings=doc['settings']
    settings.update(volume=4900,velocity_track=0,macro_control_2=.6,modulation_2_amount=-.1,
        filter_1_on=1,filter_2_on=0,filter_1_model=0,filter_1_style=1,filter_1_blend=0,
        filter_1_resonance=0,filter_1_drive=10,filter_1_keytrack=0,filter_1_mix=1,
        osc_1_destination=0,osc_3_on=1,osc_3_level=.1,osc_3_destination=0,
        osc_3_spectral_morph_type=9,osc_3_spectral_morph_amount=.2,
        wavetables=settings['wavetables'][:2]+[wave()])
    settings['modulations'].append(dict(source='macro_control_2',destination='volume'))
    original=modulation._adaptive_law_points
    def limited(fn,route,policy):
        if (failure in ('ring_curve','both') and route.target=='osc_1_distortion_amount') or (failure in ('master_curve','both') and route.target=='volume'):
            raise ValueError('Response point budget')
        return original(fn,route,policy)
    with patch.object(modulation,'_adaptive_law_points',limited):
        vital=VitalPreset(copy.deepcopy(doc))
        result=convert(vital,template);again=convert(vital,template)
        options={}
        if failure in ('ring_curve','both'):options['_disabled_ring_controls']=frozenset({'osc_1_'})
        if failure in ('master_curve','both'):options['_disabled_post_master']=True
        reference=convert(VitalPreset(copy.deepcopy(doc)),template,**options)
    assert result.preset.to_bytes()==reference.preset.to_bytes()==again.preset.to_bytes()
    assert [(a.data,a.gain) for a in result.assets]==[(a.data,a.gain) for a in reference.assets]
    assert result.report.modulations==reference.report.modulations
    assert result.report.as_dict()==again.report.as_dict()
    assert [n for n in result.report.notes if n['feature']=='filter_1_structure']==[n for n in reference.report.notes if n['feature']=='filter_1_structure']
    assert any(n['feature']=='osc_3_spectral_morph' for n in result.report.notes)
    if failure=='ring_curve':assert any(r.get('destination')=='volume' and r['status']=='mapped' for r in result.report.modulations)
    if failure=='master_curve':assert any(r.get('destination')=='osc_1_distortion_amount' and r['status']=='mapped' for r in result.report.modulations)
