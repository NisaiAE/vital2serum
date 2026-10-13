import copy
from unittest.mock import patch
import numpy as np
import pytest
from conftest import wave
from vital2serum import phase_modulation
from vital2serum.mapping import convert
from vital2serum.modulation import KNOWN_IDS
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def source(doc,source_name='env_2'):
    s=doc['settings'];x=np.arange(2048)/2048
    s.update(osc_1_on=1,osc_1_level=.4,osc_1_distortion_type=7,osc_1_distortion_amount=0,
        osc_1_unison_voices=1,osc_1_unison_detune=0,osc_1_stereo_spread=1,osc_1_stack_style=0,
        osc_1_phase=.5,osc_1_random_phase=0,osc_1_wave_frame=0,osc_1_spectral_morph_type=0,
        osc_1_frame_spread=0,osc_1_distortion_spread=0,osc_1_spectral_morph_spread=0,
        osc_1_transpose=0,osc_1_tune=0,osc_1_midi_track=1,
        osc_2_on=1,osc_2_level=0,osc_2_distortion_type=0,osc_2_spectral_morph_type=0,
        osc_2_unison_voices=1,osc_2_unison_detune=0,osc_2_stereo_spread=1,osc_2_stack_style=0,
        osc_2_phase=.5,osc_2_random_phase=0,osc_2_wave_frame=0,osc_2_transpose=0,osc_2_tune=0,
        osc_2_midi_track=1,osc_2_frame_spread=0,osc_2_distortion_spread=0,osc_2_spectral_morph_spread=0,
        wavetables=[wave(),wave(),None],modulations=[dict(source=source_name,destination='osc_1_distortion_amount')],
        modulation_1_amount=.35,modulation_1_power=0,modulation_1_bipolar=0,modulation_1_stereo=0,
        modulation_1_ramp_up=-10,modulation_1_ramp_down=-10)
    return doc


def previous(doc,template,**kwargs):
    return convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_phase_controls=frozenset({'osc_1_','osc_2_','osc_3_'}),**kwargs)


def pure_equal(a,b):
    assert a.preset.to_bytes()==b.preset.to_bytes()
    assert [(x.data,x.gain) for x in a.assets]==[(x.data,x.gain) for x in b.assets]
    left=a.report.as_dict();right=b.report.as_dict()
    diagnostics=[n for n in left['notes'] if 'complete phase response could not' in n['message']]
    for note in diagnostics:
        assert any(n['feature']==note['feature'] and n['category']=='control_partial'
                   for n in left['user_notices'])
    left['notes']=[n for n in left['notes'] if 'complete phase response could not' not in n['message']]
    left['user_notices']=[n for n in left['user_notices'] if 'complete phase response could not' not in n['message']]
    left.pop('counts',None);right.pop('counts',None)
    assert left==right


def phase_slot(result):
    return next(s for k,s in result.preset.state.items() if k.startswith('ModSlot') and s.get('destModuleTypeString')=='WTOsc' and s.get('destModuleID')==0 and s.get('destModuleParamName')=='kParamTablePos')


@pytest.mark.parametrize('name',['env_2','macro_control_1','macro_control_4'])
def test_phase_curve_tracks_the_actual_warp_pwl_and_closes_the_bank(template,doc,name):
    source(doc,name);old=previous(doc,template);result=convert(VitalPreset(copy.deepcopy(doc)),template)
    wt=result.preset.state['Oscillator0']['WTOsc0'];bank=np.asarray(wt['embeddedWTData'],dtype=np.float32).reshape(256,2048)
    assert np.array_equal(bank[0],bank[-1]) and np.array_equal(bank[0],old.preset.state['Oscillator0']['WTOsc0']['embeddedWTData'])
    assert plain(wt)['kParamInitialPhase']==0 and plain(wt)['kParamTablePos']==1+255*.5/256
    phase=phase_slot(result);pc=phase['flex'][0]
    assert phase['source'][1]==0 and plain(phase)['kParamMainCurveData']==1
    assert plain(phase)['kParamAmount']==100*255/256 and pc['numPoints']+1==len(pc['xVals'])<=480
    assert 1-pc['yVals'][0]==1
    raw=np.linspace(0,1,200001)
    warp=next(s for k,s in old.preset.state.items() if k.startswith('ModSlot') and s.get('destModuleParamName')=='kParamWarp')
    wc=warp['flex'][0]
    w=plain(warp)['kParamAmount']/100*np.interp(raw,np.asarray(wc['xVals'],dtype=np.float32).astype(float),1-np.asarray(wc['yVals'],dtype=np.float32).astype(float))
    q=np.interp(raw,np.asarray(pc['xVals'],dtype=np.float32).astype(float),1-np.asarray(pc['yVals'],dtype=np.float32).astype(float))
    assert np.max(np.abs(q-(1-8*w**4)))<2.1e-5
    for key,value in old.preset.state.items():
        if key!='Oscillator0' and key!='ModSlot1':assert result.preset.state[key]==value
    assert result.preset.state['Oscillator0']['plainParams']==old.preset.state['Oscillator0']['plainParams']
    assert any(n['feature']=='osc_1_phase_response' for n in result.report.notes)


@pytest.mark.parametrize('change',[
    {'osc_1_distortion_amount':.2}, {'osc_1_distortion_type':10}, {'osc_1_phase':.4},
    {'osc_1_random_phase':.1}, {'osc_1_unison_voices':2}, {'osc_2_unison_voices':2},
    {'osc_1_spectral_morph_type':2}, {'osc_2_spectral_morph_type':2},
    {'osc_2_distortion_type':7}, {'osc_2_on':0}, {'osc_2_phase':.25},
    {'osc_2_transpose':12}, {'osc_2_stereo_spread':.5}, {'osc_1_midi_track':0},
    {'modulation_1_amount':.45}, {'modulation_1_amount':-.35},
    {'modulation_1_bipolar':1}, {'modulation_1_stereo':1},
    {'modulation_1_power':2}, {'modulation_1_ramp_up':-5},
])
def test_unmeasured_signal_or_response_scope_keeps_complete_old_conversion(template,doc,change):
    source(doc);doc['settings'].update(change)
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


@pytest.mark.parametrize('target',['osc_1_wave_frame','osc_1_spectral_morph_amount','osc_1_phase',
    'osc_2_wave_frame','osc_2_transpose','osc_2_on','modulation_1_amount','env_2_attack'])
def test_independent_axes_and_source_dependencies_keep_old_conversion(template,doc,target):
    source(doc);doc['settings']['modulations'].append(dict(source='env_3',destination=target));doc['settings']['modulation_2_amount']=.1
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


@pytest.mark.parametrize('name',['env_1','env_3','lfo_1','velocity'])
def test_unmeasured_sources_keep_old_conversion(template,doc,name):
    source(doc,name);pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


@pytest.mark.parametrize('failure',['id','warp_id','slots','points','source_gain'])
@pytest.mark.parametrize('embed',[False,True])
def test_failure_reconverts_without_partial_assets_slots_or_other_features(template,doc,failure,embed):
    source(doc);s=doc['settings']
    s.update(osc_3_on=1,osc_3_level=.1,osc_3_spectral_morph_type=9,osc_3_spectral_morph_amount=.2,
             wavetables=s['wavetables'][:2]+[wave()],filter_1_on=1,filter_1_model=0,filter_1_style=1,
             filter_1_resonance=0,filter_1_mix=1,filter_1_drive=20)
    options=dict(embed=embed);ids=dict(KNOWN_IDS)
    if failure in ('id','warp_id'):ids.pop(('WTOsc','kParamTablePos' if failure=='id' else 'kParamWarp'));options['destination_ids']=ids
    actual=phase_modulation.apply
    def fail_gain(c,ids):
        if failure=='source_gain':c.source_gain['env_2']=.9
        return actual(c,ids)
    with patch.object(phase_modulation,'MAX_SLOTS',1 if failure=='slots' else 64), \
         patch.object(phase_modulation,'MAX_POINTS',33 if failure=='points' else 480), \
         patch.object(phase_modulation,'apply',side_effect=fail_gain):
        v=VitalPreset(copy.deepcopy(doc));a=convert(v,template,**options);again=convert(v,template,**options)
    b=previous(doc,template,**options)
    pure_equal(a,b)
    assert a.preset.to_bytes()==again.preset.to_bytes() and a.report.as_dict()==again.report.as_dict()


@pytest.mark.parametrize('shared',[False,True])
def test_external_asset_references_have_no_orphans_or_lost_shared_supplier(template,doc,shared):
    source(doc)
    if not shared:
        x=np.arange(2048)/2048;doc['settings']['wavetables'][0]=wave(list(.9*np.sin(4*np.pi*x)))
    result=convert(VitalPreset(copy.deepcopy(doc)),template,embed=False)
    references={body[f'WTOsc{i}']['relativePathToWT'] for i,body in enumerate([result.preset.state[f'Oscillator{i}'] for i in range(3)]) if plain(body).get('kParamEnable') and f'WTOsc{i}' in body and 'relativePathToWT' in body[f'WTOsc{i}']}
    assert references=={a.serum_reference for a in result.assets}
    assert len(result.assets)==2
    assert any(a.frames==256*2048 for a in result.assets)
    assert {record['path'] for record in result.report.assets}=={a.relative_path for a in result.assets}
    assert not any('embeddedWTData' in body.get(f'WTOsc{i}',{}) for i,body in enumerate([result.preset.state[f'Oscillator{i}'] for i in range(3)]))


@pytest.mark.parametrize('amplitude',[.5,2.,4.])
def test_supplier_amplitude_ratio_is_not_inferred_from_normalized_native_pcm(template,doc,amplitude):
    source(doc);x=np.arange(2048)/2048;doc['settings']['wavetables'][1]=wave(list(amplitude*np.sin(2*np.pi*x)))
    doc['settings']['modulation_1_amount']=.1
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


def test_static_amount_without_routes_is_byte_unchanged(template,doc):
    source(doc);doc['settings'].update(osc_1_distortion_amount=.2,modulations=[])
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


def test_carrier_used_as_a_silent_supplier_keeps_old_source_graph(template,doc):
    source(doc);s=doc['settings'];s.update(osc_3_on=1,osc_3_level=0,osc_3_distortion_type=7,osc_3_distortion_amount=.1,wavetables=s['wavetables'][:2]+[wave()])
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


def test_consumer_enabled_by_modulation_keeps_old_source_graph(template,doc):
    source(doc);s=doc['settings'];s.update(osc_3_on=0,osc_3_level=0,osc_3_distortion_type=7,
        osc_3_distortion_amount=.1,wavetables=s['wavetables'][:2]+[wave()])
    s['modulations'].append(dict(source='env_3',destination='osc_3_on'));s['modulation_2_amount']=1.
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


@pytest.mark.parametrize('mode',['master','ring','trajectory','mix'])
def test_pure_phase_failure_keeps_other_complete_control_groups(template,doc,mode):
    source(doc);s=doc['settings']
    if mode=='master':
        s['modulations'].append(dict(source='macro_control_2',destination='volume'))
        s.update(modulation_2_amount=.03,macro_control_2=.5)
    elif mode=='ring':
        s.update(osc_3_on=1,osc_3_level=.1,osc_3_distortion_type=11,osc_3_distortion_amount=.1,
                 osc_3_unison_voices=1,osc_3_stereo_spread=1,wavetables=s['wavetables'][:2]+[wave()])
        s['modulations'].append(dict(source='env_3',destination='osc_3_distortion_amount'));s['modulation_2_amount']=.3
    elif mode=='trajectory':
        s.update(osc_3_on=1,osc_3_level=.1,osc_3_distortion_type=6,osc_3_distortion_amount=.1,
                 wavetables=s['wavetables'][:2]+[wave()])
        s['modulations'] += [dict(source='env_3',destination='osc_3_wave_frame'),dict(source='env_3',destination='osc_3_distortion_amount')]
        s.update(modulation_2_amount=.4,modulation_3_amount=.3)
    else:
        s.update(distortion_on=1,distortion_type=0,distortion_drive=0,distortion_mix=.3)
        s['modulations'].append(dict(source='macro_control_2',destination='distortion_mix'));s.update(modulation_2_amount=.3,macro_control_2=.5)
    old=previous(doc,template)
    with patch.object(phase_modulation,'MAX_POINTS',33):a=convert(VitalPreset(copy.deepcopy(doc)),template)
    pure_equal(a,old)
    full=convert(VitalPreset(copy.deepcopy(doc)),template)
    if mode=='ring':
        # Sharing the supplier with RM changes its asset headroom and hence the FM ratio.
        pure_equal(full,old)
        assert any(r['destination']=='osc_3_distortion_amount' and r['status']=='mapped'
                   for r in old.report.modulations)
        assert plain(old.preset.state['Oscillator2']['WTOsc2'])['kParamWarpMenu']=='kRM_OSC2'
        return
    assert phase_slot(full)
    # Every preexisting slot and noncarrier stage is untouched when the phase group succeeds.
    for k,v in old.preset.state.items():
        if k!='Oscillator0' and not (k.startswith('ModSlot') and not v.get('destModuleParamName')):
            assert full.preset.state[k]==v
    if mode=='master':assert any(n['feature']=='master_volume' and 'complete' in n['message'].lower() for n in old.report.notes)
    elif mode=='trajectory':assert any(n['feature']=='osc_3_trajectory' for n in old.report.notes)
    else:assert any(r.get('atomic_law_destinations') for r in old.report.modulations if r['destination']=='distortion_mix')


def test_multiframe_carrier_keeps_original_selected_frame_and_asset(template,doc):
    source(doc)
    frames=doc['settings']['wavetables'][0]['groups'][0]['components'][0]['keyframes']
    other=copy.deepcopy(wave(list(.9*np.cos(2*np.pi*np.arange(2048)/2048)))['groups'][0]['components'][0]['keyframes'][0]);other['position']=256;frames.append(other)
    pure_equal(convert(VitalPreset(copy.deepcopy(doc)),template),previous(doc,template))


def test_supplier_output_controls_do_not_change_the_prelevel_signal(template,doc):
    source(doc);s=doc['settings'];s['modulations'].append(dict(source='env_3',destination='osc_2_level'));s['modulation_2_amount']=.1
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert phase_slot(result)


@pytest.mark.parametrize('embed',[False,True])
def test_multiple_carriers_reserve_distinct_slots_and_deduplicate_shared_bank_asset(template,doc,embed):
    source(doc);s=doc['settings']
    s.update(osc_1_distortion_type=8,osc_2_distortion_type=8,osc_2_distortion_amount=0,
        osc_2_level=.2,osc_3_on=1,osc_3_level=0,osc_3_distortion_type=0,osc_3_phase=.5,
        osc_3_random_phase=0,osc_3_unison_voices=1,osc_3_unison_detune=0,osc_3_stack_style=0,
        osc_3_stereo_spread=1,osc_3_wave_frame=0,osc_3_transpose=0,osc_3_tune=0,osc_3_midi_track=1,
        wavetables=s['wavetables'][:2]+[wave()])
    s['modulations'].append(dict(source='macro_control_1',destination='osc_2_distortion_amount'));s['modulation_2_amount']=.3
    result=convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    phases=[v for k,v in result.preset.state.items() if k.startswith('ModSlot') and v.get('destModuleParamName')=='kParamTablePos']
    assert len(phases)==2 and {p['destModuleID'] for p in phases}=={0,1}
    if not embed:
        assert len(result.assets)==2 and len(result.report.assets)==2
        assert {r['path'] for r in result.report.assets}=={a.relative_path for a in result.assets}
    old=previous(doc,template,embed=embed)
    with patch.object(phase_modulation,'MAX_SLOTS',2):failed=convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    pure_equal(failed,old)


def test_unknown_native_supplier_fields_keep_the_old_group(template,doc):
    source(doc);actual=phase_modulation.apply
    def inject(c,ids):
        c.s.state['Oscillator1']['WTOsc1']['plainParams']['kParamWarp2']=.2
        return actual(c,ids)
    with patch.object(phase_modulation,'apply',side_effect=inject):a=convert(VitalPreset(copy.deepcopy(doc)),template)
    with patch.object(phase_modulation,'apply',side_effect=lambda c,ids:(c.s.state['Oscillator1']['WTOsc1']['plainParams'].update(kParamWarp2=.2),set())[1]):b=previous(doc,template)
    pure_equal(a,b)


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('embed',[False,True])
def test_nyquist_without_a_phase_quadrature_keeps_the_complete_old_group(template,doc,mixed,embed):
    source(doc);x=np.arange(2048);base=(-1.)**x
    if mixed:base=.5*np.sin(2*np.pi*x/2048)+.001*base
    doc['settings']['wavetables'][0]=wave(list(base))
    old=previous(doc,template,embed=embed)
    result=convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    pure_equal(result,old)
    assert not any(n['feature']=='osc_1_phase_response' and 'follows a closed' in n['message']
                   for n in result.report.notes)
