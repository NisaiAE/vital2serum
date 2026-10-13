import copy
import math
import numpy as np
import pytest
from vital2serum import modulation
from vital2serum import fm
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.trajectory import correlated_controls, render_trajectory, source_grid, position_law
from vital2serum.vital import VitalPreset
from vital2serum.wavetable import FRAME_SIZE, phase_warp, spectral_morph


def setup(doc, source='env_2', kind=4):
    s=doc['settings']
    s.update(osc_1_distortion_type=kind,osc_1_distortion_amount=.2,osc_1_wave_frame=10,
             modulation_1_amount=.75,modulation_2_amount=.6,
             modulations=[{'source':source,'destination':'osc_1_wave_frame'},
                          {'source':source,'destination':'osc_1_distortion_amount'}])
    return s


def test_correlated_shapes_are_summed_before_clamp(doc):
    s=setup(doc)
    s.update(modulation_2_bipolar=1,modulation_2_power=2,
             modulation_3_amount=-.8,modulation_4_amount=-.25)
    s['modulations'] += [
        {'source':'env_2','destination':'osc_1_distortion_amount',
         'line_mapping':{'points':[0,1,.5,0,1,1],'powers':[0,0]}},
        {'source':'env_2','destination':'osc_1_wave_frame'}]
    plan=correlated_controls(VitalPreset(doc),'osc_1_')
    assert plan is not None
    for raw in [0,.1,.5,.7,1]:
        warp=.2+sum(m.amount*m.value(raw) for m in plan.routes if m.target.endswith('distortion_amount'))
        values=plan.values(raw)
        assert values['osc_1_wave_frame']==pytest.approx(10+128*raw)
        assert values['osc_1_distortion_amount']==pytest.approx(max(0,min(1,warp)))


@pytest.mark.parametrize('kind',[1,2,3,4,5,6])
def test_trajectory_frames_equal_direct_warp_evaluation(doc,kind):
    setup(doc,kind=kind)
    plan=correlated_controls(VitalPreset(doc),'osc_1_')
    phase=2*np.pi*np.arange(FRAME_SIZE)/FRAME_SIZE
    frames=np.stack([np.sin(phase),np.sin(phase)+.2*np.sin(3*phase)])
    out=render_trajectory(frames,plan,count=7)
    for row,raw in zip(out,source_grid(7)):
        values=plan.values(float(raw));frame=values['osc_1_wave_frame'];warp=values['osc_1_distortion_amount']
        expected=phase_warp(frames[min(int(frame),1):min(int(frame),1)+1],kind,warp,plan.warp_phase)[0]
        assert np.array_equal(row,expected)


@pytest.mark.parametrize('reason',['independent','aux','ramp','phase','morph_phase','unsupported_morph','stereo','spread','moving_spread','warp_type','morph_type'])
def test_trajectory_excludes_unrepresented_axes(doc,reason):
    s=setup(doc)
    if reason=='independent':s['modulations'][1]['source']='lfo_1'
    elif reason=='aux':
        s['modulations'].append({'source':'macro_control_1','destination':'modulation_2_amount'})
        s['modulation_3_amount']=.1
    elif reason=='ramp':s['modulation_1_ramp_up']=-2
    elif reason=='phase':
        s['modulations'].append({'source':'env_2','destination':'osc_1_distortion_phase'})
        s['modulation_3_amount']=.1
    elif reason=='morph_phase':
        s['modulations'].append({'source':'env_2','destination':'osc_1_spectral_morph_phase'})
        s.update(modulation_3_amount=.1,osc_1_spectral_morph_type=7)
    elif reason=='unsupported_morph':s['osc_1_spectral_morph_type']=10
    elif reason=='stereo':s['modulation_1_stereo']=1
    elif reason=='spread':s.update(osc_1_unison_voices=3,osc_1_frame_spread=.1)
    elif reason=='moving_spread':
        s.update(osc_1_unison_voices=3,modulation_3_amount=.1)
        s['modulations'].append({'source':'env_2','destination':'osc_1_frame_spread'})
    elif reason in ('warp_type','morph_type'):
        s['modulations'].append({'source':'env_2','destination':'osc_1_'+('distortion_type' if reason=='warp_type' else 'spectral_morph_type')})
        s['modulation_3_amount']=.1
    assert correlated_controls(VitalPreset(doc),'osc_1_') is None


def test_joint_routes_share_one_slot_and_macro_keeps_moving(template,doc):
    setup(doc,source='macro_control_1')
    doc['settings']['macro_control_1']=.4
    v=VitalPreset(doc);result=convert(v,template)
    routes=[m for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')]
    table=result.preset.state['Oscillator0']['WTOsc0']
    assert v.folded==set()
    assert len(routes)==1 and routes[0]['source']==[25,0]
    assert routes[0]['destModuleParamName']=='kParamTablePos'
    assert plain(table)['kParamTablePos']==position_law()[0]
    assert plain(routes[0])['kParamAmount']==position_law()[1]
    assert 'kParamWarpMenu' not in plain(table)
    assert plain(result.preset.state['Macro0'])['kParamValue']==40
    records=[m for m in result.report.modulations if m.get('joint_trajectory')]
    assert [m['vital_slot'] for m in records]==[1,2]
    assert all(m['serum_slots']==[0] for m in records)


@pytest.mark.parametrize('failure',['id','slots','source_gain'])
def test_trajectory_failure_restores_complete_conversion(template,doc,monkeypatch,failure):
    s=setup(doc,source='lfo_1')
    options={}
    if failure=='id':
        ids=dict(modulation.KNOWN_IDS);del ids[('WTOsc','kParamTablePos')]
        options['destination_ids']=ids
    elif failure=='slots':
        s['modulations'].insert(0,{'source':'env_2','destination':'osc_1_pan'})
        s.update(modulation_1_amount=.1,modulation_2_amount=.75,modulation_3_amount=.6)
        monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    else:
        s.update(lfo_1_sync=0,lfo_1_frequency=6,lfo_1_smooth_mode=1,lfo_1_smooth_time=-8)
    result=convert(VitalPreset(copy.deepcopy(doc)),template,**options)
    ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_trajectories=frozenset({'osc_1_'}),**options)
    assert result.preset.state==ref.preset.state
    assert [(a.data,a.gain) for a in result.assets]==[(a.data,a.gain) for a in ref.assets]
    assert result.report.modulations==ref.report.modulations
    assert [n for n in result.report.notes if 'original conversion is retained' not in n['message']]==ref.report.notes
    assert any('original conversion is retained' in n['message'] for n in result.report.notes)
    assert not any(m.get('joint_trajectory') for m in result.report.modulations)


@pytest.mark.parametrize('oscillators',[2,3])
def test_partial_failure_preserves_other_trajectories_and_reuse(template,doc,oscillators):
    s=setup(doc)
    s['wavetables']=[copy.deepcopy(s['wavetables'][0]) for _ in range(3)]
    for i in range(2,oscillators+1):
        source='lfo_1' if i==2 else 'env_3'
        s.update({f'osc_{i}_on':1,f'osc_{i}_distortion_type':4,f'osc_{i}_distortion_amount':.2})
        for target,amount in [('wave_frame',.4),('distortion_amount',.5)]:
            s['modulations'].append({'source':source,'destination':f'osc_{i}_{target}'})
            s[f'modulation_{len(s["modulations"])}_amount']=amount
    s.update(lfo_1_sync=0,lfo_1_frequency=6,lfo_1_smooth_mode=1,lfo_1_smooth_time=-8)
    v=VitalPreset(copy.deepcopy(doc));first=convert(v,template);second=convert(v,template)
    ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_trajectories=frozenset({'osc_2_'}))
    assert first.preset.state==second.preset.state==ref.preset.state
    assert first.report.modulations==second.report.modulations==ref.report.modulations
    assert first.report.notes==second.report.notes
    assert [n for n in first.report.notes if 'original conversion is retained' not in n['message']]==ref.report.notes
    assert {m['joint_trajectory'] for m in first.report.modulations if m.get('joint_trajectory')}==(
        {'osc_1_'} if oscillators==2 else {'osc_1_','osc_3_'})


def test_candidate_lookahead_does_not_consume_unsupported_settings(doc):
    s=setup(doc);s.update(osc_1_unison_voices=3,osc_1_frame_spread=.1)
    v=VitalPreset(doc);v.used.add('test')
    assert correlated_controls(v,'osc_1_') is None
    assert v.used=={'test'}


@pytest.mark.parametrize('name',[42,{'label':'table'},['table']])
def test_trajectory_table_name_uses_text_only(template,doc,name):
    setup(doc)
    doc['settings']['wavetables'][0]['name']=name
    result=convert(VitalPreset(doc),template)
    assert result.preset.state['Oscillator0']['WTOsc0']['tableDisplayName']=='Vital table'


@pytest.mark.parametrize('count',[1,7,256])
def test_position_law_selects_nearest_endpoint_grid(count):
    grid=source_grid(count);base,depth=position_law(count)
    assert grid[0]==0
    if count>1:assert grid[-1]==1
    for raw in [0,.001,.25,.5,.75,.999,1]:
        bucket=min(count-1,int(count*((base-1)/255+depth*raw/100)))
        nearest=(count-1)*raw
        expected=min(count-1,int(nearest+.5))
        assert bucket==expected or abs(nearest-math.floor(nearest)-.5)<1e-12
        if count>1:assert abs(grid[bucket]-raw)<=.5/(count-1)+1e-12
        else:assert bucket==0


@pytest.mark.parametrize('morph',[3,4,7,8])
@pytest.mark.parametrize('moving_warp',[False,True])
def test_frame_morph_and_three_axes_match_direct_transform_order(doc,morph,moving_warp):
    s=setup(doc)
    s.update(osc_1_distortion_type=4 if moving_warp else 0,osc_1_spectral_morph_type=morph,
             osc_1_spectral_morph_amount=.15)
    s['modulations'][1]['destination']='osc_1_spectral_morph_amount'
    if moving_warp:
        s['modulations'].append({'source':'env_2','destination':'osc_1_distortion_amount'})
        s['modulation_3_amount']=.5
    plan=correlated_controls(VitalPreset(doc),'osc_1_');assert plan is not None
    phase=2*np.pi*np.arange(FRAME_SIZE)/FRAME_SIZE
    frames=np.stack([np.sin(phase)+(j/255)*.3*np.sin(5*phase) for j in range(256)])
    result=render_trajectory(frames,plan,count=7)
    for row,raw in zip(result,source_grid(7)):
        index=min(int(10+192*raw),255)
        expected=spectral_morph(frames[index:index+1],morph,.15+.6*raw)
        if moving_warp:expected=phase_warp(expected,4,.2+.5*raw,.5)
        assert np.array_equal(row,expected[0])


def test_three_axes_sum_signed_shapes_and_clamp_at_boundaries(doc):
    s=setup(doc,source='macro_control_1')
    s.update(osc_1_wave_frame=80,osc_1_distortion_amount=.6,osc_1_spectral_morph_type=7,osc_1_spectral_morph_amount=.7,
             modulation_1_amount=1,modulation_2_amount=-1,modulation_2_bipolar=1,modulation_2_power=2,
             modulation_3_amount=-1,modulation_4_amount=-.25,modulation_5_amount=-.5,modulation_6_amount=1)
    s['modulations'] += [
        {'source':'macro_control_1','destination':'osc_1_spectral_morph_amount'},
        {'source':'macro_control_1','destination':'osc_1_wave_frame'},
        {'source':'macro_control_1','destination':'osc_1_distortion_amount'},
        {'source':'macro_control_1','destination':'osc_1_spectral_morph_amount',
         'line_mapping':{'points':[0,1,.5,0,1,1],'powers':[0,0]}}]
    plan=correlated_controls(VitalPreset(doc),'osc_1_');assert plan is not None
    probes={0,.5,1,.3,.9,1.7/3,1/256,(256-80)/192}
    probes.update((j-80)/192 for j in [80,81,127,255] if 0<=(j-80)/192<=1)
    probes={min(1,max(0,x+d)) for x in probes for d in [-1e-9,0,1e-9]}
    clamp=lambda x,hi=1:max(0,min(hi,x))
    for u in probes:
        side=2*u-1
        bent=.5+.5*math.copysign(math.expm1(-2*abs(side))/math.expm1(-2),side)
        triangle=1-abs(2*u-1)
        expected={'osc_1_wave_frame':clamp(80+192*u,256),
                  'osc_1_distortion_amount':clamp(.6-(bent-.5)-.5*u),
                  'osc_1_spectral_morph_amount':clamp(.7-u+triangle)}
        assert plan.values(u)==pytest.approx(expected,abs=1e-12)


@pytest.mark.parametrize('morph',[3,4,7,8])
def test_fixed_frame_warp_morph_matches_direct_transform_order(doc,morph):
    s=setup(doc)
    s.update(osc_1_wave_frame=71,osc_1_spectral_morph_type=morph,osc_1_spectral_morph_amount=.15)
    s['modulations'][0]['destination']='osc_1_spectral_morph_amount'
    plan=correlated_controls(VitalPreset(doc),'osc_1_');assert plan is not None
    phase=2*np.pi*np.arange(FRAME_SIZE)/FRAME_SIZE
    frames=np.stack([np.sin(phase)+(j/255)*.3*np.sin(5*phase) for j in range(256)])
    for row,u in zip(render_trajectory(frames,plan,count=7),source_grid(7)):
        expected=spectral_morph(frames[71:72],morph,min(1,.15+.75*u))
        expected=phase_warp(expected,4,.2+.6*u,.5)
        assert np.array_equal(row,expected[0])


@pytest.mark.parametrize('phase',[0,.5-2e-6,.5+2e-6,1])
def test_non_neutral_fixed_morph_phase_is_not_claimed(template,doc,phase):
    s=setup(doc);s.update(osc_1_spectral_morph_type=7,osc_1_spectral_morph_phase=phase)
    assert correlated_controls(VitalPreset(doc),'osc_1_') is None
    result=convert(VitalPreset(doc),template)
    assert not any(m.get('joint_trajectory') for m in result.report.modulations)
    assert any(n['kind']=='unsupported' and n['feature']=='osc_1_spectral_morph_phase' for n in result.report.notes)


@pytest.mark.parametrize('spread',['frame_spread','distortion_spread','spectral_morph_spread'])
@pytest.mark.parametrize('moving',[False,True])
def test_dynamic_voice_count_excludes_fixed_or_moving_spread(doc,spread,moving):
    s=setup(doc);s.update(osc_1_unison_voices=1,modulation_3_amount=.5)
    s['modulations'].append({'source':'env_3','destination':'osc_1_unison_voices'})
    if moving:
        s['modulations'].append({'source':'env_2','destination':'osc_1_'+spread})
        s['modulation_4_amount']=.1
    else:s['osc_1_'+spread]=.1
    assert correlated_controls(VitalPreset(doc),'osc_1_') is None


def test_dynamic_voice_count_with_neutral_spreads_keeps_trajectory(doc):
    s=setup(doc);s.update(osc_1_unison_voices=1,modulation_3_amount=.5)
    s['modulations'].append({'source':'env_3','destination':'osc_1_unison_voices'})
    assert correlated_controls(VitalPreset(doc),'osc_1_') is not None


@pytest.mark.parametrize('source',['env_2','macro_control_1','lfo_1'])
def test_shared_source_voice_structure_is_not_projected_onto_table(doc,source):
    s=setup(doc,source=source);s.update(osc_1_unison_voices=1,modulation_3_amount=.5)
    s['modulations'].append({'source':source,'destination':'osc_1_unison_voices'})
    assert correlated_controls(VitalPreset(doc),'osc_1_') is None


@pytest.mark.parametrize('condition',['inactive','bypassed','other_oscillator'])
def test_unrelated_voice_route_keeps_trajectory(doc,condition):
    s=setup(doc);s.update(osc_1_unison_voices=1,modulation_3_amount=.5)
    route={'source':'env_2','destination':'osc_1_unison_voices'}
    s['modulations'].append(route)
    if condition=='inactive':s['modulation_3_amount']=0
    elif condition=='bypassed':s['modulation_3_bypass']=1
    else:route['destination']='osc_2_unison_voices'
    assert correlated_controls(VitalPreset(doc),'osc_1_') is not None


def test_signed_source_range_is_not_projected_onto_unit_grid(doc):
    setup(doc,source='pitch_wheel')
    assert correlated_controls(VitalPreset(doc),'osc_1_') is None


def test_trajectory_ring_modulator_keeps_shared_table_headroom(template,doc):
    s=setup(doc)
    s['wavetables']=[copy.deepcopy(s['wavetables'][0]) for _ in range(3)]
    s.update(osc_2_on=1,osc_2_distortion_type=10,osc_2_distortion_amount=.25,modulation_3_amount=.5)
    s['modulations'].append({'source':'env_3','destination':'osc_2_distortion_amount'})
    result=convert(VitalPreset(doc),template)
    note=next(n for n in result.report.notes if n['feature']=='osc_1_trajectory')
    assert note['table_gain']==pytest.approx(fm.SERUM_RM_GAIN)
    wt=result.preset.state['Oscillator0']['WTOsc0']
    assert max(abs(x) for x in wt['embeddedWTData'])<=1/fm.SERUM_RM_GAIN+1e-6


def test_trajectory_source_curve_only_applies_existing_source_law(template,doc):
    s=setup(doc,source='env_1')
    s['modulations'][0]['line_mapping']={'points':[0,1,.5,0,1,1],'powers':[1,-1]}
    s.update(modulation_1_power=2,modulation_2_bipolar=1)
    result=convert(VitalPreset(doc),template)
    slot=next(x for k,x in result.preset.state.items() if k.startswith('ModSlot') and x.get('destModuleParamName'))
    raw=modulation.Modulation(1,'env_1','osc_1_wave_frame',1,False,0,False)
    assert slot['flex'][0]==modulation._trajectory_source_curve(raw)
