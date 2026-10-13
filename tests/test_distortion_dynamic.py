import copy
import math

import pytest

from vital2serum import distortion_model as d
from vital2serum import modulation
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def moving_clip(doc,kind=1,**settings):
    doc['settings'].update(distortion_on=1,distortion_type=kind,distortion_mix=1,
                           distortion_filter_order=0,distortion_drive=-30,
                           modulation_1_amount=1,
                           modulations=[{'source':'env_2','destination':'distortion_drive'}])
    doc['settings'].update(settings)
    return VitalPreset(doc)


@pytest.mark.parametrize('kind',[d.SOFT_CLIP,d.HARD_CLIP])
def test_dynamic_clip_has_one_verified_input_gain_destination(template,doc,kind):
    result=convert(moving_clip(doc,kind),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='mapped'
    assert (route['serum_module'],route['serum_parameter'],route['serum_parameter_id'])==('FXEQ','kParamLevelOut',9)
    entry=result.preset.state['FXRack0']['FX'][route['serum_module_id']]
    assert plain(entry['FXEQ'])['kParamLevelOut']==pytest.approx(d.drive_input_level(-30))
    assert len(route['serum_slots'])==1


@pytest.mark.parametrize('kind',[d.SOFT_CLIP,d.HARD_CLIP])
def test_input_gain_and_fixed_clip_preserve_the_full_drive_range(template,doc,kind):
    result=convert(moving_clip(doc,kind),template)
    entries=result.preset.state['FXRack0']['FX']
    eq=[plain(e['FXEQ']) for e in entries if 'FXEQ' in e]
    effect=next(e for e in entries if 'FXDistortion' in e)
    params=plain(effect['FXDistortion'])
    output=(2*params['kParamLevelOut'])**2
    fixed=math.prod((2*p['kParamLevelOut'])**2 for p in eq[1:])
    for drive in (-30,-15,0,15,30):
        gain=(2*d.drive_input_level(drive))**2*fixed
        for x in (-2,-.5,-.03,0,.03,.5,2):
            if kind==d.SOFT_CLIP:
                actual=d.serum_shaper(effect['flex'][0],gain*x)*output
            else:
                u=gain*x*(.5+2*(params['kParamDrive']/100)**3)
                actual=min(.8278,max(-.8278,u))*output
            assert actual==pytest.approx(d.vital_shape(kind,x,drive),abs=1e-4)
    if kind==d.SOFT_CLIP:
        assert len(effect['flex'][0]['xVals'])<=d.SHAPER_ENTRIES
        assert any(n['feature']=='distortion_type' and '12 Hz' in n['message'] for n in result.report.notes)


@pytest.mark.parametrize('settings',[
    {'distortion_mix':.7},
    {'distortion_filter_order':1},
    {'distortion_filter_order':2,
     'modulations':[{'source':'env_2','destination':'distortion_drive'},
                    {'source':'env_3','destination':'distortion_filter_cutoff'}],'modulation_2_amount':.3},
    {'modulation_1_bipolar':1},
    {'modulation_1_stereo':1},
    {'modulations':[{'source':'env_2','destination':'distortion_drive'},
                    {'source':'lfo_1','destination':'distortion_drive'}],'modulation_2_amount':.3},
    {'modulations':[{'source':'env_2','destination':'distortion_drive'},
                    {'source':'lfo_1','destination':'modulation_1_amount'}],'modulation_2_amount':.3},
])
def test_unverified_dynamic_clip_paths_remain_omitted(template,doc,settings):
    result=convert(moving_clip(doc,**settings),template)
    routes=[m for m in result.report.modulations if m.get('destination')=='distortion_drive']
    assert routes and all(m['status']=='omitted' for m in routes)


def test_fixed_auxiliary_drive_is_folded_at_the_stored_macro(template,doc):
    vital=moving_clip(doc,0,modulation_1_amount=0,macro_control_1=.75,
        modulation_2_amount=.5,
        modulations=[{'source':'env_2','destination':'distortion_drive'},
                     {'source':'macro_control_1','destination':'modulation_1_amount'}])
    result=convert(vital,template)
    slot=next(m for name,m in result.preset.state.items() if name.startswith('ModSlot') and m.get('destModuleTypeString')=='FXEQ')
    assert slot['source'][1]==0
    assert plain(slot)['kParamAmount']==pytest.approx(100*(d.drive_input_level(15)-d.drive_input_level(-30)))
    assert any(n['kind']=='approximation' and n.get('feature')=='modulation_1_static_aux' for n in result.report.notes)


@pytest.mark.parametrize('kind',[d.LINEAR_FOLD,d.SINE_FOLD])
def test_dynamic_fold_uses_a_fixed_curve_with_a_moving_input_gain(template,doc,kind):
    result=convert(moving_clip(doc,kind),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='mapped'
    entries=result.preset.state['FXRack0']['FX']
    effect=next(e for e in entries if 'FXDistortion' in e)
    output=(2*plain(effect['FXDistortion'])['kParamLevelOut'])**2
    assert len(effect['flex'][0]['xVals'])<=d.SHAPER_ENTRIES
    for drive in (-30,-15,0,15,30):
        gain=(2*d.drive_input_level(drive))**2
        for x in (-1.65,-.5,-.031,0,.031,.5,1.65):
            actual=d.serum_shaper(effect['flex'][0],gain*x)*output
            assert actual==pytest.approx(d.vital_shape(kind,x,drive),abs=.02 if kind==d.SINE_FOLD else 1e-10)
    assert not any('a modulated drive cannot move it' in n['message'] for n in result.report.notes)


def test_sine_fold_retains_stored_drive_when_input_span_reduces_resolution(template,doc):
    doc['settings']['wavetables'][1]=copy.deepcopy(doc['settings']['wavetables'][0])
    result=convert(moving_clip(doc,d.SINE_FOLD,osc_1_level=1,osc_2_level=1,osc_2_on=1),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='omitted'
    assert any(n['feature']=='distortion_drive' and 'too few shaper points' in n['message'] for n in result.report.notes)


@pytest.mark.parametrize('settings',[{'distortion_mix':.7},{'distortion_filter_order':2}])
def test_unverified_dynamic_fold_mix_and_filter_paths_remain_omitted(template,doc,settings):
    result=convert(moving_clip(doc,d.LINEAR_FOLD,**settings),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='omitted'


@pytest.mark.parametrize('bipolar',[0,1])
def test_dynamic_downsample_routes_the_native_hold_control(template,doc,bipolar):
    result=convert(moving_clip(doc,d.DOWN_SAMPLE,distortion_drive=0,
                              distortion_mix=.3,modulation_1_amount=.5,modulation_1_bipolar=bipolar),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='mapped'
    assert (route['serum_module'],route['serum_parameter'],route['serum_parameter_id'])==('FXDistortion','kParamDrive',2)
    entries=result.preset.state['FXRack0']['FX']
    assert plain(entries[route['serum_module_id']]['FXDistortion'])['kParamDrive']==pytest.approx(d.downsample_drive(0))
    assert any(n['feature']=='distortion_drive' and '44.1 kHz' in n['message'] for n in result.report.notes)


@pytest.mark.parametrize('settings',[
    {'modulation_1_stereo':1},
    {'modulations':[{'source':'env_2','destination':'distortion_drive'},
                    {'source':'lfo_1','destination':'distortion_drive'}],'modulation_2_amount':.3},
    {'modulations':[{'source':'env_2','destination':'distortion_drive'},
                    {'source':'lfo_1','destination':'modulation_1_amount'}],'modulation_2_amount':.3},
])
def test_downsample_retains_stored_hold_for_unverified_drive_routes(template,doc,settings):
    result=convert(moving_clip(doc,d.DOWN_SAMPLE,distortion_drive=5,**settings),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='omitted'
    effect=next(e for e in result.preset.state['FXRack0']['FX'] if 'FXDistortion' in e)
    assert plain(effect['FXDistortion'])['kParamDrive']==pytest.approx(d.downsample_drive(5))


def test_full_modulation_rack_retains_the_stored_clip_drive(template,doc):
    mods=[{'source':'env_2','destination':'osc_1_pan'} for _ in range(63)]
    mods.append({'source':'env_2','destination':'distortion_drive'})
    settings={f'modulation_{i}_amount':.1 for i in range(1,64)}
    settings.update(modulations=mods,modulation_64_amount=1,velocity_track=1)
    result=convert(moving_clip(doc,**settings),template)
    route=next(m for m in result.report.modulations if m.get('destination')=='distortion_drive')
    assert route['status']=='omitted'
    assert len([name for name in result.preset.state if name.startswith('ModSlot')])==64
    input_eq=next(e['FXEQ'] for e in result.preset.state['FXRack0']['FX'] if 'FXEQ' in e)
    assert plain(input_eq)['kParamLevelOut']==pytest.approx(d.drive_input_level(-30))


def auxiliary_clip(doc,kind=1,**settings):
    return moving_clip(doc,kind,modulation_1_amount=0,modulation_2_amount=1,
        modulations=[{'source':'lfo_1','destination':'distortion_drive'},
                     {'source':'lfo_2','destination':'modulation_1_amount'}],**settings)


@pytest.mark.parametrize('kind',[d.SOFT_CLIP,d.HARD_CLIP])
def test_moving_auxiliary_combines_drive_before_its_nonlinear_scale(template,doc,monkeypatch,kind):
    result=convert(auxiliary_clip(doc,kind),template)
    route=next(m for m in result.report.modulations if m.get('destination')=='distortion_drive')
    bus=route['complete_control_bus']
    input_slot=result.preset.state[f"ModSlot{bus['input_slot']}"]
    output=result.preset.state[f"ModSlot{bus['output_slot']}"]
    assert input_slot['destModuleTypeString']=='Macro'
    assert input_slot['source']==[6,7]
    assert input_slot['plainParams']['kParamAuxCurveData']==1
    assert output['source']==[25+bus['macro'],0]
    assert output['destModuleTypeString']=='FXEQ'
    assert len(route['serum_slots'])==1
    assert plain(result.preset.state[f"Macro{bus['macro']}"])['kParamValue']==0


@pytest.mark.parametrize('failure',['missing_id','no_macro','full_slots'])
@pytest.mark.parametrize('kind',[d.SOFT_CLIP,d.HARD_CLIP])
def test_incomplete_auxiliary_group_restores_the_stored_clip(template,doc,monkeypatch,kind,failure):
    v=auxiliary_clip(doc,kind)
    ids=dict(modulation.KNOWN_IDS)
    if failure=='missing_id': ids.pop(('FXEQ','kParamLevelOut'))
    if failure=='no_macro': monkeypatch.setattr(modulation,'BUS_MACROS',())
    if failure=='full_slots':
        routes=[{'source':'env_2','destination':'osc_1_pan'} for _ in range(62)]
        routes+=v.settings['modulations']
        routes[-1]['destination']='modulation_63_amount'
        v.settings['modulations']=routes
        v.settings.update({f'modulation_{i}_amount':.1 for i in range(1,63)})
        v.settings.update(modulation_63_amount=0,modulation_64_amount=1,velocity_track=1)
    result=convert(v,template,destination_ids=ids)
    control=copy.deepcopy(v.doc);control['settings']['modulations']=[]
    expected=convert(VitalPreset(control),template)
    actual_entry=next(e for e in result.preset.state['FXRack0']['FX'] if 'FXDistortion' in e)
    expected_entry=next(e for e in expected.preset.state['FXRack0']['FX'] if 'FXDistortion' in e)
    assert actual_entry==expected_entry
    assert all(plain(e['FXEQ'])['kParamLevelOut']==.5 for e in result.preset.state['FXRack0']['FX'] if 'FXEQ' in e)
    route=next(m for m in result.report.modulations if m.get('destination')=='distortion_drive')
    assert route['status']=='omitted'
    assert 'complete_control_bus' not in route
    assert all(not (name.startswith('ModSlot') and m.get('destModuleTypeString')=='Macro') for name,m in result.preset.state.items())


@pytest.mark.parametrize('kind',[d.SOFT_CLIP,d.HARD_CLIP])
def test_fixed_post_filter_keeps_drive_before_the_filter(template,doc,kind):
    result=convert(moving_clip(doc,kind,distortion_filter_order=2,distortion_filter_cutoff=72),template)
    drive=next(m for m in result.report.modulations if m.get('destination')=='distortion_drive')
    assert drive['status']=='mapped'
    entries=result.preset.state['FXRack0']['FX']
    effect=next(e for e in entries if 'FXDistortion' in e)
    assert plain(effect['FXDistortion'])['kParamPrePost']==2
    assert any(n['feature']=='distortion_filter' and n['kind']=='approximation' for n in result.report.notes)


def test_auxiliary_ramp_time_is_reported(template,doc):
    vital=auxiliary_clip(doc)
    vital.settings['modulation_2_ramp_up']=0
    result=convert(vital,template)
    assert any(n['feature']=='modulation_2_ramp' and n['kind']=='unsupported' for n in result.report.notes)


def test_auxiliary_clamp_precedes_main_source_multiplication(template,doc):
    result=convert(auxiliary_clip(doc),template)
    route=next(m for m in result.report.modulations if m.get('destination')=='distortion_drive')
    bus=route['complete_control_bus'];curve=result.preset.state[f"ModSlot{bus['input_slot']}"]['flex'][1]
    for x,expected in [(0,0),(.25,.5),(.5,1),(.75,1),(1,1)]:
        value=(d.serum_shaper(curve,d.SHAPER_IN*(2*x-1))/d.SHAPER_OUT+1)/2
        assert value==pytest.approx(expected)


def test_missing_helper_ids_restore_high_drive_gain_stages(template,doc):
    vital=auxiliary_clip(doc)
    vital.settings['distortion_drive']=25
    ids=dict(modulation.KNOWN_IDS);ids.pop(('Macro','kParamValue'))
    result=convert(vital,template,destination_ids=ids)
    control=copy.deepcopy(vital.doc);control['settings']['modulations']=[]
    expected=convert(VitalPreset(control),template)
    def stage_gain(r):
        return math.prod((2*plain(e['FXEQ'])['kParamLevelOut'])**2 for e in r.preset.state['FXRack0']['FX'] if 'FXEQ' in e)
    assert stage_gain(result)==pytest.approx(stage_gain(expected))
    actual=next(e for e in result.preset.state['FXRack0']['FX'] if 'FXDistortion' in e)
    original=next(e for e in expected.preset.state['FXRack0']['FX'] if 'FXDistortion' in e)
    assert actual==original
