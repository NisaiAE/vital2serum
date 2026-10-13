import copy
import math

import pytest

from vital2serum import distortion_model as d,modulation
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def moving_mixed(doc,**settings):
    doc['settings'].update(distortion_on=1,distortion_type=0,distortion_drive=-30,
                           distortion_mix=.64,distortion_filter_order=0,modulation_1_amount=1,
                           modulations=[{'source':'env_2','destination':'distortion_drive'}])
    doc['settings'].update(settings)
    return VitalPreset(doc)


@pytest.mark.parametrize('mix',[0,1e-8,.01,.64,.99,1])
def test_mixed_drive_controls_preserve_both_signal_parts_within_headroom(mix):
    for drive in (-40,-30,-15,0,15,30,40):
        wet,level=d.partial_drive_params(drive,mix)
        dry_weight,wet_weight=d.serum_mix(wet)
        output=((2*level)**2)**5
        pre=(2*d.drive_input_level(drive))**2
        assert 0<=level<=1
        assert output*dry_weight*pre==pytest.approx(1-mix,abs=1e-12)
        assert output*wet_weight*d.SHAPER_OUT==pytest.approx(mix,abs=1e-12)


def test_mixed_drive_writes_all_controls_together(template,doc):
    result=convert(moving_mixed(doc),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='mapped'
    group=route['atomic_law_destinations']
    assert len(group)==7
    assert len({item['slot'] for item in group})==7
    assert {(item['module'],item['parameter_id']) for item in group}=={('FXEQ',9),('FXDistortion',1),('FXDistortion',8)}
    entries=result.preset.state['FXRack0']['FX']
    assert len(entries)==6
    effect=next(e for e in entries if 'FXDistortion' in e)
    assert plain(effect['FXDistortion'])['kParamMode']=='kXShaperAsym'
    for drive in (-30,-15,0,15,30):
        wet,level=d.partial_drive_params(drive,.64)
        dry,weight=d.serum_mix(wet);output=((2*level)**2)**5;pre=(2*d.drive_input_level(drive))**2
        for sample in (-.8,-.03,0,.03,.8):
            actual=output*(dry*pre*sample+weight*d.serum_shaper(effect['flex'][0],pre*sample))
            expected=.36*sample+.64*d.vital_shape(0,sample,drive)
            assert actual==pytest.approx(expected,abs=1e-4)


def assert_static_fallback(result,expected):
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='omitted'
    assert not route.get('atomic_law_destinations')
    fx=result.preset.state['FXRack0']['FX']
    actual=next(e for e in fx if 'FXDistortion' in e)
    assert 'flex' not in actual
    assert actual==next(e for e in expected.preset.state['FXRack0']['FX'] if 'FXDistortion' in e)
    assert all(plain(e['FXEQ'])['kParamLevelOut']==.5 for e in fx if 'FXEQ' in e)


def test_mixed_drive_missing_id_restores_static_curve_and_gains(template,doc,monkeypatch):
    expected_doc=copy.deepcopy(doc);expected_doc['settings'].update(distortion_on=1,distortion_type=0,
        distortion_drive=-30,distortion_mix=.64,distortion_filter_order=0,modulations=[])
    expected=convert(VitalPreset(expected_doc),template)
    monkeypatch.delitem(modulation.KNOWN_IDS,('FXDistortion','kParamLevelOut'))
    assert_static_fallback(convert(moving_mixed(doc),template),expected)


@pytest.mark.parametrize('users',[57,58])
def test_mixed_drive_respects_remaining_slots_atomically(template,doc,users):
    vital=moving_mixed(doc);vital.settings['modulations']=[{'source':'lfo_1','destination':'osc_1_pan'} for _ in range(users)]+[{'source':'env_2','destination':'distortion_drive'}]
    for i in range(1,users+1):vital.settings[f'modulation_{i}_amount']=.01
    vital.settings[f'modulation_{users+1}_amount']=1
    result=convert(vital,template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert all(m['status']=='mapped' for m in result.report.modulations if m['destination']=='osc_1_pan')
    assert route['status']==('mapped' if users==57 else 'omitted')
    written=[name for name,value in result.preset.state.items() if name.startswith('ModSlot') and value.get('destModuleTypeString')]
    assert len(written)==(64 if users==57 else 58)
    if users==58:
        effect=next(e for e in result.preset.state['FXRack0']['FX'] if 'FXDistortion' in e)
        assert 'flex' not in effect
        assert all(plain(e['FXEQ'])['kParamLevelOut']==.5 for e in result.preset.state['FXRack0']['FX'] if 'FXEQ' in e)


@pytest.mark.parametrize('settings',[
    {'distortion_type':1}, {'distortion_filter_order':1}, {'distortion_filter_order':2},
    {'modulation_1_bipolar':1}, {'modulation_1_stereo':1},
    {'chorus_on':1,'compressor_on':1,'delay_on':1,'eq_on':1,'reverb_on':1},
    {'modulation_2_amount':.2,'modulations':[{'source':'env_2','destination':'distortion_drive'},
                                           {'source':'env_3','destination':'distortion_mix'}]},
])
def test_mixed_drive_excludes_unverified_configs(template,doc,settings):
    result=convert(moving_mixed(doc,**settings),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='omitted'


def test_mixed_drive_reserves_trailing_stereo_utility(template,doc):
    vital=moving_mixed(doc,compressor_on=1,eq_on=1,filter_fx_on=1,
                      eq_low_gain=0,eq_band_gain=1,eq_high_gain=1,effect_chain_order=3024)
    base=convert(vital,template)
    route=next(m for m in base.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='mapped'
    assert len(base.preset.state['FXRack0']['FX'])==9
    vital.settings.update(stereo_routing=.5,stereo_mode=0)
    result=convert(vital,template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='omitted' and not route.get('atomic_law_destinations')
    assert len(result.preset.state['FXRack0']['FX'])==5
    assert 'FXUtils' in result.preset.state['FXRack0']['FX'][-1]


def test_mixed_drive_reserves_both_trailing_eq_entries(template,doc):
    vital=moving_mixed(doc,compressor_on=1,eq_on=1,filter_fx_on=1,
                      eq_low_gain=1,eq_band_gain=1,eq_high_gain=1,effect_chain_order=0)
    result=convert(vital,template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_drive')
    assert route['status']=='omitted' and not route.get('atomic_law_destinations')
    entries=result.preset.state['FXRack0']['FX']
    assert len(entries)==5 and sum('FXEQ' in entry for entry in entries)==2
