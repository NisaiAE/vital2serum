import math

import pytest

from vital2serum import distortion_model as d
from vital2serum.mapping import convert
from vital2serum.serum import SerumPreset, plain
from vital2serum.vital import VitalPreset


def rack(result, name):
    return [entry[name] for entry in result.preset.state['FXRack0']['FX'] if name in entry]


@pytest.mark.parametrize('kind', [d.LINEAR_FOLD, d.SINE_FOLD, d.BIT_CRUSH])
@pytest.mark.parametrize('drive', [-30, -12, 0, 12, 30])
def test_transfer_curve_preserves_both_polarities(kind, drive):
    curve, top = d.shaper_curve(kind, drive)
    assert len(curve['xVals']) <= d.SHAPER_ENTRIES
    assert curve['xVals'][0] == 0 and curve['xVals'][-1] == 1
    for i in range(-397, 398):
        x = i / 250
        actual = d.serum_shaper(curve, x) * top / d.SHAPER_OUT
        tolerance = 0.020 if kind == d.SINE_FOLD else 1e-8
        if kind == d.BIT_CRUSH and d.vital_step(drive) < d.SHAPER_IN / 119:
            tolerance = 0.02
        assert actual == pytest.approx(d.vital_shape(kind, x, drive), abs=tolerance)


@pytest.mark.parametrize('mix', [0, 1e-10, 0.1, 0.5, 0.9, 1])
@pytest.mark.parametrize('wet_gain,pre_gain', [(1, 1), (1.7, 1), (1.2, 0.55), (1.2, 10)])
def test_mix_preserves_independent_dry_and_wet_coefficients(mix, wet_gain, pre_gain):
    wet, gain = d.mix_params(mix, wet_gain, pre_gain)
    dry_weight, wet_weight = d.serum_mix(wet)
    assert gain * dry_weight * pre_gain == pytest.approx(1 - mix, abs=1e-10)
    assert gain * wet_weight / wet_gain == pytest.approx(mix, abs=1e-10)
    assert math.isfinite(wet) and math.isfinite(gain)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('blend', [0, 0.25, 1, 1.75, 2])
def test_filter_matches_frequency_q_and_shape_coefficients(order, blend):
    params, gain = d.filter_params(order, 84, blend, 0.5)
    sf, sq, sw = d.serum_filter(params['kParamFreq'], params['kParamLPHP'], params['kParamBW'])
    vf, vq, vw = d.vital_filter(84, blend, 0.5)
    assert params['kParamPrePost'] == order
    assert sf == pytest.approx(vf)
    assert sq == pytest.approx(vq)
    assert [x * gain for x in sw] == pytest.approx(vw)


def test_filter_low_resonance_is_explicitly_reported():
    values=dict(type=d.LINEAR_FOLD,drive=6,mix=.5,filter_order=2,
                filter_cutoff=72,filter_blend=0,filter_resonance=0)
    _, notes=d.convert(values)
    assert any(feature=='distortion_filter_resonance' for _,feature,_ in notes)


@pytest.mark.parametrize('drive', [-30, -12, 0, 12, 22])
def test_down_sample_holds_nearest_host_sample(drive):
    rate = d.vital_hold_hz(drive)
    wanted = 44100 / rate
    actual = d.serum_hold(d.downsample_drive(drive))
    assert actual == pytest.approx(max(1, wanted), abs=0.51)


def test_long_sample_hold_is_reported_as_clipped():
    values=dict(type=d.DOWN_SAMPLE,drive=30,mix=1,filter_order=0,
                filter_cutoff=80,filter_blend=0,filter_resonance=.5)
    modules, notes=d.convert(values)
    assert modules[-1][2]['kParamDrive']==100
    assert any(kind=='clipped' and feature=='distortion_drive' for kind,feature,_ in notes)


@pytest.mark.parametrize('kind,mode', [(0, 'kTube'), (1, 'kHardClip'), (2, 'kXShaperAsym'),
                                     (3, 'kXShaperAsym'), (4, 'kXShaperAsym'), (5, 'kDownsample')])
def test_all_types_are_loaded_in_the_rack(template, doc, kind, mode):
    doc['settings'].update(distortion_on=1,distortion_type=kind,distortion_drive=6,
                           distortion_mix=.5,distortion_filter_order=2,
                           distortion_filter_cutoff=72,distortion_filter_resonance=.5)
    result=convert(VitalPreset(doc),template)
    effect,=rack(result,'FXDistortion')
    assert plain(effect)['kParamMode']==mode
    assert plain(effect)['kParamFreq']==pytest.approx((72+.39)/137.86)
    assert 'distortion' in result.report.mapped
    entries=result.preset.state['FXRack0']['FX']
    if kind in (2,3,4):
        assert 'flex' in entries[-1] and 'flex' not in effect
    assert SerumPreset.from_bytes(result.preset.to_bytes()).state==result.preset.state


def test_high_drive_hard_clip_keeps_dry_path_at_its_original_level(template, doc):
    doc['settings'].update(distortion_on=1,distortion_type=1,distortion_drive=30,distortion_mix=.25)
    result=convert(VitalPreset(doc),template)
    gain=math.prod((2*plain(e)['kParamLevelOut'])**2 for e in rack(result,'FXEQ'))
    effect,=rack(result,'FXDistortion');params=plain(effect)
    dry,_=d.serum_mix(params['kParamWet'])
    assert gain*(2*params['kParamLevelOut'])**2*dry==pytest.approx(.75)


@pytest.mark.parametrize('kind', [d.LINEAR_FOLD, d.SINE_FOLD, d.BIT_CRUSH])
def test_pre_filter_shape_gain_reaches_the_nonlinear_curve(kind):
    values=dict(type=kind,drive=12,mix=1,filter_order=1,filter_cutoff=84,
                filter_blend=.5,filter_resonance=.5)
    modules,_=d.convert(values)
    params=modules[-1][2];curve=modules[-1][3]['flex'][0]
    _,filter_gain=d.filter_params(1,84,.5,.5)
    output_gain=(2*params['kParamLevelOut'])**2
    for x in (-.3,-.1,.1,.3):
        assert d.serum_shaper(curve,x)*output_gain==pytest.approx(d.vital_shape(kind,x*filter_gain,12),abs=.005)


def test_shaper_range_limit_and_output_limit_are_reported():
    values=dict(type=d.BIT_CRUSH,drive=0,mix=1,filter_order=0,
                filter_cutoff=80,filter_blend=0,filter_resonance=.5)
    modules,notes=d.convert(values,span=20)
    assert any(kind=='clipped' and feature=='distortion_input' for kind,feature,_ in notes)
    assert any(kind=='clipped' and feature=='distortion' for kind,feature,_ in notes)
    assert modules[-1][2]['kParamLevelOut']==1


@pytest.mark.parametrize('kind', [d.LINEAR_FOLD, d.SINE_FOLD, d.BIT_CRUSH])
def test_shaper_headroom_stage_cancels_at_zero_mix(kind):
    values=dict(type=kind,drive=12,mix=0,filter_order=1,
                filter_cutoff=84,filter_blend=.5,filter_resonance=.5)
    modules,_=d.convert(values,span=3)
    assert modules[0][0]=='FXEQ'
    stage_gain=(2*modules[0][2]['kParamLevelOut'])**2
    dist=modules[-1][2]
    assert dist['kParamWet']==0
    assert stage_gain*(2*dist['kParamLevelOut'])**2==pytest.approx(1)


def test_high_drive_sine_curve_stays_within_entry_limit():
    values=dict(type=d.SINE_FOLD,drive=30,mix=.5,filter_order=1,
                filter_cutoff=84,filter_blend=.5,filter_resonance=.5)
    modules,_=d.convert(values,span=6.6)
    assert len(modules[-1][3]['flex'][0]['xVals'])<=d.SHAPER_ENTRIES


@pytest.mark.parametrize('kind,drive,index', [(d.LINEAR_FOLD,6,0),(d.HARD_CLIP,20,1)])
def test_filter_envelope_addresses_distortion_after_any_gain_stage(template,doc,kind,drive,index):
    doc['settings'].update(distortion_on=1,distortion_type=kind,distortion_drive=drive,
                           distortion_filter_order=2,distortion_filter_cutoff=36,
                           modulations=[{'source':'env_2','destination':'distortion_filter_cutoff'}],
                           modulation_1_amount=.5)
    result=convert(VitalPreset(doc),template)
    slot=result.preset.state['ModSlot0']
    assert slot['destModuleTypeString']=='FXDistortion'
    assert slot['destModuleID']==index
    assert slot['destModuleParamName']=='kParamFreq'
    assert slot['destModuleParamID']==5
    assert plain(slot)['kParamAmount']==pytest.approx(50*128/137.86)


def test_static_macro_drive_uses_the_full_decibel_range(template,doc):
    doc['settings'].update(distortion_on=1,distortion_type=d.LINEAR_FOLD,distortion_drive=0,
                           modulations=[{'source':'macro_control_1','destination':'distortion_drive'}],
                           modulation_1_amount=.5,macro_control_1=.5)
    result=convert(VitalPreset(doc),template)
    curve=result.preset.state['FXRack0']['FX'][0]['flex'][0]
    expected,_=d.shaper_curve(d.LINEAR_FOLD,15)
    assert curve==expected
    assert any(n.get('destination')=='distortion_drive' and 'folded' in n['message'] for n in result.report.notes)


def test_moving_mixed_fold_drive_is_explicitly_unsupported(template,doc):
    doc['settings'].update(distortion_on=1,distortion_type=d.LINEAR_FOLD,distortion_mix=.5,
                           modulations=[{'source':'env_2','destination':'distortion_drive'}],
                           modulation_1_amount=.5)
    result=convert(VitalPreset(doc),template)
    assert result.report.modulations[0]['status']=='omitted'
    assert any(n['kind']=='unsupported' and n.get('destination')=='distortion_drive' for n in result.report.notes)


@pytest.mark.parametrize('mix', [1e-155,1e-200,1e-300,5e-324])
def test_tiny_mix_keeps_finite_output_and_original_dry_level(template,doc,mix):
    wet,gain=d.mix_params(mix)
    dry,_=d.serum_mix(wet)
    assert math.isfinite(wet) and math.isfinite(gain)
    assert dry*gain==pytest.approx(1)
    doc['settings'].update(distortion_on=1,distortion_type=d.LINEAR_FOLD,distortion_mix=mix)
    result=convert(VitalPreset(doc),template)
    effect,=rack(result,'FXDistortion')
    assert math.isfinite(plain(effect)['kParamWet'])
    assert math.isfinite(plain(effect)['kParamLevelOut'])


def test_mix_underflow_keeps_finite_pure_dry_result():
    wet,gain=d.mix_params(5e-324,wet_gain=.01,pre_gain=.5)
    assert (wet,gain)==(0,2)
