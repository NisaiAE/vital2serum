import pytest
from vital2serum.effects import comb_feedback, phaser_fit, resonator_cutoff
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def configure(doc,model=6,style=1,**settings):
    doc['settings'].update(filter_1_on=1,filter_1_model=model,filter_1_style=style,
                           filter_1_cutoff=60,filter_1_resonance=.5,filter_1_blend=0,
                           filter_1_blend_transpose=42,filter_1_drive=0,filter_1_mix=1,
                           filter_1_keytrack=0)
    doc['settings'].update(settings)
    return doc


def slots(result):
    return [s for k,s in result.preset.state.items() if k.startswith('ModSlot') and
            isinstance(s,dict) and s.get('destModuleTypeString')]


def test_comb_feedback_has_signed_square_root_and_flange_is_unipolar():
    assert [comb_feedback(0,r) for r in (.0,.375,.5,.625,1)]==pytest.approx([-1,-.5,0,.5,1])
    assert [comb_feedback(3,r) for r in (.0,.5,1)]==pytest.approx([-1,0,1])
    assert comb_feedback(2,.6)==pytest.approx(.6)


def test_negative_flange_uses_half_delay(template,doc):
    configure(doc,style=2)
    params=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])
    assert params['kParamEnable']==1 and params['kParamType'].startswith('Flange')
    assert params['kParamType'].endswith('N')
    assert params['kParamFreq']==pytest.approx(resonator_cutoff(72),abs=.001)


def test_phaser_stage_count_and_cutoff_alignment(template,doc):
    configure(doc,model=7,style=0)
    params=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])
    assert params['kParamType']=='Phase24P'
    assert params['kParamFreq']==pytest.approx(71.9/136.8)
    configure(doc,model=7,style=1,filter_1_blend=1)
    assert plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])['kParamType']=='Phase48N'


def test_phaser_notch_floor_uses_partial_wet_and_makeup():
    res,wet,gain=phaser_fit(1,4,.5)
    assert (res,wet,gain)==pytest.approx((40,75,3.92))
    assert phaser_fit(0,4,0)==pytest.approx((0,100,.65))


def test_zero_filter_mix_preserves_dry_gain(template,doc):
    configure(doc,model=7,style=1,filter_1_mix=0)
    params=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])
    assert params['kParamEnable']==0


def test_resonator_keyboard_root_uses_its_cutoff_span(template,doc):
    configure(doc,model=7,filter_1_keytrack=1)
    params=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])
    assert params['kParamKeyTrack']==1
    assert params['kParamFreq']==pytest.approx((71.9-28)/136.8)
    configure(doc,model=7,filter_1_keytrack=.5)
    result=convert(VitalPreset(doc),template)
    slot,=slots(result)
    assert slot['source']==[17,0]
    assert plain(slot)['kParamAmount']==pytest.approx(100*.5*127/136.8)


def test_resonator_cutoff_modulation_uses_delay_scale(template,doc):
    configure(doc)
    doc['settings']['modulations']=[{'source':'env_2','destination':'filter_1_cutoff'}]
    doc['settings']['modulation_1_amount']=.25
    slot,=slots(convert(VitalPreset(doc),template))
    assert slot['destModuleParamName']=='kParamFreq'
    assert plain(slot)['kParamAmount']==pytest.approx(100*.25*128/136.8)


@pytest.mark.parametrize('style',[2,5])
def test_negative_flange_resonance_modulation_retains_depth(template,doc,style):
    configure(doc,style=style,filter_1_blend=1)
    doc['settings']['modulations']=[{'source':'env_2','destination':'filter_1_resonance'}]
    doc['settings']['modulation_1_amount']=.2
    slot,=slots(convert(VitalPreset(doc),template))
    assert slot['destModuleParamName']=='kParamReso'
    assert plain(slot)['kParamAmount']>0


def test_fx_resonator_mix_difference_keeps_bypass(template,doc):
    configure(doc,model=7,filter_1_on=0)
    doc['settings'].update(filter_fx_on=1,filter_fx_model=7,filter_fx_style=0,
                           filter_fx_cutoff=60,filter_fx_keytrack=.5,filter_fx_blend=0,
                           filter_fx_drive=0,filter_fx_resonance=.5,filter_fx_mix=1)
    result=convert(VitalPreset(doc),template)
    assert not result.preset.state['FXRack0']['FX']
    assert any(n['kind']=='unsupported' and n['feature']=='filter_fx_' for n in result.report.notes)


@pytest.mark.parametrize('settings',[{'filter_1_blend':2},{'filter_1_drive':10}])
def test_unmatched_phaser_settings_keep_bypass_and_report(template,doc,settings):
    configure(doc,model=7,**settings)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['VoiceFilter0'])['kParamEnable']==0
    assert any(n['kind']=='unsupported' and n['feature']=='filter_1_' for n in result.report.notes)


def test_uncalibrated_high_feedback_keeps_bypass(template,doc):
    configure(doc,filter_1_resonance=1)
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['VoiceFilter0'])['kParamEnable']==0
    assert any(n['kind']=='unsupported' and 'feedback' in n['message'] for n in result.report.notes)


def test_stage_count_modulation_is_reported_as_unsupported(template,doc):
    configure(doc,model=7)
    doc['settings']['modulations']=[{'source':'lfo_1','destination':'filter_1_blend'}]
    doc['settings']['modulation_1_amount']=.5
    result=convert(VitalPreset(doc),template)
    assert not slots(result)
    assert result.report.modulations[0]['status']=='omitted'
    assert any(n['kind']=='unsupported' for n in result.report.notes)


def test_static_resonance_macro_is_fitted_at_its_stored_value(template,doc):
    configure(doc,model=7,style=1,filter_1_resonance=0)
    doc['settings'].update(macro_control_1=.5,modulation_1_amount=1)
    doc['settings']['modulations']=[{'source':'macro_control_1','destination':'filter_1_resonance'}]
    result=convert(VitalPreset(doc),template)
    params=plain(result.preset.state['VoiceFilter0'])
    assert params['kParamReso']==pytest.approx(phaser_fit(1,4,.5)[0])
    assert not slots(result)


def test_parallel_resonator_keeps_bypass(template,doc):
    configure(doc)
    doc['settings'].update(filter_2_on=1,filter_2_model=0,filter_2_filter_input=0)
    result=convert(VitalPreset(doc),template)
    assert not any('Flange' in str(s) for s in result.preset.state.values())
    assert any(n['kind']=='unsupported' and 'Combined' in n['message'] for n in result.report.notes)
