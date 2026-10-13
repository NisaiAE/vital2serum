import math
import pytest
from vital2serum import unison
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


@pytest.mark.parametrize('value',[0,.25,.5,.635,1])
def test_saved_wheel_percentage_and_active_source(template,doc,value):
    doc['settings'].update(mod_wheel=value,modulations=[{'source':'mod_wheel','destination':'osc_1_tune'}],modulation_1_amount=.25)
    result=convert(VitalPreset(doc),template)
    wheel=plain(result.preset.state['Global0'])['kParamModWheel']
    route=result.preset.state['ModSlot0']
    assert wheel==pytest.approx(100*value)
    assert route['source']==[1,0] and route['destModuleParamName']=='kParamFine'
    assert wheel/100*plain(route)['kParamAmount']/100==pytest.approx(value*.25)


@pytest.mark.parametrize('span',[0,.01,.5,1,1.52,1.76,2,3.375,26.24,45.44,48])
@pytest.mark.parametrize('detune',[0,3.5,10])
def test_integer_range_preserves_outer_pitch(span,detune):
    native=unison.serum_range(span)
    stored=unison.serum_detune(detune,span)
    assert 0<=native<=48 and native==math.ceil(span)
    assert 0<=stored<=1
    assert native*stored**1.25==pytest.approx(span*(detune/10)**2,abs=1e-12)
    assert unison.detune_position(detune,span)**2==pytest.approx(stored)


def test_fractional_range_writes_integer_and_compensated_detune(template,doc):
    doc['settings'].update(osc_1_unison_voices=3,osc_1_detune_range=1.52,osc_1_unison_detune=3.5,
                           modulations=[{'source':'lfo_1','destination':'osc_1_unison_detune'}],modulation_1_amount=.5)
    result=convert(VitalPreset(doc),template)
    params=plain(result.preset.state['Oscillator0'])
    assert 'kParamUnisonRange' not in params
    assert params['kParamDetune']==pytest.approx(unison.serum_detune(3.5,1.52))
    route=result.preset.state['ModSlot0']
    assert route['destModuleParamName']=='kParamDetune' and route.get('flex')
    assert any(n['feature']=='osc_1_detune_range' and n.get('converted')==2 for n in result.report.notes)


def test_zero_range_has_no_detune_at_any_source_value(template,doc):
    doc['settings'].update(osc_1_unison_voices=3,osc_1_detune_range=0,osc_1_unison_detune=10,
                           modulations=[{'source':'lfo_1','destination':'osc_1_unison_detune'}],modulation_1_amount=1)
    result=convert(VitalPreset(doc),template)
    params=plain(result.preset.state['Oscillator0'])
    assert params['kParamUnisonRange']==0 and params['kParamDetune']==0
    for slot in result.preset.state.values():
        if isinstance(slot,dict) and slot.get('destModuleParamName')=='kParamDetune':
            assert plain(slot).get('kParamAmount',0)==0
