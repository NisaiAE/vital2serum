import copy
import pytest
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def _stack(template,doc,version,style):
    doc['synth_version']=version
    doc['settings'].update(osc_1_unison_voices=3,osc_1_stack_style=style)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    unsupported=any(n['feature']=='osc_1_stack_style' and n['kind']=='unsupported' for n in result.report.notes)
    return plain(result.preset.state['Oscillator0']).get('kParamUnisonStack'),unsupported


# Vital 1.6.4 shifts the stack styles of a 1.0.8 preset from 4 up by one ('Octave + 7' was inserted)
# and keeps those of 1.1.0 and later as stored.
@pytest.mark.parametrize('version,style,serum',[
    ('1.0.7',3,'kOctave1'),('1.0.7',4,'kOctave2'),('1.0.7',5,'kOctaveFifth1'),('1.0.7',6,'kOctaveFifth2'),
    ('',5,'kOctaveFifth1'),
    ('1.5.5',3,'kOctave1'),('1.6.4',5,'kOctave2'),('1.6.4',6,'kOctaveFifth1'),('1.6.4',7,'kOctaveFifth2'),
])
def test_unison_stack_follows_the_numbering_of_the_preset_version(template,doc,version,style,serum):
    assert _stack(template,doc,version,style)==(serum,False)


@pytest.mark.parametrize('style',[4,11,12])        # Octave + 7, Odd Harmonics, Sub Harmonics
def test_unison_stacks_without_a_counterpart_are_reported(template,doc,style):
    assert _stack(template,doc,'1.6.4',style)==(None,True)


def test_lfo_point_cycle_is_reported_instead_of_rejected(template,doc):
    doc['synth_version']='1.6.4'
    doc['settings'].update(lfo_1_sync_type=6,modulations=[{'source':'lfo_1','destination':'osc_1_tune'}],
                           modulation_1_amount=.5)
    result=convert(VitalPreset(doc),template)
    assert any(n['feature']=='lfo_1_trigger' and n['kind']=='unsupported' for n in result.report.notes)


@pytest.mark.parametrize('morph',range(12,17))     # added after Vital 1.0
def test_new_spectral_morphs_are_reported_instead_of_rejected(template,doc,morph):
    doc['synth_version']='1.6.4'
    doc['settings'].update(osc_1_spectral_morph_type=morph,osc_1_spectral_morph_amount=.5)
    result=convert(VitalPreset(doc),template)
    assert any(n['feature']=='osc_1_spectral_morph' and n['kind']=='unsupported' for n in result.report.notes)
