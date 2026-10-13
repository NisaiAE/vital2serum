import copy
import pytest
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset
from conftest import wave


@pytest.mark.parametrize('transpose',[-48,-19,0,12,28,48])
@pytest.mark.parametrize('note',[48,60])
def test_keytracking_off_keeps_the_fixed_pitch_range(template,doc,transpose,note):
    doc['settings'].update(osc_1_midi_track=0,osc_1_transpose=transpose)
    osc=plain(convert(VitalPreset(doc),template).preset.state['Oscillator0'])
    assert osc['kParamPitchTrack']==0
    assert (osc['kParamOctave'],osc['kParamPitch'],osc['kParamCoarsePit'])==(4,0,transpose)
    # With key tracking off Serum plays note 0 regardless of the MIDI note.
    played=(note if osc['kParamPitchTrack'] else 0)+12*osc['kParamOctave']+osc['kParamPitch']+osc['kParamCoarsePit']
    assert played==48+transpose


@pytest.mark.parametrize('transpose,amount,expected_coarse',[
    (-48,1,48),(48,-1,-48),(12,.375,48),(28,-.75,-44),
])
def test_fixed_pitch_modulation_has_room_at_both_ends(template,doc,transpose,amount,expected_coarse):
    doc['settings'].update(osc_1_midi_track=0,osc_1_transpose=transpose,
        modulations=[{'source':'env_2','destination':'osc_1_transpose'}],modulation_1_amount=amount)
    result=convert(VitalPreset(doc),template)
    osc=plain(result.preset.state['Oscillator0']);slot=result.preset.state['ModSlot0']
    assert slot['source']==[3,0] and slot['destModuleParamName']=='kParamCoarsePit'
    assert plain(slot)['kParamAmount']==pytest.approx(amount*75)
    coarse=osc['kParamCoarsePit']+128*plain(slot)['kParamAmount']/100
    assert coarse==pytest.approx(expected_coarse) and -64<=coarse<=64
    assert all(m['status']=='mapped' for m in result.report.modulations)


def test_keytracking_on_keeps_its_pitch_controls(template,doc):
    doc['settings'].update(osc_1_midi_track=1,osc_1_transpose=19)
    osc=plain(convert(VitalPreset(doc),template).preset.state['Oscillator0'])
    assert (osc['kParamOctave'],osc['kParamPitch'])==(1,7)
    assert 'kParamCoarsePit' not in osc and 'kParamPitchTrack' not in osc


@pytest.mark.parametrize('warp',[7,10])
def test_fixed_pitch_preserves_cross_warp_and_unison_stack(template,doc,warp):
    doc['settings'].update(osc_2_on=1,wavetables=[wave(),wave()],osc_1_transpose=12,
        osc_1_distortion_type=warp,osc_1_distortion_amount=.3,osc_1_stack_style=5,
        osc_1_unison_voices=3)
    tracked=convert(VitalPreset(copy.deepcopy(doc)),template).preset.state['Oscillator0']
    doc['settings']['osc_1_midi_track']=0
    fixed=convert(VitalPreset(doc),template).preset.state['Oscillator0']
    assert plain(fixed)['kParamUnisonStack']==plain(tracked)['kParamUnisonStack']=='kOctaveFifth1'
    assert plain(fixed)['kParamUnison']==plain(tracked)['kParamUnison']==3
    assert plain(fixed['WTOsc0'])==plain(tracked['WTOsc0'])
