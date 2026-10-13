import copy
import pytest
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def setup_axis(doc, axis='frame'):
    settings=doc['settings']
    settings.update(osc_1_distortion_type=6,osc_1_distortion_amount=.3,macro_control_1=.4,
                    modulations=[{'source':'macro_control_1','destination':'osc_1_distortion_amount'},
                                 {'source':'lfo_1','destination':f'osc_1_{"wave_frame" if axis=="frame" else "spectral_morph_amount"}'}],
                    modulation_1_amount=.5,modulation_2_amount=.4)
    if axis=='morph': settings.update(osc_1_spectral_morph_type=8,osc_1_spectral_morph_amount=.2)
    return settings


@pytest.mark.parametrize('axis',['frame','morph'])
def test_fixed_warp_leaves_one_moving_axis(template,doc,axis):
    settings=setup_axis(doc,axis)
    expected=copy.deepcopy(doc)
    expected['settings']['osc_1_distortion_amount']=.5
    expected['settings']['modulations'][0]={}
    vital=VitalPreset(doc);result=convert(vital,template)
    reference=convert(VitalPreset(expected),template)
    assert vital.folded=={1}
    wt=result.preset.state['Oscillator0']['WTOsc0']
    assert wt['embeddedWTData']==reference.preset.state['Oscillator0']['WTOsc0']['embeddedWTData']
    assert 'kParamWarpMenu' not in plain(wt)
    route=result.preset.state['ModSlot0']
    assert route['source']==[6,0] and route['destModuleParamName']=='kParamTablePos'
    assert plain(result.preset.state['Macro0'])['kParamValue']==40
    note=next(n for n in result.report.notes if n.get('remaining_axis')==axis)
    assert note['slots']==[1] and 'changing the folded control' in note['message']


@pytest.mark.parametrize('reason',['moving_macro','moving_aux','stereo','phase','spread','two_axes','no_axis'])
def test_fixed_warp_excludes_dynamic_or_multiaxis_cases(template,doc,reason):
    s=setup_axis(doc)
    if reason=='moving_macro':
        s['modulations'].append({'source':'lfo_2','destination':'macro_control_1'});s['modulation_3_amount']=.2
    elif reason=='moving_aux':
        s['modulations'].append({'source':'env_2','destination':'modulation_1_amount'});s['modulation_3_amount']=.2
    elif reason=='stereo':s['modulation_1_stereo']=1
    elif reason=='phase':
        s['modulations'].append({'source':'lfo_2','destination':'osc_1_distortion_phase'});s['modulation_3_amount']=.2
    elif reason=='spread':s.update(osc_1_unison_voices=3,osc_1_distortion_spread=.2)
    elif reason=='two_axes':
        s.update(osc_1_spectral_morph_type=8)
        s['modulations'].append({'source':'env_2','destination':'osc_1_spectral_morph_amount'});s['modulation_3_amount']=.2
    elif reason=='no_axis':s['modulations'][1]={}
    vital=VitalPreset(doc);result=convert(vital,template)
    assert 1 not in vital.folded
    assert not any('remaining_axis' in n for n in result.report.notes)


def test_static_aux_folds_only_warp_root_and_keeps_other_aux_routes(template,doc):
    s=setup_axis(doc)
    s.update(macro_control_2=.5,modulation_1_amount=.1,modulation_3_amount=.25,
             modulation_4_amount=.2,modulation_5_amount=.1)
    s['modulations'] += [
        {'source':'macro_control_2','destination':'modulation_1_amount'},
        {'source':'env_2','destination':'osc_1_pan'},
        {'source':'macro_control_2','destination':'modulation_4_amount'}]
    vital=VitalPreset(doc);result=convert(vital,template)
    assert vital.folded=={1}
    # Effective depth .1 + 2 * .5 * .25 = .35, then the macro contributes .4 * .35.
    assert vital.get('osc_1_distortion_amount')==pytest.approx(.44)
    slots=[m for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')=='kParamPan']
    assert any(m['source']==[3,0] for m in slots)
    assert any(m['source']==[3,26] for m in slots)
    assert 3 not in vital.folded and 5 not in vital.folded
