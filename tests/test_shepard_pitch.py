import copy
import pytest
import numpy as np
from vital2serum.mapping import Context, _shepard_pitch, convert
from vital2serum.modulation import KNOWN_IDS
from vital2serum.report import Report
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def setup(doc,amount=0,dynamic=False):
    s=doc['settings'];s.update(osc_1_spectral_morph_type=10,osc_1_spectral_morph_amount=amount,
                             osc_1_spectral_morph_spread=0,osc_1_wave_frame=0,osc_1_distortion_type=0)
    if dynamic:s.update(modulations=[{'source':'lfo_1','destination':'osc_1_spectral_morph_amount'}],modulation_1_amount=1)
    return s


@pytest.mark.parametrize('amount,expected',[(0,-12),(.25,-9),(.75,-3),(1,0)])
def test_static_pitch_bakes_spectrum_with_the_matching_pitch(template,doc,amount,expected):
    setup(doc,amount);reference=copy.deepcopy(doc);reference['settings']['osc_1_spectral_morph_type']=0
    result=convert(VitalPreset(doc),template);control=convert(VitalPreset(reference),template)
    assert plain(result.preset.state['Oscillator0'])['kParamCoarsePit']==expected
    pcm=np.asarray(result.preset.state['Oscillator0']['WTOsc0']['embeddedWTData'])
    dominant=int(np.argmax(np.abs(np.fft.rfft(pcm[:2048]))[1:])+1)
    assert dominant==(2 if amount<.5 else 1)
    assert any(n['feature']=='osc_1_spectral_morph' and n['kind']=='approximation' and 'rendered' in n['message'] for n in result.report.notes)
    assert any(n['feature']=='osc_1_shepard_pitch' and 'wrap phase history' in n['message'] for n in result.report.notes)


def test_single_linear_amount_gets_partial_optional_pitch(template,doc):
    setup(doc,dynamic=True);result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['Oscillator0'])['kParamCoarsePit']==-12
    route=result.preset.state['ModSlot0']
    assert route['source']==[6,0] and route['destModuleParamName']=='kParamCoarsePit'
    assert plain(route)['kParamAmount']==9.375
    assert result.report.modulations[0]['status']=='omitted'


def test_zero_amount_remains_static_with_all_native_slots_used(template,doc):
    s=setup(doc,dynamic=True);s.update(modulation_1_amount=0,velocity_track=1)
    s['modulations'] += [{'source':'env_2','destination':'osc_1_pan'} for _ in range(63)]
    for n in range(2,65):s[f'modulation_{n}_amount']=.01
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['Oscillator0'])['kParamCoarsePit']==-12
    routes=[m for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')]
    assert len(routes)==64 and sum(m['destModuleParamName']=='kParamPan' for m in routes)==63
    assert any(n['feature']=='osc_1_shepard_pitch' and n.get('dynamic') is False for n in result.report.notes)


@pytest.mark.parametrize('reason',['frame','frame_spread','spread','morph_phase','transpose','aux','multiple','stereo','power','bipolar','out_of_range','source_rate','smoothing','keytrack','envelope','fx'])
def test_dynamic_pitch_excludes_unmeasured_conditions(template,doc,reason):
    s=setup(doc,dynamic=True)
    if reason=='frame':s['modulations'].append({'source':'lfo_2','destination':'osc_1_wave_frame'});s['modulation_2_amount']=.2
    elif reason=='frame_spread':s['osc_1_frame_spread']=.2
    elif reason=='spread':s['osc_1_spectral_morph_spread']=.2
    elif reason=='morph_phase':s['modulations'].append({'source':'lfo_2','destination':'osc_1_spectral_morph_phase'});s['modulation_2_amount']=.2
    elif reason=='transpose':s['modulations'].append({'source':'env_2','destination':'osc_1_transpose'});s['modulation_2_amount']=.2
    elif reason=='aux':s['modulations'].append({'source':'env_2','destination':'modulation_1_amount'});s['modulation_2_amount']=.2
    elif reason=='multiple':s['modulations'].append({'source':'lfo_2','destination':'osc_1_spectral_morph_amount'});s['modulation_2_amount']=.2
    elif reason=='stereo':s['modulation_1_stereo']=1
    elif reason=='power':s['modulation_1_power']=2
    elif reason=='bipolar':s['modulation_1_bipolar']=1
    elif reason=='out_of_range':s['osc_1_spectral_morph_amount']=.5
    elif reason=='source_rate':s['modulations'].append({'source':'lfo_2','destination':'lfo_1_frequency'});s['modulation_2_amount']=.2
    elif reason=='smoothing':s.update(lfo_1_sync=0,lfo_1_frequency=8,lfo_1_smooth_mode=1,lfo_1_smooth_time=-7.5)
    elif reason=='keytrack':s['lfo_1_sync']=4
    elif reason=='envelope':s['lfo_1_sync_type']=2
    elif reason=='fx':s['chorus_on']=1
    result=convert(VitalPreset(doc),template)
    assert 'kParamCoarsePit' not in plain(result.preset.state['Oscillator0'])
    assert not any(n['feature']=='osc_1_shepard_pitch' for n in result.report.notes)


@pytest.mark.parametrize('velocity,expected_pitch',[(0,-12),(1,0)])
def test_optional_route_keeps_existing_paths_at_slot_limit(template,doc,velocity,expected_pitch):
    s=setup(doc,dynamic=True);s['velocity_track']=velocity
    s['modulations'] += [{'source':'env_2','destination':'osc_1_pan'} for _ in range(63)]
    for n in range(2,65):s[f'modulation_{n}_amount']=.01
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['Oscillator0']).get('kParamCoarsePit',0)==expected_pitch
    routes=[m for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')]
    assert len(routes)==64
    assert sum(m['destModuleParamName']=='kParamPan' for m in routes)==63


def test_missing_pitch_id_keeps_original_pitch(template,doc):
    setup(doc,dynamic=True);ids=dict(KNOWN_IDS);ids.pop(('Oscillator','kParamCoarsePit'))
    result=convert(VitalPreset(doc),template,destination_ids=ids)
    assert 'kParamCoarsePit' not in plain(result.preset.state['Oscillator0'])


def test_native_pitch_limit_is_checked(template,doc):
    setup(doc);ctx=Context(VitalPreset(doc),template,Report('test','test'))
    params={'kParamCoarsePit':-60}
    _shepard_pitch(ctx,'osc_1_',0,params,0)
    assert params['kParamCoarsePit']==-60 and not ctx.optional_routings


def test_other_spectral_sweep_keeps_table_position_route(template,doc):
    s=setup(doc,dynamic=True);s['osc_1_spectral_morph_type']=8
    result=convert(VitalPreset(doc),template)
    assert result.preset.state['ModSlot0']['destModuleParamName']=='kParamTablePos'
    assert not any(n['feature']=='osc_1_shepard_pitch' for n in result.report.notes)
