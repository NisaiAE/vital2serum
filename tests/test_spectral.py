import copy
import math
import numpy as np
import pytest
from conftest import wave
from vital2serum.mapping import convert
from vital2serum.modulation import KNOWN_IDS
from vital2serum.serum import plain
from vital2serum.spectral import (HARMONICS, SIZE, STATIC_MORPHS, TRAJECTORY_MORPHS,
                                  TableSpectrum, spectral_frames, static_eligible)
from vital2serum.trajectory import correlated_controls, render_trajectory
from vital2serum.vital import VitalPreset
from vital2serum.wavetable import render_wavetable


def spectrum_table(coeff):
    return TableSpectrum.from_frames(np.fft.irfft(coeff,n=SIZE,axis=-1))


def configure(doc,kind,amount=.3):
    doc['settings'].update(osc_1_spectral_morph_type=kind,osc_1_spectral_morph_amount=amount,
        osc_1_spectral_morph_phase=.5,osc_1_wave_frame=0,osc_1_unison_voices=1,
        osc_1_frame_spread=0,osc_1_spectral_morph_spread=0,osc_1_distortion_type=0)
    return doc['settings']


@pytest.mark.parametrize('kind,amount',[(2,.5),(5,0),(6,0),(9,.5),(11,0)])
def test_neutral_and_nyquist_boundary(kind,amount):
    x=np.arange(SIZE)/SIZE
    frames=np.stack([np.sin(2*np.pi*x),np.sin(2*np.pi*x)+.2*np.cos(14*np.pi*x)])
    spec=TableSpectrum.from_frames(frames)
    expected=spec.amplitudes[1]*spec.normalized[1];expected[-1]=0
    assert np.max(np.abs(spec.coefficients(kind,amount,1)-expected))<5e-12


def test_low_amplitude_normalized_phase_is_complex_interpolation():
    c=np.zeros((5,HARMONICS),complex);c[:,1]=500j;c[0,7]=1;c[-1,7]=-1
    spec=spectrum_table(c)
    assert abs(spec.normalized[2,7])<1e-12
    assert np.max(np.abs(spec.normalized[:,9]-1j))<1e-12


def test_smear_parity_recurrence_and_silent_endpoint():
    c=np.zeros(HARMONICS,complex);c[1]=512j;spec=spectrum_table(c)
    raw=1-(1-.5)**(1/3)
    out=spec.coefficients(5,raw)
    assert abs(out[3]-128j)<1e-10 and abs(out[5]-80j)<1e-10
    assert np.max(np.abs(out[::2]))<1e-10
    assert np.max(np.abs(spec.coefficients(5,1)))==0


def test_formant_scale_parity_and_pull_weights():
    c=np.zeros(HARMONICS,complex);c[3]=100+70j;spec=spectrum_table(c)
    out=spec.coefficients(2,.75)
    assert np.max(np.abs(out[::2]))<1e-10
    assert abs(out[5]-.375*c[3])<1e-10


def test_random_repeat_does_not_depend_on_other_evaluation():
    x=np.arange(SIZE)/SIZE;spec=TableSpectrum.from_frames(np.sin(2*np.pi*x))
    first=spec.coefficients(6,.79);spec.coefficients(6,.123)
    assert np.array_equal(first,spec.coefficients(6,.79))


def test_full_normalize_includes_original_position_255():
    x=np.arange(SIZE)/SIZE;sine=np.sin(2*np.pi*x)
    table=wave(sine.tolist());comp=table['groups'][0]['components'][0]
    end=copy.deepcopy(comp['keyframes'][0]);end['position']=256
    large=copy.deepcopy(comp['keyframes'][0]);large['position']=255
    import base64
    large['wave_data']=base64.b64encode((sine*4).astype('<f4').tobytes()).decode()
    comp.update(interpolation_style=0,keyframes=[comp['keyframes'][0],large,end]);table['full_normalize']=True
    legacy,_=render_wavetable(table);full,_=render_wavetable(table,full=True)
    assert len(full)==257 and np.max(np.abs(full[255]))==1
    assert np.max(np.abs(full[0]))==.25
    assert np.max(np.abs(legacy[0]))==1


@pytest.mark.parametrize('kind',list(STATIC_MORPHS))
def test_static_scalar_is_baked_with_fixed_gain_and_reuse(template,doc,kind):
    configure(doc,kind);v=VitalPreset(doc)
    result=convert(v,template);again=convert(v,template)
    assert result.preset.to_bytes()==again.preset.to_bytes()
    assert result.report.as_dict()==again.report.as_dict()
    assert any(n['feature']=='osc_1_spectral_morph' and n['kind']=='approximation' and 'rendered' in n['message'] for n in result.report.notes)
    if kind==10:assert plain(result.preset.state['Oscillator0'])['kParamCoarsePit']==pytest.approx(-8.4)


@pytest.mark.parametrize('reason',['phase','moving_phase','moving_type','moving_amount','unison_frame','unison_morph','voicecount_spread'])
def test_static_scope_guard_does_not_mutate_accounting(doc,reason):
    s=configure(doc,5)
    if reason=='phase':s['osc_1_spectral_morph_phase']=0
    elif reason.startswith('moving_'):
        s.update(modulations=[{'source':'env_2','destination':'osc_1_spectral_morph_'+reason[7:]}],modulation_1_amount=.2)
    else:
        s['osc_1_unison_voices']=2 if reason!='voicecount_spread' else 1
        s['osc_1_frame_spread']=.2 if reason=='unison_frame' else 0
        s['osc_1_spectral_morph_spread']=.2 if reason!='unison_frame' else 0
        if reason=='voicecount_spread':s.update(modulations=[{'source':'env_2','destination':'osc_1_unison_voices'}],modulation_1_amount=.2)
    v=VitalPreset(doc);v.used.add('osc_1_on');before=set(v.used)
    assert not static_eligible(v,'osc_1_')
    assert v.used==before


@pytest.mark.parametrize('reason',['frame','moving_frame','warp','frame_spread','morph_spread','moving_pitch','moving_voicecount'])
def test_shepard_cannot_enter_a_scope_without_the_fixed_pitch_law(doc,reason):
    s=configure(doc,10)
    if reason=='frame':s['osc_1_wave_frame']=1
    elif reason=='warp':s['osc_1_distortion_type']=4
    elif reason.endswith('spread'):s['osc_1_'+reason.replace('morph','spectral_morph')]=.2
    else:
        key={'moving_frame':'wave_frame','moving_pitch':'transpose','moving_voicecount':'unison_voices'}[reason]
        s.update(modulations=[{'source':'env_2','destination':'osc_1_'+key}],modulation_1_amount=.2)
    assert not static_eligible(VitalPreset(doc),'osc_1_')
    assert correlated_controls(VitalPreset(doc),'osc_1_') is None


@pytest.mark.parametrize('kind',list(TRAJECTORY_MORPHS))
def test_single_moving_spectral_axis_uses_sum_then_clamp_and_full_context(doc,kind):
    s=configure(doc,kind,.3)
    s.update(modulations=[{'source':'env_2','destination':'osc_1_spectral_morph_amount'}]*2,
             modulation_1_amount=.8,modulation_2_amount=-.2,modulation_2_bipolar=1,modulation_2_power=2)
    v=VitalPreset(doc);plan=correlated_controls(v,'osc_1_');assert plan is not None
    full,_=render_wavetable(s['wavetables'][0],full=True);spec=TableSpectrum.from_frames(full)
    rendered=render_trajectory(full,plan,count=9)
    for i,raw in enumerate(np.linspace(0,1,9)):
        amount=min(1,max(0,.3+sum(m.amount*m.value(float(raw)) for m in plan.routes)))
        assert np.max(np.abs(rendered[i]-spec.frame(kind,amount)))<1e-12


def test_single_spectral_slot_failure_is_pure_rerun_with_other_oscillator_preserved(template,doc):
    s=configure(doc,5);s.update(modulations=[{'source':'env_2','destination':'osc_1_spectral_morph_amount'}],modulation_1_amount=.6)
    s.update(osc_2_on=1,osc_2_spectral_morph_type=9,osc_2_spectral_morph_amount=.2)
    s['wavetables'][1]=copy.deepcopy(s['wavetables'][0])
    v=VitalPreset(doc);ids=dict(KNOWN_IDS);ids.pop(('WTOsc','kParamTablePos'))
    failed=convert(v,template,destination_ids=ids)
    pure=convert(VitalPreset(copy.deepcopy(doc)),template,destination_ids=ids,_disabled_trajectories=frozenset({'osc_1_'}))
    assert failed.preset.to_bytes()==pure.preset.to_bytes()
    assert failed.preset.state['Oscillator1']==pure.preset.state['Oscillator1']
    assert any(n['feature']=='osc_1_trajectory' and 'original conversion' in n['message'] for n in failed.report.notes)


@pytest.mark.parametrize('kind',list(TRAJECTORY_MORPHS))
def test_new_spectral_dynamic_does_not_expand_to_joint_frame_or_warp(doc,kind):
    s=configure(doc,kind)
    s.update(modulations=[{'source':'env_2','destination':'osc_1_spectral_morph_amount'},
                          {'source':'env_2','destination':'osc_1_wave_frame'}],
             modulation_1_amount=.2,modulation_2_amount=.1)
    assert correlated_controls(VitalPreset(doc),'osc_1_') is None


def test_folded_macro_shepard_keeps_the_partial_path(template,doc):
    s=configure(doc,10)
    s.update(macro_control_1=.6,modulations=[{'source':'macro_control_1','destination':'osc_1_spectral_morph_amount'}],modulation_1_amount=.5)
    result=convert(VitalPreset(doc),template)
    assert any(n['feature']=='osc_1_spectral_morph' and n['kind']=='unsupported' for n in result.report.notes)
    assert any(n['feature']=='osc_1_shepard_pitch' and 'Only' in n['message'] for n in result.report.notes)



def test_first_weak_phase_is_retained_before_the_first_good_frame():
    coeff=np.zeros((5,HARMONICS),complex)
    coeff[:,1]=500j
    coeff[:3,7]=.05
    coeff[3:,7]=1j
    spec=spectrum_table(coeff)
    assert abs(spec.normalized[0,7]-1)<1e-12
    assert np.max(np.abs(spec.normalized[1:,7]-1j))<1e-12
    assert abs(spec.phases[0,7])<1e-12
