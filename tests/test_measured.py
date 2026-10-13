"""Behaviour pinned to measurements of Vital 1.0.7 and Serum 2.1.5."""
import math
import pytest
from vital2serum import unison
from vital2serum.curves import curve_value, power_scale
from vital2serum.effects import (HARD_CLIP_MAX_DRIVE_DB, hard_clip, level_out, serum_eq_reso, serum_shelf_reso,
                                 soft_clip, vital_eq_q)
from vital2serum.mapping import convert, synced_delay
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def slots(result):
    state=result.preset.state
    return [state[f'ModSlot{i}'] for i in range(64) if isinstance(state[f'ModSlot{i}'],dict) and state[f'ModSlot{i}'].get('destModuleTypeString')]


def routing(doc,*mods,**settings):
    """mods: (source, destination, amount[, bipolar[, power]])."""
    doc['settings']['modulations']=[{'source':m[0],'destination':m[1]} for m in mods]
    for i,m in enumerate(mods,1):
        doc['settings'][f'modulation_{i}_amount']=m[2]
        if len(m)>3: doc['settings'][f'modulation_{i}_bipolar']=m[3]
        if len(m)>4: doc['settings'][f'modulation_{i}_power']=m[4]
    doc['settings'].update(settings)
    return doc


def rack(result,name):
    return [e[name]['plainParams'] for e in result.preset.state['FXRack0']['FX'] if name in e]


# --- modulation ---------------------------------------------------------------------------------
def test_nested_amount_modulation_goes_through_a_spare_macro(template,doc):
    # Vital: LFO depth = macro 1 * envelope 2. Serum scales a routing by one aux source only, so
    # the product is formed in macro 5 (which an aux source reads without delay).
    routing(doc,('lfo_1','osc_1_tune',0,1),('macro_control_1','modulation_1_amount',0),('env_2','modulation_2_amount',.25),
            macro_control_1=.296)
    r=convert(VitalPreset(doc),template)
    bus,leaf=slots(r)
    assert bus['source']==[3,25] and (bus['destModuleTypeString'],bus['destModuleID'],bus['destModuleParamName'])==('Macro',4,'kParamValue')
    assert plain(bus)=={'kParamAmount':100.0,'kParamBipolar':0.0}
    assert leaf['source']==[6,29] and leaf['destModuleParamName']=='kParamFine'
    # 2 * 2 * 0.25 = 1.0 of the amount range at full macro and envelope
    assert plain(leaf)=={'kParamAmount':100.0,'kParamBipolar':1.0}
    assert r.preset.state['Macro4']['name'].startswith('x ') and plain(r.preset.state['Macro4'])['kParamValue']==0
    assert all(m['status']=='mapped' for m in r.report.modulations)


def test_amount_above_100_percent_is_spread_over_slots(template,doc):
    routing(doc,('lfo_1','osc_1_tune',0,1),('macro_control_2','modulation_1_amount',1.0),macro_control_2=.262)
    r=convert(VitalPreset(doc),template)
    assert [(s['source'],plain(s)['kParamAmount']) for s in slots(r)]==[([6,26],100.0),([6,26],100.0)]
    assert not any(n['kind']=='clipped' and 'modulation' in n['feature'] for n in r.report.notes)


def test_bipolar_amount_modulator_is_split_into_aux_and_constant(template,doc):
    # amount = 0.2 + 2 * (macro - 0.5) * 0.1 = 0.1 + 0.2 * macro
    routing(doc,('lfo_1','osc_1_pan',.2),('macro_control_1','modulation_1_amount',.1,1))
    r=convert(VitalPreset(doc),template)
    assert sorted((s['source'][1],round(plain(s)['kParamAmount'],6)) for s in slots(r))==[(0,10.0),(25,20.0)]


def test_time_destination_gets_a_response_curve(template,doc):
    # Vital: seconds = control^4. Serum: seconds = 32 * position^5. The curve carries the difference.
    routing(doc,('macro_control_2','env_2_attack',.5783),env_2_attack=0.0)
    slot,=slots(convert(VitalPreset(doc),template))
    assert (slot['destModuleTypeString'],slot['destModuleID'],slot['destModuleParamName'],slot['destModuleParamID'])==('Env',1,'kParamAttack',0)
    params=plain(slot)
    assert params['kParamAmount']==pytest.approx(100*.5783**.8,rel=1e-4) and params['kParamMainCurveData']==1
    curve=slot['flex'][0]
    assert curve['numPoints']==len(curve['xVals'])-1==32
    assert 1-curve['yVals'][16]==pytest.approx(.5**.8,rel=1e-4)  # half the macro -> 0.5^0.8 of the way


def test_velocity_controls_the_glide_time(template,doc):
    routing(doc,('velocity','portamento_time',1.0),portamento_time=-10.0)
    r=convert(VitalPreset(doc),template)
    slot,=slots(r)
    assert slot['source']==[16,0] and slot['destModuleParamName']=='kParamPortamentoTime'
    assert plain(slot)['kParamAmount']==pytest.approx(100.0)
    # velocity 0.5 -> 2^-3 s in Vital -> (0.125 / 8)^(1/5) of Serum's control
    assert 1-slot['flex'][0]['yVals'][16]==pytest.approx((.125/8)**.2)
    assert any('velocity' in n['message'] for n in r.report.notes if n['kind']=='approximation')


def test_modulation_power_becomes_a_curve(template,doc):
    routing(doc,('macro_control_1','osc_1_pan',.5,0,4.0))
    slot,=slots(convert(VitalPreset(doc),template))
    curve=slot['flex'][0]
    assert curve['yVals']==[1.0,0.0] and curve['curveVals']==pytest.approx([curve_value(-4.0),.5])
    routing(doc,('macro_control_1','osc_1_pan',.5,1,4.0))
    curve=slots(convert(VitalPreset(doc),template))[0]['flex'][0]
    assert curve['xVals']==[0.0,.5,1.0] and curve['curveVals']==pytest.approx([curve_value(4.0),curve_value(-4.0),.5])


def test_stored_macro_is_folded_into_a_parameter_serum_cannot_modulate(template,doc):
    routing(doc,('macro_control_1','osc_1_unison_voices',1.0),macro_control_1=.6783,osc_1_unison_voices=1)
    r=convert(VitalPreset(doc),template)
    assert plain(r.preset.state['Oscillator0'])['kParamUnison']==11       # 1 + 0.6783 * 15 = 11.17
    assert not slots(r)
    assert any(n['kind']=='approximation' and 'folded' in n['message'] for n in r.report.notes)


def test_a_modulated_macro_is_not_folded(template,doc):
    routing(doc,('macro_control_1','osc_1_unison_voices',1.0),('velocity','macro_control_1',-1.0),macro_control_1=.6783)
    r=convert(VitalPreset(doc),template)
    assert plain(r.preset.state['Oscillator0'])['kParamUnison']==1


@pytest.mark.parametrize('macro,step',[(.19,7),(.21,7),(.69,7),(.71,8),(1.0,8)])
def test_folded_lfo_division_follows_vitals_bias(template,doc,macro,step):
    # Vital picks the division at floor(index + 0.3), including fractional fixed offsets.
    routing(doc,('macro_control_1','lfo_1_tempo',1/12),macro_control_1=macro,lfo_1_tempo=8,lfo_1_sync=1)
    params=plain(convert(VitalPreset(doc),template).preset.state['LFO0'])
    assert params['kParamRate']==pytest.approx(100*(step/14)**4)


@pytest.mark.parametrize('macro,step',[(.29,7),(.31,6)])
def test_folded_lfo_division_negative_offset_boundary(template,doc,macro,step):
    routing(doc,('macro_control_1','lfo_1_tempo',-1/12),macro_control_1=macro,lfo_1_tempo=8,lfo_1_sync=1)
    params=plain(convert(VitalPreset(doc),template).preset.state['LFO0'])
    assert params['kParamRate']==pytest.approx(100*(step/14)**4)


def test_level_modulation_follows_the_gain_on_the_oscillator(template,doc):
    # both level controls move on the square root of the amplitude, so the routing scales by sqrt(gain)
    routing(doc,('lfo_1','osc_1_level',.3),osc_1_level=.5,osc_1_unison_voices=7,osc_1_unison_detune=3)
    r=convert(VitalPreset(doc),template)
    gain=10**(unison.match(7,.8,1.0)[2]/20)
    assert plain(slots(r)[0])['kParamAmount']==pytest.approx(30*math.sqrt(gain))


def test_modulated_level_gets_headroom(template,doc):
    # base level 0, an LFO opening it fully, and a table that needs gain: the sources are lowered
    # and the master raised so the routing can reach the full level
    frame=[1.6*math.sin(2*math.pi*i/2048) for i in range(2048)]
    from conftest import wave
    doc['settings']['wavetables'][0]=wave(frame)
    routing(doc,('lfo_1','osc_1_level',1.0),osc_1_level=0.0,volume=4000.0)
    r=convert(VitalPreset(doc),template)
    assert plain(r.preset.state['Oscillator0'])['kParamVolume']==0
    assert plain(slots(r)[0])['kParamAmount']==pytest.approx(100.0)
    assert any(n['feature']=='levels' for n in r.report.notes)


def test_spectral_morph_is_rendered_into_the_table(template,doc):
    import numpy as np
    from conftest import wave
    saw=[2*(i/2048)-1 for i in range(2048)]
    doc['settings']['wavetables'][0]=wave(saw)
    doc['settings'].update(osc_1_spectral_morph_type=7,osc_1_spectral_morph_amount=.2)      # low pass at harmonic 4
    r=convert(VitalPreset(doc),template)
    table=np.array(list(r.preset.state['Oscillator0']['WTOsc0']['embeddedWTData']))[:2048]
    spectrum=np.abs(np.fft.rfft(table))
    assert spectrum[4]>100*spectrum[5] and spectrum[1]>spectrum[4]
    assert any(n['feature']=='osc_1_spectral_morph' and n['kind']=='approximation' for n in r.report.notes)
    # a macro on the amount keeps working: the table becomes a sweep and the macro moves the position
    routing(doc,('macro_control_1','osc_1_spectral_morph_amount',.5),macro_control_1=.4,osc_1_spectral_morph_amount=0.0)
    r=convert(VitalPreset(doc),template)
    osc=r.preset.state['Oscillator0']['WTOsc0']
    assert len(list(osc['embeddedWTData']))==256*2048 and plain(osc)['kParamTablePos']==1
    assert plain(slots(r)[0])['kParamAmount']==pytest.approx(50) and slots(r)[0]['destModuleParamName']=='kParamTablePos'
    # with the table position moving too, the macro is folded in where it stands: 0 + 0.4 * 0.5 = 0.2 again
    routing(doc,('macro_control_1','osc_1_spectral_morph_amount',.5),('lfo_1','osc_1_wave_frame',.2),macro_control_1=.4,osc_1_spectral_morph_amount=0.0)
    r=convert(VitalPreset(doc),template)
    table=np.array(list(r.preset.state['Oscillator0']['WTOsc0']['embeddedWTData']))
    assert len(table)==2048 and np.abs(np.fft.rfft(table))[4]>100*np.abs(np.fft.rfft(table))[5]


def test_moving_morph_amount_becomes_a_sweep_along_the_table(template,doc):
    import numpy as np
    from conftest import wave
    doc['settings']['wavetables'][0]=wave([2*(i/2048)-1 for i in range(2048)])
    routing(doc,('lfo_1','osc_1_spectral_morph_amount',.3),osc_1_spectral_morph_type=7,osc_1_spectral_morph_amount=.5)
    r=convert(VitalPreset(doc),template)
    osc=r.preset.state['Oscillator0']['WTOsc0']
    frames=np.array(list(osc['embeddedWTData'])).reshape(256,2048)
    assert plain(osc)['kParamTablePos']==pytest.approx(128.5)                # Serum plays frame floor(256 * 0.5)
    spectrum=np.abs(np.fft.rfft(frames[128]))                                 # which holds amount 128.5 / 256: 32.4 harmonics
    assert spectrum[32]>100*spectrum[34] and spectrum[33]==pytest.approx(.44*spectrum[32]*32/33,rel=.05)
    assert np.abs(np.fft.rfft(frames[0]))[2]<.01*np.abs(np.fft.rfft(frames[0]))[1]         # amount 0.002: the fundamental
    assert plain(slots(r)[0])['kParamAmount']==pytest.approx(30)
    assert any(n['feature']=='osc_1_spectral_morph' and 'laid out' in n['message'] for n in r.report.notes)
    # with the table position moving as well, the amount stays where it is stored
    routing(doc,('lfo_1','osc_1_spectral_morph_amount',.3),('lfo_2','osc_1_wave_frame',.3),osc_1_spectral_morph_type=7,osc_1_spectral_morph_amount=.2)
    r=convert(VitalPreset(doc),template)
    assert len(list(r.preset.state['Oscillator0']['WTOsc0']['embeddedWTData']))==2048
    assert any(n['feature']=='osc_1_spectral_morph_amount' and n['kind']=='unsupported' for n in r.report.notes)


def test_phase_warp_is_rendered_into_the_table(template,doc):
    import numpy as np
    from conftest import wave
    doc['settings']['wavetables'][0]=wave([math.sin(2*math.pi*i/2048) for i in range(2048)])
    doc['settings'].update(osc_1_distortion_type=1,osc_1_distortion_amount=.75)            # sync, ratio 2^(8 * 0.75 - 4) = 4
    r=convert(VitalPreset(doc),template)
    osc=r.preset.state['Oscillator0']['WTOsc0']
    spectrum=np.abs(np.fft.rfft(np.array(list(osc['embeddedWTData']))))
    assert spectrum[4]>1000*max(spectrum[1],spectrum[3],spectrum[5]) and 'kParamWarpMenu' not in plain(osc)
    # a moving amount: the table sweeps the warp and the routing moves the position
    routing(doc,('lfo_1','osc_1_distortion_amount',.25),osc_1_distortion_amount=.5)
    r=convert(VitalPreset(doc),template)
    osc=r.preset.state['Oscillator0']['WTOsc0']
    frames=np.array(list(osc['embeddedWTData'])).reshape(256,2048)
    assert plain(osc)['kParamTablePos']==pytest.approx(128.5) and 'kParamWarpMenu' not in plain(osc)
    assert np.abs(np.fft.rfft(frames[192]))[4]>10*np.abs(np.fft.rfft(frames[192]))[1]    # amount 0.752: about 4 cycles
    assert plain(slots(r)[0])['kParamAmount']==pytest.approx(25)
    # table position and warp both moving: Serum's own warp, as before
    routing(doc,('lfo_1','osc_1_distortion_amount',.25),('lfo_2','osc_1_wave_frame',.3),osc_1_distortion_amount=.5)
    r=convert(VitalPreset(doc),template)
    assert plain(r.preset.state['Oscillator0']['WTOsc0'])['kParamWarpMenu']=='kSync'


def test_modulation_envelopes_hold_the_square_of_serums_sustain(template,doc):
    # Serum's envelope rests at sustain^2 (0.25 -> 0.061, 0.5 -> 0.248, 0.75 -> 0.561 measured)
    doc['settings'].update(env_1_sustain=.36,env_2_sustain=.36,env_3_sustain=.25)
    r=convert(VitalPreset(doc),template)
    state=r.preset.state
    assert plain(state['Env0'])['kParamSustain']==pytest.approx(.36)          # the amplitude envelope is squared in Vital too
    assert plain(state['Env1'])['kParamSustain']==pytest.approx(.6) and plain(state['Env2'])['kParamSustain']==pytest.approx(.5)


def test_amplitude_envelope_as_a_source_is_bent_back(template,doc):
    routing(doc,('env_1','osc_1_tune',.2),('env_2','osc_1_pan',.2))
    r=convert(VitalPreset(doc),template)
    first,second=slots(r)[:2]
    curve=first['flex'][0]
    assert plain(first)['kParamMainCurveData']==1 and 'flex' not in second
    # Serum's source is Vital's envelope squared: a quarter in reads a half out (y counts from the top)
    assert curve['xVals'][16]==pytest.approx(.25) and curve['yVals'][16]==pytest.approx(.5)
    assert curve['xVals'][-1]==1 and curve['yVals'][-1]==0


def test_remap_curve_of_a_routing_becomes_the_slot_curve(template,doc):
    from vital2serum.curves import line_value
    from vital2serum.modulation import active_modulations
    # Vital's remap editor: points as x, y pairs with y = 0 at the top; here nothing until 7/8 of the way
    routing(doc,('macro_control_1','osc_1_tune',.5),('env_2','osc_1_pan',.4,1),macro_control_1=.3)
    doc['settings']['modulations'][0]['line_mapping']={'num_points':3,'points':[0,1,.875,1,1,0],'powers':[0,0],'smooth':False}
    doc['settings']['modulations'][1]['line_mapping']={'num_points':2,'points':[0,1,1,0],'powers':[0],'smooth':False}       # the default
    mods=active_modulations(VitalPreset(doc))
    assert mods[1].value(.3)==0 and mods[1].value(.9375)==pytest.approx(.5) and mods[2].points==()
    assert line_value(((0,1),(.5,1),(.5,.25),(1,0)),(0,0,0),.5)==pytest.approx(.75)          # a step: the later point holds
    doc['settings']['modulations'][1]['line_mapping']={'num_points':4,'points':[0,1,.5,1,1,0,1,1],'powers':[0,0,0],'smooth':False}
    assert active_modulations(VitalPreset(doc))[2].shaped(1.0)==1            # a closing step back to the start is not part of the curve
    doc['settings']['modulations'][1]['line_mapping']={'num_points':2,'points':[0,1,1,0],'powers':[0],'smooth':False}
    r=convert(VitalPreset(doc),template)
    first,second=slots(r)[:2]
    assert first['flex'][0]['xVals']==[0,.875,1] and first['flex'][0]['yVals']==[1,1,0] and 'flex' not in second
    # with a power as well the two are sampled together: the curve is looked up first, then bent
    doc['settings']['modulation_1_power']=4.0
    r=convert(VitalPreset(doc),template)
    curve=slots(r)[0]['flex'][0]
    assert len(curve['xVals'])>=33 and 1-curve['yVals'][curve['xVals'].index(.9375)]==pytest.approx(mods[1].value(.9375)*0+
           (1-math.exp(-4*.5))/(1-math.exp(-4)))


def test_fm_and_rm_between_oscillators(template,doc):
    from conftest import wave
    from vital2serum import fm
    sine=[math.sin(2*math.pi*i/2048) for i in range(2048)]
    doc['settings']['wavetables'][:2]=[wave(sine),wave(sine)]
    doc['settings'].update(osc_1_on=1,osc_2_on=1,osc_2_level=0,osc_1_distortion_type=7,osc_1_distortion_amount=.5)
    r=convert(VitalPreset(doc),template)
    osc=plain(r.preset.state['Oscillator0']['WTOsc0'])
    # Vital's FM is phase modulation, 12 pi a^2 rad; Serum's kPD_OSC gives 50.42 w^4 (kParamWarp runs 0..1)
    assert osc['kParamWarpMenu']=='kPD_OSC' and osc['kParamWarp']==pytest.approx((12*math.pi*.25/50.42)**.25)
    assert fm.modulator(1,7)==2 and fm.modulator(1,8)==3 and fm.modulator(3,7)==1
    # RM: Serum's sidebands are twice Vital's, so the warp is lower and the carrier level makes up the rest
    doc['settings'].update(osc_1_distortion_type=10,osc_1_distortion_amount=.6)
    r=convert(VitalPreset(doc),template)
    osc=plain(r.preset.state['Oscillator0']['WTOsc0'])
    assert osc['kParamWarpMenu']=='kRM_OSC' and osc['kParamWarp']==pytest.approx(.6/(.6+2.0064*.4))
    # a moving RM amount: the modulator's table is halved instead, and the warp is Vital's amount
    routing(doc,('lfo_1','osc_1_distortion_amount',.2))
    r=convert(VitalPreset(doc),template)
    assert plain(r.preset.state['Oscillator0']['WTOsc0'])['kParamWarp']==pytest.approx(.6)
    assert max(r.preset.state['Oscillator1']['WTOsc1']['embeddedWTData'])==pytest.approx(1/2.0064,rel=1e-3)
    # the modulator switched off: no warp, and Vital's RM leaves 1 - amount of the carrier
    doc['settings'].update(osc_2_on=0)
    r=convert(VitalPreset(doc),template)
    assert 'kParamWarpMenu' not in plain(r.preset.state['Oscillator0']['WTOsc0'])


def test_filter_cutoff_resonance_and_key_tracking(template,doc):
    from vital2serum.effects import filter_fit, filter_type, serum_cutoff
    # Serum's cutoff control: 137.86 semitones from note -0.39 (with note / 135 it sat 0.4 to 1.9 semitones high)
    assert serum_cutoff(72)==pytest.approx(.5251,abs=1e-4)
    # resonance with the same response shape, and Serum's level above Vital's (analog 12 dB at 0.5: 34 and 6 dB)
    assert filter_fit(('analog',12),.5)==(34,6.0) and filter_fit(('ladder',24),.3)==(48,-2.2)
    assert filter_fit(('digital',12),.35)==pytest.approx((12.5,.8))
    assert filter_type(0,1,0)[:2]==('LBH24',0) and filter_type(2,0,0)[0]=='MgL12' and filter_type(4,0,0)[0]=='LadderEMS'
    assert filter_type(0,2,1)[:2]==('LNH24',50) and filter_type(0,4,1.5)[:2]==('BPN24',42)        # 24 dB shapes
    routing(doc,('lfo_1','filter_1_resonance',.5),filter_1_on=1,filter_1_cutoff=72.0,filter_1_resonance=.3,filter_1_keytrack=1.0)
    r=convert(VitalPreset(doc),template)
    f=plain(r.preset.state['VoiceFilter0'])
    # the switch follows the keyboard from note 32 and Vital from 60: 28 semitones go into the cutoff
    assert f['kParamKeyTrack']==1 and f['kParamFreq']==pytest.approx(serum_cutoff(72-28)) and f['kParamReso']==12
    assert slots(r)[0]['destModuleParamName']=='kParamReso' and 'flex' in slots(r)[0]        # resonance has its own scale
    # partial tracking: the Note source (note / 127, centred on 63.5) carries it
    doc['settings'].update(filter_1_keytrack=.5); doc['settings']['modulations']=[]
    r=convert(VitalPreset(doc),template)
    f=plain(r.preset.state['VoiceFilter0']); slot=slots(r)[0]
    assert 'kParamKeyTrack' not in f and f['kParamFreq']==pytest.approx(serum_cutoff(72+.5*3.5))
    assert slot['source']==[17,0] and plain(slot)['kParamAmount']==pytest.approx(50*127/137.86) and plain(slot)['kParamBipolar']==1


def test_fx_rack_filters_take_modulation_and_key_tracking(template,doc):
    from vital2serum.effects import serum_cutoff
    # the FX rack is addressed by position; Serum's FX filter has no key tracking, the Note source stands in
    routing(doc,('lfo_1','filter_fx_cutoff',.25),('macro_control_1','filter_fx_mix',.5),filter_fx_on=1,filter_fx_cutoff=72.0,
            filter_fx_keytrack=1.0,filter_fx_mix=.4)
    r=convert(VitalPreset(doc),template)
    rack=r.preset.state['FXRack0']['FX']
    index=next(i for i,e in enumerate(rack) if 'FXFilter' in e)
    assert plain(rack[index]['FXFilter'])['kParamFreq']==pytest.approx(serum_cutoff(72+3.5))
    note,lfo,macro=slots(r)[:3]
    assert note['source']==[17,0] and note['destModuleTypeString']=='FXFilter' and note['destModuleID']==index
    assert plain(note)['kParamAmount']==pytest.approx(100*127/137.86) and plain(note)['kParamBipolar']==1
    assert lfo['destModuleParamName']=='kParamFreq' and plain(lfo)['kParamAmount']==pytest.approx(25*128/137.86)
    assert macro['destModuleParamName']=='kParamWet' and 'flex' in macro            # the wet share is not Vital's mix
    # A native serial chain keeps the second stage's per-voice routings.
    doc['settings'].update(filter_fx_on=0,filter_1_on=1,filter_2_on=1,filter_2_filter_input=1)
    routing(doc,('env_2','filter_2_cutoff',.3),filter_fx_on=0,filter_1_on=1,filter_2_on=1,filter_2_filter_input=1)
    r=convert(VitalPreset(doc),template)
    assert slots(r)[0]['destModuleTypeString']=='VoiceFilter' and slots(r)[0]['destModuleID']==1


def test_filter_mix_accounts_for_serums_output_level(template,doc):
    # Output level follows the voice mix, and its drive also amplifies the dry path.
    doc['settings'].update(filter_1_on=1,filter_1_resonance=.5,filter_1_mix=.5)
    f=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])
    doc['settings'].update(filter_1_mix=1.0)
    full=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])
    gain=(2*full['kParamLevelOut'])**2
    dry_gain=(1+full['kParamDrive']/100)**4
    amplitude=.5/dry_gain+.5*gain
    assert f['kParamWet']==pytest.approx(100*.5*gain/amplitude)
    assert (2*f['kParamLevelOut'])**2==pytest.approx(amplitude)


def test_chorus_and_reverb_wet_levels(template,doc):
    from vital2serum.effects import wet_mix
    # Vital's four-voice chorus is 5.9 dB louder than Serum's at full wet: all wet, +5.9 dB out
    assert wet_mix(1.0,5.9)==pytest.approx((100.0,5.9)) and wet_mix(0.0,5.9)==pytest.approx((0.0,0.0))
    wet,gain=wet_mix(.5,0.0)                                       # equal gains: half the power each way
    assert wet==pytest.approx(50.0) and gain==pytest.approx(20*math.log10(2**.5))
    doc['settings'].update(chorus_on=1,chorus_dry_wet=1.0,chorus_voices=4,reverb_on=1,reverb_dry_wet=1.0,reverb_decay_time=3.0,reverb_size=.5)
    r=convert(VitalPreset(doc),template)
    chorus,reverb=rack(r,'FXChorus')[0],rack(r,'FXReverb')[0]
    assert chorus['kParamWet']==pytest.approx(100) and (2*chorus['kParamLevelOut'])**2==pytest.approx(10**(5.9/20))
    assert reverb['kParamType']=='kHall' and reverb['kParamWet']==pytest.approx(100)


def test_stereo_routing_becomes_a_utility_at_the_end_of_the_rack(template,doc):
    # routing 0 sums the channels: a centred voice comes out 3.01 dB louder and the sides vanish (measured)
    doc['settings'].update(stereo_routing=0.0,chorus_on=1)
    r=convert(VitalPreset(doc),template)
    last=r.preset.state['FXRack0']['FX'][-1]
    assert last['type']==12 and plain(last['FXUtils'])['kParamWidth']==pytest.approx(0,abs=1e-9)
    assert (2*plain(last['FXUtils'])['kParamLevelOut'])**2==pytest.approx(2**.5)
    doc['settings'].update(stereo_routing=0.5)             # +2.32 dB in the middle, sides at tan(pi/8 ...) of it
    util=plain(convert(VitalPreset(doc),template).preset.state['FXRack0']['FX'][-1]['FXUtils'])
    assert 20*math.log10((2*util['kParamLevelOut'])**2)==pytest.approx(2.32,abs=.01) and util['kParamWidth']==pytest.approx(100*.5412/1.3066,abs=.01)
    doc['settings'].update(stereo_routing=1.0)
    assert not rack(convert(VitalPreset(doc),template),'FXUtils')


def test_velocity_tracking_goes_through_the_voice_amplitude(template,doc):
    doc['settings'].update(velocity_track=1.0)
    r=convert(VitalPreset(doc),template)
    slot=slots(r)[0]
    assert plain(r.preset.state['Global0'])['kParamVoiceAmp']==0 and slot['source']==[16,0]
    assert slot['destModuleParamName']=='kParamVoiceAmp' and plain(slot)['kParamAmount']==50          # the control is half the amplitude
    curve=slot['flex'][0]
    assert 1-curve['yVals'][16]==pytest.approx(.25) and 1-curve['yVals'][-1]==1                        # velocity squared
    doc['settings'].update(velocity_track=.5)
    curve=slots(convert(VitalPreset(doc),template))[0]['flex'][0]
    assert 1-curve['yVals'][16]==pytest.approx(.75**2)                                                 # -4.95 dB at velocity 64 (measured)
    doc['settings'].update(velocity_track=0.0)
    assert 'kParamVoiceAmp' not in plain(convert(VitalPreset(doc),template).preset.state['Global0'])


def test_fixed_pitch_oscillator_and_always_glide(template,doc):
    # Vital without key tracking: 130.78 Hz (C3) on every key; Serum's switch gives note 0, so 48 semitones are added
    doc['settings'].update(osc_1_midi_track=0,osc_1_transpose=-19,portamento_force=1)
    r=convert(VitalPreset(doc),template)
    osc=plain(r.preset.state['Oscillator0'])
    assert osc['kParamPitchTrack']==0 and osc['kParamOctave']==4 and osc['kParamPitch']==0
    assert osc['kParamCoarsePit']==-19
    assert plain(r.preset.state['Global0'])['kParamPortaAlways']==1


def test_settings_added_by_vital_1_6_are_reported(template,doc):
    routing(doc,('lfo_1','osc_1_tune',.5),modulation_1_ramp_up=-2.0,modulation_1_ramp_down=-10.0)
    doc['tuning']={'default':False,'scale':[0.0,1.1]}
    r=convert(VitalPreset(doc),template)
    assert any(n['feature']=='modulation_1_ramp' and n['kind']=='unsupported' for n in r.report.notes)
    assert any(n['feature']=='tuning' and n['kind']=='unsupported' for n in r.report.notes)
    assert plain(slots(r)[0])['kParamAmount']==pytest.approx(50)                # the routing itself is still written
    doc['settings'].update(modulation_1_ramp_up=-10.0); doc['tuning']={'default':True}
    r=convert(VitalPreset(doc),template)
    assert not any(n['feature'] in ('modulation_1_ramp','tuning') for n in r.report.notes)


# --- LFO ----------------------------------------------------------------------------------------
def test_lfo_smoothing_and_depth_correction(template,doc):
    # Vital's default smoothing (1.6 ms) passes 87 % of a 56 Hz LFO, Serum's minimum (2.5 ms) 75 %.
    routing(doc,('lfo_1','osc_1_tune',.1,1),lfo_1_sync=0,lfo_1_frequency=math.log2(56.262),lfo_1_smooth_time=-7.5)
    r=convert(VitalPreset(doc),template)
    assert plain(r.preset.state['LFO0'])['kParamSmooth']==.5
    passed=lambda tau: 1/math.sqrt(1+(2*math.pi*56.262*tau)**2)
    assert plain(slots(r)[0])['kParamAmount']==pytest.approx(10*passed(2**-7.5/math.log(32))/passed(.0025))
    doc['settings']['lfo_1_smooth_time']=-10.0
    r=convert(VitalPreset(doc),template)
    assert 'kParamSmooth' not in plain(r.preset.state['LFO0']) and plain(slots(r)[0])['kParamAmount']==pytest.approx(10)


def test_lfo_fade_is_not_written(template,doc):
    doc['settings'].update(lfo_1_fade_time=2.0)
    r=convert(VitalPreset(doc),template)
    assert 'kParamRise' not in plain(r.preset.state['LFO0'])
    assert any(n['feature']=='lfo_1_fade_time' and n['kind']=='info' for n in r.report.notes)


def test_lfo_delay_units(template,doc):
    doc['settings'].update(lfo_1_sync=0,lfo_1_frequency=2.0,lfo_1_delay_time=.75,lfo_2_sync=1,lfo_2_delay_time=.25,beats_per_minute=2.0)
    state=convert(VitalPreset(doc),template).preset.state
    assert plain(state['LFO0'])['kParamDelay']==.75                       # seconds while free-running
    assert plain(state['LFO1'])['kParamDelay']==pytest.approx(6.3/3.5)    # an eighth note at 120 BPM
    assert synced_delay(2.0,2.0)==pytest.approx(9.3/3.5) and synced_delay(0,2.0)==0


def test_curve_value_matches_measured_points():
    # Serum: half-way level 4 (1 - c)^3; fitted exponents 2.14 at 0.6, 5.41 at 0.75, 11.2 at 0.9
    for c,power in ((.6,2.13),(.75,5.42),(.9,11.03)):
        assert curve_value(power)==pytest.approx(c,abs=2e-3) and curve_value(-power)==pytest.approx(1-c,abs=2e-3)
    assert curve_value(0)==.5 and power_scale(.5,0)==.5


# --- envelopes ----------------------------------------------------------------------------------
def test_envelope_stage_curves(template,doc):
    doc['settings'].update(env_2_attack_power=-1.7573,env_2_decay_power=-2.0,env_2_release_power=3.0,
                           env_1_attack_power=0.0,env_1_decay_power=0.0,env_1_sustain=0.0,env_1_release_power=0.0)
    state=convert(VitalPreset(doc),template).preset.state
    env=plain(state['Env1'])
    assert env['kParamCurve1']==pytest.approx(100*curve_value(-1.7573))       # slow start, as Vital draws it
    assert env['kParamCurve2']==pytest.approx(100*curve_value(2.0)) and env['kParamCurve3']==pytest.approx(100*curve_value(-3.0))
    amp=plain(state['Env0'])
    # Vital squares the amplitude envelope: a straight attack sounds like t^2, a straight fall like 1 - (1 - t)^2
    assert amp['kParamCurve1']==pytest.approx(100*curve_value(-2.08),abs=.3)
    assert amp['kParamCurve2']==pytest.approx(100*curve_value(2.08),abs=.3)==pytest.approx(amp['kParamCurve3'],abs=.3)


# --- unison -------------------------------------------------------------------------------------
def test_unison_model_reference_points():
    assert unison.match(1,.8,1.0)==(None,None,0.0)
    assert unison.match(1,.8,0.0)[2]==pytest.approx(3.01,abs=.01)           # the lone voice adds in phase
    assert unison.match(2,.8,1.0)[1:]==pytest.approx((100.0,1.87),abs=.01)
    blend,width,gain=unison.match(7,.8,1.0)
    assert (blend,width)==(55.0,100.0) and gain==pytest.approx(3.05,abs=.05)
    assert unison.match(11,.8,1.0)[2]==pytest.approx(2.22,abs=.05)
    assert unison.serum_detune(4.4721)**1.25==pytest.approx(.2,abs=1e-4)     # 20 % of the range in both
    assert unison.voice_positions(7)==pytest.approx([1/3,2/3,1]) and unison.voice_positions(8)==pytest.approx([1/7,3/7,5/7,1])


@pytest.mark.parametrize('power,mode',[(-5,'kDetuneInv'),(0,None),(1.5,'kDetuneSuper'),(5,'kDetuneExp')])
def test_unison_settings(template,doc,power,mode):
    doc['settings'].update(osc_1_unison_voices=7,osc_1_unison_detune=4.4721,osc_1_detune_power=power,
                           osc_1_unison_blend=.5,osc_1_stereo_spread=.5,osc_1_detune_range=4.0)
    p=plain(convert(VitalPreset(doc),template).preset.state['Oscillator0'])
    blend,width,_=unison.match(7,.5,.5)
    assert p.get('kParamDetuneMode')==mode
    assert p['kParamDetuneWid']==blend and p['kParamUnisonStereo']==pytest.approx(width)==pytest.approx(34.1,abs=.2)
    assert p['kParamUnisonRange']==4.0 and p['kParamDetune']==pytest.approx(.2**.8,rel=1e-4)


def test_hot_source_restores_common_gain_before_voice_processing(template,doc):
    doc['settings'].update(osc_1_level=1.0,osc_1_unison_voices=3,osc_1_unison_detune=3,volume=4000.0)
    quiet=dict(doc['settings'],osc_1_unison_voices=1)
    base=plain(convert(VitalPreset(dict(doc,settings=quiet)),template).preset.state['Global0'])['kParamMasterVolume']
    r=convert(VitalPreset(doc),template)
    gain=10**(unison.match(3,.8,1.0)[2]/20)
    assert plain(r.preset.state['Oscillator0'])['kParamVolume']==pytest.approx(1.0)
    assert plain(r.preset.state['Global0'])['kParamVoiceAmp']==pytest.approx(gain/2)
    assert plain(r.preset.state['Global0'])['kParamMasterVolume']==pytest.approx(base)
    doc['settings'].update(distortion_on=1,distortion_drive=6.0)
    r=convert(VitalPreset(doc),template)
    assert plain(r.preset.state['Global0'])['kParamMasterVolume']==pytest.approx(base)
    assert rack(r,'FXEQ')==[]
    assert plain(r.preset.state['Global0'])['kParamVoiceAmp']==pytest.approx(gain/2)
    assert r.preset.state['FXRack0']['FX'][0]['type']==0
    assert 'FXDistortion' in r.preset.state['FXRack0']['FX'][0]


# --- effects ------------------------------------------------------------------------------------
def test_eq_bands_translate_exactly(template,doc):
    doc['settings'].update(eq_on=1,eq_low_mode=1,eq_low_cutoff=60.0,eq_low_resonance=.1329,
                           eq_high_gain=-12.0,eq_high_cutoff=84.0,eq_high_resonance=.6,
                           eq_band_gain=6.0,eq_band_cutoff=72.0,eq_band_resonance=.4473)
    first,second=rack(convert(VitalPreset(doc),template),'FXEQ')
    assert (first['kParamType1'],first['kParamFreq1'])==(2.0,pytest.approx(261.626,abs=.01))
    assert first['kParamReso1']==pytest.approx(serum_eq_reso(vital_eq_q(.1329)))==pytest.approx(38.15,abs=.02)   # Q stays near 0.5 below resonance 0.2
    amplitude=10**(-12/40)
    assert (first['kParamType2'],first['kParamGain2'])==(0.0,-12.0)
    assert first['kParamFreq2']==pytest.approx(1046.502/math.sqrt(amplitude),abs=.01)
    assert first['kParamReso2']==pytest.approx(serum_shelf_reso(1.223))
    # the third band needs a second module; Vital's peak at Q is Serum's symmetric peak at Q / sqrt(gain)
    assert (second['kParamType1'],second['kParamGain1'],second['kParamFreq1'])==(1.0,6.0,pytest.approx(523.251,abs=.01))
    assert second['kParamReso1']==pytest.approx(serum_eq_reso(.625/10**(6/40)))


def test_single_high_band_uses_serums_high_side(template,doc):
    doc['settings'].update(eq_on=1,eq_high_mode=1,eq_high_cutoff=100.0)
    eq,=rack(convert(VitalPreset(doc),template),'FXEQ')
    assert eq['kParamType2']==2.0 and 'kParamType1' not in eq


def test_eq_laws():
    assert vital_eq_q(0)==pytest.approx(.5) and vital_eq_q(.9)==pytest.approx(8.737) and vital_eq_q(1)==pytest.approx(16)
    assert .1+18.9*(serum_eq_reso(2.548)/100)**4==pytest.approx(2.548) and serum_eq_reso(.05)==0
    assert .4+1.741*(serum_shelf_reso(1.291)/100)**3==pytest.approx(1.291) and serum_shelf_reso(9)==pytest.approx(100)


def test_distortion_tables():
    assert soft_clip(0.0)==('kTube',34.5,2.92)
    mode,drive,gain=soft_clip(10.1162)
    assert mode=='kTube' and 80.5<drive<85.0 and 3.64<gain<3.81
    assert soft_clip(15.0)[0]=='kSoftSat' and soft_clip(30.0)==('kDiode1',100.0,2.16)
    # hard clip: Serum's gain ahead of the clip is 0.5 + 2 (drive / 100)^3 and its ceiling 0.8278 of Vital's
    mode,drive,gain=hard_clip(6.0)
    assert mode=='kHardClip' and .5+2*(drive/100)**3==pytest.approx(10**(6/20)*.8278) and gain==pytest.approx(1.64,abs=.01)
    assert hard_clip(HARD_CLIP_MAX_DRIVE_DB)[1]==pytest.approx(100)
    assert hard_clip(-20.0)[1:]==(0.0,pytest.approx(-20+6.02,abs=.01))


def test_hard_clip_beyond_serums_drive_gets_a_gain_stage(template,doc):
    doc['settings'].update(distortion_on=1,distortion_type=1,distortion_drive=20.0)
    r=convert(VitalPreset(doc),template)
    names=[list(e)[0] for e in r.preset.state['FXRack0']['FX']]
    assert names==['FXEQ','FXDistortion']
    stage,=rack(r,'FXEQ');dist,=rack(r,'FXDistortion')
    assert stage['kParamLevelOut']==pytest.approx(level_out(20.0-HARD_CLIP_MAX_DRIVE_DB))
    assert dist['kParamMode']=='kHardClip' and dist['kParamDrive']==pytest.approx(100)
