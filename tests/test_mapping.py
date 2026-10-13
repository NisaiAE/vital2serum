import copy
import math
import itertools
import random
import pytest
from vital2serum.errors import ConversionError
from vital2serum.effects import decode_order,EFFECTS,DELAY_KNOBS
from vital2serum import unison
from vital2serum.curves import curve_value
from vital2serum.mapping import convert
from vital2serum.modulation import learn_ids,KNOWN_IDS
from vital2serum.serum import SerumPreset,plain
from vital2serum.vital import VitalPreset
from conftest import wave


def test_core_mapping(template,doc):
    before=copy.deepcopy(template.state)
    doc['settings'].update(osc_1_transpose=19,osc_1_tune=.12,osc_1_pan=-.4,
        osc_1_level=.6,osc_1_phase=.125,osc_1_random_phase=0,
        osc_1_unison_voices=3,osc_1_unison_detune=5,env_1_attack=.5,env_1_decay=1.2,
        env_1_release=.75,env_1_sustain=.4,macro_control_1=.25,polyphony=1,legato=1)
    result=convert(VitalPreset(doc),template)
    p=plain(result.preset.state['Oscillator0'])
    assert p['kParamOctave']==1 and p['kParamPitch']==7
    assert p['kParamFine']==12 and p['kParamPan']==-20
    # quadratic level -> linear gain, plus the measured level difference of a three-voice stack
    blend,width,gain_db=unison.match(3,.8,1.0)
    assert p['kParamVolume']==pytest.approx(.36*10**(gain_db/20)) and p['kParamUnison']==3
    assert p['kParamDetuneWid']==blend and p['kParamUnisonStereo']==width==100
    # Vital spreads the outer voices by (detune/10)^2 of the range, Serum by detune^1.25
    assert p['kParamDetune']==pytest.approx(.5**1.6)
    wt=result.preset.state['Oscillator0']['WTOsc0']
    assert plain(wt)['kParamInitialPhase']==225
    assert plain(wt)['kParamRandomPhase']==0
    assert wt['numFrames']==2048
    env=plain(result.preset.state['Env0'])
    assert env['kParamAttack']==.5**4 and env['kParamRelease']==.75**4
    assert env['kParamSustain']==.4
    assert plain(result.preset.state['Macro0'])['kParamValue']==25
    assert result.preset.state['Macro0']['name']=='Colour'
    assert plain(result.preset.state['Global0'])['kParamMonoToggle']==1
    assert template.state==before
    assert SerumPreset.from_bytes(result.preset.to_bytes()).state==result.preset.state


def test_three_oscillators(template,doc):
    doc['settings'].update(osc_2_on=1,osc_3_on=1,wavetables=[wave(),wave(),wave()])
    r=convert(VitalPreset(doc),template)
    assert all(plain(r.preset.state[f'Oscillator{i}'])['kParamEnable']==1 for i in range(3))
    assert all(len(r.preset.state[f'Oscillator{i}'][f'WTOsc{i}']['embeddedWTData'])==2048 for i in range(3))
    r=convert(VitalPreset(doc),template,embed=False)
    assert len(r.assets)==1  # identical WAV is content-addressed and deduplicated
    assert plain(r.preset.state['Oscillator4'])['kParamEnable']==0

@pytest.mark.parametrize('volume',[0,1000,3000,5473.0404,7399.4404])
def test_master_inverse(template,doc,volume):
    doc['settings']['volume']=volume
    result=convert(VitalPreset(doc),template)
    master=plain(result.preset.state['Global0'])['kParamMasterVolume']
    if volume==0: assert master==0
    else:
        vital_db=math.sqrt(volume)-80
        serum_db=20*math.log10(master)+3.0103          # Serum's displayed master level
        if master==1:
            assert vital_db>serum_db
            output=plain(result.preset.state['FXRack0']['FX'][-1]['FXEQ'])
            serum_db+=20*math.log10(4*output['kParamLevelOut']**2)
            assert serum_db==pytest.approx(vital_db,abs=1e-3)
            assert 'master_output_gain' in result.report.mapped
            assert not any(n['kind']=='clipped' and n['feature']=='master_volume' for n in result.report.notes)
        else:
            assert serum_db==pytest.approx(vital_db,abs=1e-3)

@pytest.mark.parametrize('tempo',range(1,13))
def test_lfo_sync_ladder(template,doc,tempo):
    doc['settings'].update(lfo_1_tempo=tempo,lfo_1_sync=1)
    params=plain(convert(VitalPreset(doc),template).preset.state['LFO0'])
    assert params['kParamRate']==pytest.approx(100*((tempo-1)/14)**4)
    assert params['kParamBeatSync']==1
    assert 'kParamTriplets' not in params and 'kParamDotted' not in params
    # The curve LFO is Serum's default type: writing a type selects the path LFO, which never moves.
    assert 'kParamType' not in params


@pytest.mark.parametrize('tempo',range(2,13))
def test_lfo_triplet_and_dotted_sit_between_ladder_steps(template,doc,tempo):
    # Measured on Serum 2.1.5: with TRIP on the triplet of step k is at k + 2/3, with DOT on the
    # dotted value is at k - 2/3. The switch alone does not change the rate.
    doc['settings'].update(lfo_1_tempo=tempo,lfo_1_sync=3,lfo_2_tempo=tempo,lfo_2_sync=2)
    state=convert(VitalPreset(doc),template).preset.state
    trip,dot=plain(state['LFO0']),plain(state['LFO1'])
    assert trip['kParamRate']==pytest.approx(100*((tempo-1+2/3)/14)**4) and trip['kParamTriplets']==1
    assert dot['kParamRate']==pytest.approx(100*((tempo-1-2/3)/14)**4) and dot['kParamDotted']==1
    assert 'kParamDotted' not in trip and 'kParamTriplets' not in dot


def test_lfo_free_rate_is_hz_with_tenfold_switch(template,doc):
    doc['settings'].update(lfo_1_sync=0,lfo_1_frequency=math.log2(56.25),lfo_2_sync=0,lfo_2_frequency=math.log2(240.0))
    state=convert(VitalPreset(doc),template).preset.state
    slow,fast=plain(state['LFO0']),plain(state['LFO1'])
    assert slow['kParamBeatSync']==0 and slow['kParamRate']==pytest.approx(56.25) and 'kParamRate10x' not in slow
    assert fast['kParamRate10x']==1 and fast['kParamRate']==pytest.approx(24.0)


@pytest.mark.parametrize('sync_type,mode',[(0,None),(1,'Free'),(2,'Envelope')])
def test_lfo_modes(template,doc,sync_type,mode):
    doc['settings'].update(lfo_1_sync_type=sync_type)
    params=plain(convert(VitalPreset(doc),template).preset.state['LFO0'])
    assert params.get('kParamMode')==mode


def test_lfo_shape_units(template,doc):
    # Same axes as Vital (y = 0 is the top). The last entry closes the loop; numPoints excludes it.
    doc['settings']['lfos']=[{'num_points':3,'points':[0,.1,.3,.8,1,.1],'powers':[6,-6,0]}]
    m=convert(VitalPreset(doc),template).preset.state['LFO0']
    assert m['curveData']['xVals']==[0,.3,1]
    assert m['curveData']['yVals']==pytest.approx([.1,.8,.1])
    assert m['curveData']['curveVals']==pytest.approx([curve_value(6),curve_value(-6),.5])
    assert m['curveData']['numPoints']==2 and m['curveData']['loopbackPointNum']==481


def test_lfo_shape_that_does_not_close_gets_a_wrap_point(template,doc):
    # Vital's rising saw: bottom at the start, top at the end. Serum stores the end level first.
    doc['settings']['lfos']=[{'num_points':2,'points':[0,1,1,0],'powers':[3,0]}]
    cd=convert(VitalPreset(doc),template).preset.state['LFO0']['curveData']
    assert cd['xVals']==[0,0,1] and cd['yVals']==[0,1,0]
    assert cd['curveVals']==pytest.approx([.5,curve_value(3),.5]) and cd['numPoints']==2


def test_lfo_smooth_shape_is_sampled(template,doc):
    doc['settings']['lfos']=[{'num_points':3,'points':[0,1,.5,0,1,1],'powers':[0,0,0],'smooth':True}]
    cd=convert(VitalPreset(doc),template).preset.state['LFO0']['curveData']
    assert cd['numPoints']==16 and len(cd['xVals'])==17
    # a quarter of the way up the first segment the eased curve has covered sin^2(pi/8) of it
    assert cd['xVals'][2]==pytest.approx(.125) and cd['yVals'][2]==pytest.approx(1-math.sin(math.pi/8)**2)
    assert set(cd['curveVals'])=={.5}


def test_modulation_percent_not_normalized(template,doc):
    doc['settings'].update(modulations=[{'source':'lfo_1','destination':'osc_1_level'}],
                           modulation_1_amount=.25,modulation_1_bipolar=1)
    r=convert(VitalPreset(doc),template)
    slot=r.preset.state['ModSlot0']
    assert slot['source']==[6,0]
    assert slot['destModuleParamID']==1
    assert slot['destModuleParamName']=='kParamVolume'
    assert plain(slot)['kParamAmount']==25  # NOT 0.25
    assert plain(slot)['kParamBipolar']==1


def test_transpose_modulation_uses_the_coarse_pitch_control(template,doc):
    # Serum's semitone control stops at +-12; the coarse control spans 128 semitones (measured),
    # so Vital's 48 semitones here are 37.5 % of it and nothing is clipped.
    doc['settings'].update(modulations=[{'source':'macro_control_1','destination':'osc_1_transpose'}],modulation_1_amount=.5)
    r=convert(VitalPreset(doc),template)
    slot=r.preset.state['ModSlot0']
    assert slot['destModuleParamName']=='kParamCoarsePit' and slot['destModuleParamID']==6
    assert plain(slot)['kParamAmount']==pytest.approx(37.5)
    assert not any(x['kind']=='clipped' for x in r.report.notes)


def test_amount_beyond_the_control_range_is_clipped(template,doc):
    doc['settings'].update(modulations=[{'source':'macro_control_1','destination':'filter_1_cutoff'}],modulation_1_amount=1.0,
                           filter_1_on=1)
    r=convert(VitalPreset(doc),template)
    # Vital's cutoff range is 128 semitones, Serum's control 137.86: the scale is below one and fits.
    assert plain(r.preset.state['ModSlot0'])['kParamAmount']==pytest.approx(100*128/137.86)


def test_unknown_ids_are_not_guessed(template,doc):
    doc['settings'].update(modulations=[{'source':'macro_control_1','destination':'env_1_sustain'}],modulation_1_amount=.2)
    without={k:n for k,n in KNOWN_IDS.items() if k!=('Env','kParamSustain')}
    r=convert(VitalPreset(doc),template,destination_ids=without)
    assert 'source' not in r.preset.state['ModSlot0'] or not any(r.preset.state['ModSlot0']['source'])
    assert any('Numeric Serum' in x['message'] for x in r.report.notes)
    r=convert(VitalPreset(doc),template,destination_ids={**without,('Env','kParamSustain'):123})
    # Synthetic learned ID used only to exercise plumbing.
    assert r.preset.state['ModSlot0']['destModuleParamID']==123
    # The built-in table carries the number read from 188 routings in real presets.
    assert convert(VitalPreset(doc),template).preset.state['ModSlot0']['destModuleParamID']==3


def test_pan_uses_library_id(template,doc):
    doc['settings'].update(modulations=[{'source':'macro_control_1','destination':'osc_1_pan'}],modulation_1_amount=.2)
    slot=convert(VitalPreset(doc),template).preset.state['ModSlot0']
    assert slot['destModuleParamName']=='kParamPan' and slot['destModuleParamID']==2


def test_learn_conflicts(template,tmp_path):
    old=copy.deepcopy(template);old.meta['productVersion']='2.0.11'
    old.state['ModSlot0']={'destModuleTypeString':'Oscillator','destModuleParamName':'kParamDetune','destModuleParamID':10}
    new=copy.deepcopy(template);new.meta['productVersion']='2.0.12'
    new.state['ModSlot0']={'destModuleTypeString':'Oscillator','destModuleParamName':'kParamDetune','destModuleParamID':26}
    paths=[]
    for name,p in (('old',old),('new',new)):
        path=tmp_path/f'{name}.SerumPreset';path.write_bytes(p.to_bytes());paths.append(path)
    warnings=[]
    ids=learn_ids(paths,warn=warnings.append)
    assert ids[('Oscillator','kParamDetune')]==26
    assert any('several IDs' in w for w in warnings)


def test_dual_filters_are_chained(template,doc):
    doc['settings'].update(filter_1_on=1,filter_2_on=1,osc_1_destination=2,filter_1_model=2,
                           filter_1_style=1,filter_1_cutoff=67.5,filter_2_cutoff=40.5,filter_2_filter_input=1)
    r=convert(VitalPreset(doc),template)
    f=plain(r.preset.state['VoiceFilter0'])
    assert f['kParamFreq']==pytest.approx((67.5+.39)/137.86) and f['kParamType']=='MgL24'
    # Sending one source to both filters retains the FX-rack approximation.
    assert plain(r.preset.state['VoiceFilter1'])['kParamEnable']==0
    first=r.preset.state['FXRack0']['FX'][0]
    assert first['type']==8 and plain(first['FXFilter'])['kParamFreq']==pytest.approx((40.5+.39)/137.86)
    assert 'filter_2->FXFilter' in r.report.mapped
    assert any(n['feature']=='filter_routing' for n in r.report.notes)


def test_parallel_filters_are_noted(template,doc):
    doc['settings'].update(filter_1_on=1,filter_2_on=1,filter_2_filter_input=0,osc_1_destination=2)
    r=convert(VitalPreset(doc),template)
    assert any(n['feature']=='filter_routing' for n in r.report.notes)


def test_single_filter_keeps_balance_default(template,doc):
    doc['settings'].update(filter_1_on=1,filter_2_on=0)
    r=convert(VitalPreset(doc),template)
    assert 'kParamFilterBalance' not in plain(r.preset.state['RoutingSlot0'])
    assert not any(n['feature']=='filter_routing' for n in r.report.notes)


def test_filter_level_compensates_resonance(template,doc):
    doc['settings'].update(filter_1_on=1,filter_1_model=0,filter_1_style=0,filter_1_resonance=1.0)
    loud=plain(convert(VitalPreset(copy.deepcopy(doc)),template).preset.state['VoiceFilter0'])['kParamLevelOut']
    doc['settings']['filter_1_resonance']=0.0
    quiet=plain(convert(VitalPreset(doc),template).preset.state['VoiceFilter0'])['kParamLevelOut']
    assert loud < quiet <= 0.55        # more resonance -> more attenuation, unity is 0.5


def test_dirty_template_rejected(template,vital):
    template.state['FXRack0']['FX']=[{'type':4}]
    with pytest.raises(ConversionError,match='effects'): convert(vital,template)


def test_filter_env_unknown_feature_report(template,doc):
    doc['settings'].update(osc_1_frame_spread=20,env_1_delay=.5,osc_1_spectral_morph_type=1)
    r=convert(VitalPreset(doc),template)
    features={n['feature'] for n in r.report.notes if n['kind']=='unsupported'}
    assert {'osc_1_frame_spread','env_1_delay','osc_1_spectral_morph'}<=features


def test_effect_order_inversion():
    rng=random.Random(19)
    for _ in range(300):
        indices=list(range(9));rng.shuffle(indices)
        code=0
        for i in range(1,9): code=code*(i+1)+sum(indices[i]<indices[j] for j in range(i))
        assert decode_order(code)==[EFFECTS[i] for i in indices]
    assert decode_order(0)==EFFECTS
    assert decode_order(362879)==EFFECTS[::-1]

@pytest.mark.parametrize('sync,tempo,index,offset',[(1,8,8,1),(2,9,9,1.5),(3,8,9,4/3)])
def test_delay_sync_native(template,doc,sync,tempo,index,offset):
    doc['settings'].update(delay_on=1,delay_sync=sync,delay_tempo=tempo,delay_style=0)
    r=convert(VitalPreset(doc),template)
    fx=plain(r.preset.state['FXRack0']['FX'][0]['FXDelay'])
    assert fx['kParamBeatSync']==1
    assert fx['kParamTimeL']==DELAY_KNOBS[index]
    assert fx['kParamOffsetL']==offset
    assert fx['kParamTimeR']==fx['kParamTimeL']


def test_unsupported_fx_not_claimed(template,doc):
    doc['settings'].update(filter_fx_on=1,filter_fx_model=5)
    r=convert(VitalPreset(doc),template)
    assert r.preset.state['FXRack0']['FX']==[]
    assert 'filter_fx' not in r.report.mapped
    assert r.report.lossy


def test_compressor_eq_and_fx_filter(template,doc):
    doc['settings'].update(compressor_on=1,eq_on=1,eq_low_gain=-6.0,eq_high_gain=4.0,
                           filter_fx_on=1,filter_fx_model=0,filter_fx_style=0,filter_fx_cutoff=90.0,
                           filter_fx_resonance=0.5,filter_fx_mix=1.0)
    r=convert(VitalPreset(doc),template)
    fx={list(f)[0] if list(f)[0]!='kUIParamMixOrGain' else list(f)[1]:f for f in r.preset.state['FXRack0']['FX']}
    assert {'FXComp','FXEQ','FXFilter'} <= set(fx)
    comp=plain(fx['FXComp']['FXComp'])
    assert comp['kParamMultiband']==1.0 and 0<comp['kParamThresh']<1 and comp['kParamAttack']>0
    eq=plain(fx['FXEQ']['FXEQ'])
    assert eq['kParamGain1']==-6.0 and eq['kParamGain2']==4.0 and eq['kParamFreq1']<eq['kParamFreq2']
    filt=plain(fx['FXFilter']['FXFilter'])
    assert filt['kParamType'].startswith('LBH') and filt['kParamFreq']==pytest.approx((90+.39)/137.86)
    assert {'compressor','eq','filter_fx'} <= set(r.report.mapped)


def test_reverb_chorus_distortion_write(template,doc):
    doc['settings'].update(chorus_on=1,chorus_sync=0,reverb_on=1,distortion_on=1)
    r=convert(VitalPreset(doc),template)
    types=[fx['type'] for fx in r.preset.state['FXRack0']['FX']]
    assert types==[3,0,6]
    assert SerumPreset.from_bytes(r.preset.to_bytes()).state==r.preset.state


def test_inactive_lfo_rate_is_not_false_loss(template,doc):
    doc['settings'].update(lfo_1_sync=1,lfo_1_frequency=6)
    result=convert(VitalPreset(doc),template)
    assert not any(n['feature']=='lfo_1_frequency' for n in result.report.notes)


def test_sample_becomes_a_noise_oscillator(template,doc):
    import base64,struct
    doc['settings'].update(sample_on=1,sample_loop=0,sample_pan=.2,sample_keytrack=1,sample_transpose=-7,sample_tune=.25,
        sample={'length':2,'sample_rate':48000,'samples':base64.b64encode(struct.pack('<2h',12,-34)).decode()})
    r=convert(VitalPreset(doc),template)
    noise=r.preset.state['Oscillator3']['NoiseOsc3']
    # Serum's Noise oscillator follows the keyboard from C5 and reads the clip's rate from detuneFactor (measured)
    assert noise['detuneFactor']==pytest.approx(48000/2**16.5,rel=1e-5) and noise['numFrames']==2
    assert plain(noise)['kParamOneShot']==1 and plain(noise)['kParamColor']==.5+5/128 and plain(noise)['kParamFine']==.25
    assert plain(r.preset.state['Oscillator3'])['kParamPitchTrack']==1
    assert 0<plain(r.preset.state['Oscillator3'])['kParamPan']<10
    # without key tracking the pitch goes into the colour, clip rate included
    doc['settings'].update(sample_keytrack=0,sample_transpose=0,sample_tune=0,sample_bounce=1,sample_random_phase=1)
    r=convert(VitalPreset(doc),template)
    noise=r.preset.state['Oscillator3']['NoiseOsc3']
    assert plain(noise)['kParamColor']==pytest.approx(.5+12*math.log2(48000/44100)/96) and 'kParamFine' not in plain(noise)
    assert 'kParamPitchTrack' not in plain(r.preset.state['Oscillator3'])
    assert noise['numFrames']==4 and [round(x*32768) for x in noise['embeddedNoiseData'][0]]==[12,-34,-34,12]
    assert plain(noise)['kParamRandomPhase']==pytest.approx(70.71,abs=.01)                # the forward half of the mirrored clip


def test_unknown_template_state_survives(template,vital):
    template.state['UnknownUiExtension']={'opaque':b'\x00\xff','position':[10,20]}
    result=convert(vital,template)
    assert SerumPreset.from_bytes(result.preset.to_bytes()).state['UnknownUiExtension']==template.state['UnknownUiExtension']


@pytest.mark.parametrize('target,base,amount,offset', [
    ('phaser_mod_depth',0.0,.5,12.0),
    ('phaser_frequency',0.0,.5,1.75),
    ('phaser_phase_offset',0.0,.5,.25),
])
def test_fixed_macro_on_unrouted_phaser_control_is_folded(doc,target,base,amount,offset):
    from vital2serum.mapping import fold_static_modulations
    from vital2serum.report import Report
    doc['settings'].update(phaser_on=1,macro_control_1=.5,
                           modulations=[{'source':'macro_control_1','destination':target}],
                           modulation_1_amount=amount)
    doc['settings'][target]=base
    vital=VitalPreset(doc)
    fold_static_modulations(vital,Report('test'))
    assert vital.get(target)==pytest.approx(base+offset)
    assert vital.folded=={1}
