"""Relevant Vital 1.5.x defaults, derived from serum2vital/vital_defaults.py, which is generated from
Vital's src/common/synth_parameters.cpp (mtytel/vital, Copyright 2013-2019 Matt Tytel, GPL-3.0-or-later).
Missing scalar values mean Vital defaults, not zero.
SPDX-License-Identifier: GPL-3.0-or-later
"""
DEFAULTS = {
    'volume':5473.0404, 'polyphony':8, 'legato':0, 'pitch_bend_range':2,
    'portamento_time':-10, 'portamento_force':0, 'portamento_scale':0, 'portamento_slope':0,
    'voice_transpose':0, 'voice_tune':0, 'voice_amplitude':1, 'velocity_track':0,
    'stereo_mode':0, 'stereo_routing':1, 'oversampling':1, 'bypass':0,
    'beats_per_minute':2, 'mod_wheel':0, 'pitch_wheel':0, 'voice_override':0,
    'voice_priority':4, 'view_spectrogram':0, 'effect_chain_order':0,
}
MODULE_DEFAULTS = {
    'env': dict(attack=.1495,attack_power=0,decay=1,decay_power=-2,delay=0,hold=0,
                release=.5476,release_power=-2,sustain=1),
    'lfo': dict(delay_time=0,fade_time=0,frequency=1,keytrack_transpose=-12,keytrack_tune=0,
                phase=0,smooth_mode=1,smooth_time=-7.5,stereo=0,sync=1,sync_type=0,tempo=7),
    'osc': dict(detune_power=1.5,detune_range=2,distortion_amount=.5,distortion_phase=.5,
                distortion_spread=0,distortion_type=0,frame_spread=0,level=.70710678119,
                midi_track=1,on=0,pan=0,phase=.5,random_phase=1,smooth_interpolation=0,
                spectral_morph_amount=.5,spectral_morph_phase=.5,spectral_morph_spread=0,
                spectral_morph_type=0,spectral_unison=1,stack_style=0,stereo_spread=1,
                transpose=0,transpose_quantize=0,tune=0,unison_blend=.8,
                unison_detune=4.472135955,unison_voices=1,view_2d=1,wave_frame=0),
    'filter': dict(blend=0,blend_transpose=42,cutoff=60,drive=0,filter_input=0,
                   formant_resonance=.85,formant_spread=0,formant_transpose=0,formant_x=.5,
                   formant_y=.5,keytrack=0,mix=1,model=0,on=0,resonance=.5,style=0),
    'random': dict(frequency=1,keytrack_transpose=-12,keytrack_tune=0,stereo=0,style=0,
                   sync=1,sync_type=0,tempo=8),
}
for family,count in [('env',6),('lfo',8),('osc',3),('filter',2),('random',4)]:
    for i in range(1,count+1):
        DEFAULTS.update({f'{family}_{i}_{k}':v for k,v in MODULE_DEFAULTS[family].items()})
DEFAULTS.update({f'filter_fx_{k}':v for k,v in MODULE_DEFAULTS['filter'].items()})
for i,dest in [(1,0),(2,1),(3,3)]:
    DEFAULTS[f'osc_{i}_destination'] = dest
DEFAULTS['osc_1_on'] = 1
for i in range(1,5): DEFAULTS[f'macro_control_{i}'] = 0
for i in range(1,65):
    for p in ['amount','bipolar','bypass','power','stereo']:
        DEFAULTS[f'modulation_{i}_{p}'] = 0
EFFECT_DEFAULTS = {
    'chorus':dict(cutoff=60,delay_1=-9,delay_2=-7,dry_wet=.5,feedback=.4,frequency=-3,
                  mod_depth=.5,on=0,spread=1,sync=1,tempo=4,voices=4),
    'delay':dict(aux_frequency=2,aux_sync=1,aux_tempo=9,dry_wet=.3334,feedback=.5,
                 filter_cutoff=60,filter_spread=1,frequency=2,on=0,style=0,sync=1,tempo=9),
    'distortion':dict(drive=0,filter_blend=0,filter_cutoff=80,filter_order=0,
                      filter_resonance=.5,mix=1,on=0,type=0),
    'reverb':dict(chorus_amount=.223607,chorus_frequency=-2,decay_time=0,delay=0,dry_wet=.25,
                  high_shelf_cutoff=90,high_shelf_gain=-1,low_shelf_cutoff=0,low_shelf_gain=0,
                  on=0,pre_high_cutoff=110,pre_low_cutoff=0,size=.5),
    'flanger':dict(center=64,dry_wet=.5,feedback=.5,frequency=2,mod_depth=.5,on=0,
                   phase_offset=1/3,sync=1,tempo=4),
    'phaser':dict(blend=1,center=80,dry_wet=1,feedback=.5,frequency=-3,mod_depth=24,on=0,
                  phase_offset=1/3,sync=1,tempo=3),
    # EQ and compressor defaults from Vital's synth_parameters.cpp.
    'eq':dict(on=0,low_mode=0,low_cutoff=40,low_gain=0,low_resonance=.3163,
              band_mode=0,band_cutoff=80,band_gain=0,band_resonance=.4473,
              high_mode=0,high_cutoff=100,high_gain=0,high_resonance=.3163),
    'compressor':dict(on=0,enabled_bands=0,attack=.5,release=.5,mix=1,
                      low_upper_threshold=-28,band_upper_threshold=-25,high_upper_threshold=-30,
                      low_lower_threshold=-35,band_lower_threshold=-36,high_lower_threshold=-35,
                      low_upper_ratio=.9,band_upper_ratio=.857,high_upper_ratio=1,
                      low_lower_ratio=.8,band_lower_ratio=.8,high_lower_ratio=.8,
                      low_gain=16.3,band_gain=11.7,high_gain=16.3,low_band_unused=1),
    'sample':dict(bounce=0,destination=3,keytrack=0,level=.70710678119,loop=1,on=0,pan=0,
                  random_phase=0,transpose=0,transpose_quantize=0,tune=0),
}
for prefix,params in EFFECT_DEFAULTS.items():
    DEFAULTS.update({f'{prefix}_{k}':v for k,v in params.items()})
