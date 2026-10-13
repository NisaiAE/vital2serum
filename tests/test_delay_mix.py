import math
import copy
from bisect import bisect_right

import pytest

from vital2serum.effects import delay_mix, level_out
from vital2serum import effects, modulation, distortion_model
from vital2serum.mapping import convert
from vital2serum.vital import VitalPreset
from vital2serum.serum import SerumPreset, plain


@pytest.mark.parametrize('style',range(4))
def test_delay_mix_dry_and_wet_coefficients_across_range(style):
    scale=math.sqrt(.5) if style>=2 else 1.0
    for i in range(4097):
        mix=i/4096
        wet,db=delay_mix(mix,style)
        share=math.sin(wet/100*math.pi/2)**2
        output=level_out(db)
        gain=(2*output)**2
        assert gain*(1-share**2)==pytest.approx(math.cos(mix*math.pi/2),abs=2e-15)
        assert scale*gain*share==pytest.approx(math.sin(mix*math.pi/2),abs=2e-15)
        assert 0<=wet<=100
        assert 0<output<1


@pytest.mark.parametrize('style',range(4))
@pytest.mark.parametrize('mix',[0,.25,.5,.75,1])
def test_delay_mix_rack_preserves_timing_and_feedback(template,doc,style,mix):
    doc['settings'].update(delay_on=1,delay_style=style,delay_sync=0,delay_aux_sync=0,
                           delay_frequency=2,delay_aux_frequency=3,delay_feedback=.6,delay_dry_wet=mix)
    result=convert(VitalPreset(doc),template)
    entries=result.preset.state['FXRack0']['FX']
    assert len(entries)==1
    params=entries[0]['FXDelay']['plainParams']
    wet,db=delay_mix(mix,style)
    assert params['kParamWet']==wet
    assert params['kParamLevelOut']==level_out(db)
    assert params['kParamTimeL']==.25
    assert params['kParamTimeR']==(.25 if style==0 else .125)
    assert params['kParamFeedback']==60
    assert params['kParamMode']==(1 if style>=2 else 0)


@pytest.mark.parametrize('style',range(4))
def test_delay_mix_endpoints_and_clamping(style):
    assert delay_mix(-1,style)==delay_mix(0,style)==(0.,0.)
    assert delay_mix(2,style)==delay_mix(1,style)
    wet,db=delay_mix(1,style)
    assert wet==100
    assert db==pytest.approx(20*math.log10(math.sqrt(2)) if style>=2 else 0)


def test_delay_mix_keeps_stored_controls_when_moving_mix_is_unsupported(template,doc):
    doc['settings'].update(delay_on=1,delay_style=2,delay_dry_wet=.25,
                           modulations=[{'source':'env_2','destination':'delay_dry_wet'}],
                           modulation_1_amount=.5,modulation_1_bipolar=1,modulation_1_stereo=1)
    result=convert(VitalPreset(doc),template)
    params=result.preset.state['FXRack0']['FX'][0]['FXDelay']['plainParams']
    wet,db=delay_mix(.25,2)
    assert params['kParamWet']==wet
    assert params['kParamLevelOut']==level_out(db)
    assert all(not v.get('source',[0,0])[0] for k,v in result.preset.state.items() if k.startswith('ModSlot'))
    assert result.report.modulations[0]['status']=='omitted'


def moving_mix(doc,**changes):
    doc['settings'].update(delay_on=1,delay_style=0,delay_dry_wet=0,
                           modulations=[{'source':'env_2','destination':'delay_dry_wet'}],
                           modulation_1_amount=1)
    doc['settings'].update(changes)
    return VitalPreset(doc)


def curve_value(curve,x):
    xs,ys=curve['xVals'],curve['yVals'];i=min(len(xs)-2,max(0,bisect_right(xs,x)-1))
    t=(x-xs[i])/(xs[i+1]-xs[i])
    return 1-(ys[i]+t*(ys[i+1]-ys[i]))


@pytest.mark.parametrize('style',range(3))
@pytest.mark.parametrize('source,power,aux',[('env_1',0,False),('env_2',0,False),
                                          ('lfo_1',3,False),('velocity',-3,False),('env_2',2,True)])
def test_serialized_delay_mix_curves_preserve_coefficients(template,doc,style,source,power,aux):
    vital=moving_mix(doc,delay_style=style,modulation_1_power=power,chorus_on=1)
    vital.settings['modulations'][0]['source']=source
    if aux:
        vital.settings.update(modulation_1_amount=.2,modulation_2_amount=.4,macro_control_1=.75)
        vital.settings['modulations'].append({'source':'macro_control_1','destination':'modulation_1_amount'})
    result=convert(vital,template);saved=SerumPreset.from_bytes(result.preset.to_bytes())
    route=next(z for z in result.report.modulations if z['destination']=='delay_dry_wet')
    assert route['status']=='mapped'
    index=next(i for i,e in enumerate(saved.state['FXRack0']['FX']) if 'FXDelay' in e)
    group=route['atomic_law_destinations']
    assert {(e['module_id'],e['parameter_id']) for e in group}=={(index,1),(index,12)}
    controls=saved.state['FXRack0']['FX'][index]['FXDelay']['plainParams']
    mod=modulation.active_modulations(vital)[1];source_law=modulation.SOURCE_LAWS.get(source,lambda x:x)
    for i in range(4097):
        x=i/4096;positions={'kParamWet':controls['kParamWet']/100,'kParamLevelOut':controls['kParamLevelOut']}
        for item in group:
            slot=saved.state[f"ModSlot{item['slot']}"]
            assert len(slot['flex'][0]['xVals'])<=480
            positions[item['parameter']]+=plain(slot)['kParamAmount']/100*curve_value(slot['flex'][0],x)
        gain=(2*positions['kParamLevelOut'])**2;share=math.sin(positions['kParamWet']*math.pi/2)**2
        mix=min(1,max(0,(.8 if aux else 1)*mod.value(source_law(x))))
        assert gain*(1-share**2)==pytest.approx(math.cos(mix*math.pi/2),abs=3e-4)
        assert gain*share*(math.sqrt(.5) if style>=2 else 1)==pytest.approx(math.sin(mix*math.pi/2),abs=3e-4)


@pytest.mark.parametrize('missing',[('FXDelay','kParamWet'),('FXDelay','kParamLevelOut')])
def test_delay_mix_missing_id_restores_all_static_controls(template,doc,monkeypatch,missing):
    vital=moving_mix(doc,delay_style=2,delay_dry_wet=.25);static=copy.deepcopy(vital);static.settings['modulations']=[]
    expected=convert(static,template)
    monkeypatch.delitem(modulation.KNOWN_IDS,missing)
    result=convert(vital,template)
    assert result.preset.state['FXRack0']==expected.preset.state['FXRack0']
    assert result.report.modulations[0]['status']=='omitted'
    assert not any(v.get('destModuleTypeString')=='FXDelay' for k,v in result.preset.state.items() if k.startswith('ModSlot'))


@pytest.mark.parametrize('users',[61,62])
def test_delay_mix_signed_slot_budget_is_atomic(template,doc,users):
    vital=moving_mix(doc,delay_dry_wet=.9)
    vital.settings['modulations']=[{'source':'lfo_1','destination':'osc_1_pan'} for _ in range(users)]+[
        {'source':'env_2','destination':'delay_dry_wet'}]
    for i in range(1,users+1):vital.settings[f'modulation_{i}_amount']=.01
    vital.settings[f'modulation_{users+1}_amount']=-.9
    result=convert(vital,template)
    route=next(m for m in result.report.modulations if m['destination']=='delay_dry_wet')
    assert route['status']==('mapped' if users==61 else 'omitted')
    if users==62:
        assert not any(v.get('destModuleTypeString')=='FXDelay' for k,v in result.preset.state.items() if k.startswith('ModSlot'))


@pytest.mark.parametrize('changes',[
    {'modulation_1_stereo':1},
    {'modulations':[{'source':'env_2','destination':'delay_dry_wet'},
                    {'source':'lfo_1','destination':'delay_dry_wet'}],'modulation_2_amount':.2},
    {'modulations':[{'source':'env_2','destination':'delay_dry_wet'},
                    {'source':'lfo_1','destination':'delay_feedback'}],'modulation_2_amount':.2}])
def test_delay_mix_guards_keep_static_response(template,doc,changes):
    vital=moving_mix(doc,delay_dry_wet=.25,**changes)
    static=copy.deepcopy(vital);static.settings['modulations']=[]
    expected=convert(static,template);result=convert(vital,template)
    assert result.preset.state['FXRack0']==expected.preset.state['FXRack0']
    assert all(m['status']=='omitted' for m in result.report.modulations)


def test_delay_mix_nonunit_source_gain_is_atomic(template,doc,monkeypatch):
    original=effects.delay_mix_control
    def changed(ctx,style,index):
        ctx.source_gain['env_2']=.5
        original(ctx,style,index)
    monkeypatch.setattr(effects,'delay_mix_control',changed)
    result=convert(moving_mix(doc),template)
    assert result.report.modulations[0]['status']=='omitted'
    assert not any(v.get('destModuleTypeString')=='FXDelay' for k,v in result.preset.state.items() if k.startswith('ModSlot'))


def test_delay_mix_curve_budget_failure_is_atomic(template,doc,monkeypatch):
    original=modulation._adaptive_law_points
    monkeypatch.setattr(modulation,'_adaptive_law_points',lambda function,m,policy:original(function,m,(1e-12,80)))
    vital=moving_mix(doc);static=copy.deepcopy(vital);static.settings['modulations']=[]
    expected=convert(static,template);result=convert(vital,template)
    assert result.preset.state['FXRack0']==expected.preset.state['FXRack0']
    assert result.report.modulations[0]['status']=='omitted'
    assert not any(v.get('destModuleTypeString')=='FXDelay' for k,v in result.preset.state.items() if k.startswith('ModSlot'))


@pytest.mark.parametrize('kind',[distortion_model.SOFT_CLIP,distortion_model.HARD_CLIP,
                                distortion_model.DOWN_SAMPLE])
@pytest.mark.parametrize('order',[0,15120])
def test_delay_fixed_aux_mix_survives_drive_registration_in_either_order(template,doc,kind,order):
    vital=moving_mix(doc,delay_dry_wet=.25,distortion_on=1,distortion_type=kind,
        distortion_drive=0,distortion_mix=1,distortion_filter_order=0,
        effect_chain_order=order,macro_control_1=.5,
        modulation_1_amount=.2,modulation_2_amount=.2,modulation_3_amount=.3,
        modulations=[{'source':'env_2','destination':'delay_dry_wet'},
                     {'source':'macro_control_1','destination':'modulation_1_amount'},
                     {'source':'env_3','destination':'distortion_drive'}])
    result=convert(vital,template)
    routes={route['destination']:route for route in result.report.modulations}
    delay=routes['delay_dry_wet'];drive=routes['distortion_drive']
    assert delay['status']==drive['status']=='mapped'
    group=delay['atomic_law_destinations']
    assert {item['parameter_id'] for item in group}=={1,12}
    index=next(i for i,entry in enumerate(result.preset.state['FXRack0']['FX']) if 'FXDelay' in entry)
    assert {item['module_id'] for item in group}=={index}
    assert any(note.get('destination')=='delay_dry_wet' and
               note['feature']=='modulation_1_static_aux' for note in result.report.notes)
    isolated=copy.deepcopy(vital);isolated.settings['distortion_on']=0
    expected=convert(isolated,template)
    expected_group=next(route for route in expected.report.modulations
                        if route['destination']=='delay_dry_wet')['atomic_law_destinations']
    assert len(group)==len(expected_group)
    for actual,wanted in zip(group,expected_group):
        assert actual['parameter']==wanted['parameter']
        assert actual['amount']==wanted['amount']
        assert result.preset.state[f"ModSlot{actual['slot']}"]['flex']==expected.preset.state[f"ModSlot{wanted['slot']}"]['flex']


@pytest.mark.parametrize('order',[0,15120])
def test_fixed_aux_mix_registrations_coexist_in_either_effect_order(template,doc,order):
    vital=moving_mix(doc,delay_dry_wet=.25,distortion_on=1,distortion_type=0,
        distortion_drive=0,distortion_mix=.25,distortion_filter_order=0,
        effect_chain_order=order,macro_control_1=.5,
        modulation_1_amount=.2,modulation_2_amount=.2,
        modulation_3_amount=.3,modulation_4_amount=.2,
        modulations=[{'source':'env_2','destination':'delay_dry_wet'},
                     {'source':'macro_control_1','destination':'modulation_1_amount'},
                     {'source':'env_3','destination':'distortion_mix'},
                     {'source':'macro_control_1','destination':'modulation_3_amount'}])
    result=convert(vital,template)
    for target,module in [('delay_dry_wet','FXDelay'),('distortion_mix','FXDistortion')]:
        route=next(route for route in result.report.modulations if route['destination']==target)
        assert route['status']=='mapped'
        group=route['atomic_law_destinations']
        assert {item['parameter'] for item in group}=={'kParamWet','kParamLevelOut'}
        index=next(i for i,entry in enumerate(result.preset.state['FXRack0']['FX']) if module in entry)
        assert {item['module_id'] for item in group}=={index}
