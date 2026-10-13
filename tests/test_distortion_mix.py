import copy
import math
from bisect import bisect_right

import pytest

from vital2serum import distortion_model as d, modulation
from vital2serum.effects import hard_clip, soft_clip
from vital2serum.mapping import convert, Context
from vital2serum.report import Report
from vital2serum.serum import plain, SerumPreset
from vital2serum.vital import VitalPreset


def settings(kind=0, **changes):
    values=dict(type=kind,drive=6,mix=.3,filter_order=0,filter_cutoff=72,
                filter_blend=.4,filter_resonance=.4)
    return dict(values,**changes)


@pytest.mark.parametrize('kind',range(6))
@pytest.mark.parametrize('order',[0,1,2])
def test_mix_controls_preserve_dry_and_wet_coefficients(kind,order):
    values=settings(kind,filter_order=order)
    clip=soft_clip(6) if kind==0 else hard_clip(6) if kind==1 else None
    wet_gain,pre_gain=d.mix_control_scales(values,clip,span=2.3)
    for mix in (0,1e-8,.01,.25,.5,.75,.99,1):
        wet,amplitude=d.mix_params(mix,wet_gain,pre_gain)
        dry,weight=d.serum_mix(wet)
        assert 0<=.5*math.sqrt(amplitude)<=1
        assert amplitude*dry*pre_gain==pytest.approx(1-mix,abs=1e-12)
        assert amplitude*weight/wet_gain==pytest.approx(mix,abs=1e-12)


def moving_mix(doc,kind=0,**changes):
    doc['settings'].update(distortion_on=1,distortion_type=kind,distortion_drive=6,
                           distortion_mix=.3,distortion_filter_order=0,
                           modulation_1_amount=.7,
                           modulations=[{'source':'env_2','destination':'distortion_mix'}])
    doc['settings'].update(changes)
    return VitalPreset(doc)


@pytest.mark.parametrize('kind',range(6))
def test_mix_writes_both_controls_at_actual_rack_position(template,doc,kind):
    result=convert(moving_mix(doc,kind,chorus_on=1),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_mix')
    assert route['status']=='mapped'
    group=route['atomic_law_destinations']
    assert 2<=len(group)<=3
    assert {(item['module'],item['parameter_id']) for item in group}=={('FXDistortion',1),('FXDistortion',8)}
    entries=result.preset.state['FXRack0']['FX']
    index=next(i for i,entry in enumerate(entries) if 'FXDistortion' in entry)
    assert {item['module_id'] for item in group}=={index}


@pytest.mark.parametrize('missing',[('FXDistortion','kParamWet'),('FXDistortion','kParamLevelOut')])
def test_mix_missing_identifier_leaves_exact_static_state(template,doc,monkeypatch,missing):
    vital=moving_mix(doc)
    static=copy.deepcopy(vital);static.settings['modulations']=[]
    expected=convert(static,template)
    monkeypatch.delitem(modulation.KNOWN_IDS,missing)
    result=convert(vital,template)
    assert result.preset.state['FXRack0']==expected.preset.state['FXRack0']
    route=next(m for m in result.report.modulations if m['destination']=='distortion_mix')
    assert route['status']=='omitted' and not route.get('atomic_law_destinations')
    assert not any(value.get('destModuleTypeString')=='FXDistortion'
                   for name,value in result.preset.state.items() if name.startswith('ModSlot'))


@pytest.mark.parametrize('users',[62,63])
def test_mix_slot_budget_is_atomic(template,doc,users):
    vital=moving_mix(doc)
    vital.settings['modulations']=[{'source':'lfo_1','destination':'osc_1_pan'} for _ in range(users)]+[
        {'source':'env_2','destination':'distortion_mix'}]
    for i in range(1,users+1):vital.settings[f'modulation_{i}_amount']=.01
    vital.settings[f'modulation_{users+1}_amount']=.7
    result=convert(vital,template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_mix')
    assert route['status']==('mapped' if users==62 else 'omitted')
    assert all(m['status']=='mapped' for m in result.report.modulations if m['destination']=='osc_1_pan')
    if users==63:assert not route.get('atomic_law_destinations')


@pytest.mark.parametrize('changes',[
    {'modulation_1_stereo':1},
    {'modulations':[{'source':'env_2','destination':'distortion_mix'},
                    {'source':'lfo_1','destination':'distortion_drive'}],'modulation_2_amount':.1},
    {'modulations':[{'source':'env_2','destination':'distortion_mix'},
                    {'source':'lfo_1','destination':'distortion_filter_resonance'}],'modulation_2_amount':.1},
])
def test_mix_excludes_unverified_simultaneous_controls(template,doc,changes):
    result=convert(moving_mix(doc,**changes),template)
    route=next(m for m in result.report.modulations if m['destination']=='distortion_mix')
    assert route['status']=='omitted'


def curve_position(curve,x):
    xs,ys=curve['xVals'],curve['yVals'];i=min(len(xs)-2,max(0,bisect_right(xs,x)-1))
    t=(x-xs[i])/(xs[i+1]-xs[i])
    return 1-(ys[i]+t*(ys[i+1]-ys[i]))


@pytest.mark.parametrize('source,power,aux',[
    ('env_2',0,False),('env_1',0,False),('lfo_1',3,False),
    ('velocity',-3,False),('env_2',2,True),
])
def test_serialized_mix_curves_preserve_weights_across_source_shapes(template,doc,source,power,aux):
    vital=moving_mix(doc,distortion_mix=0,modulation_1_amount=1,modulation_1_power=power)
    vital.settings['modulations'][0]['source']=source
    if aux:
        vital.settings.update(modulation_1_amount=.2,modulation_2_amount=.4,macro_control_1=.75)
        vital.settings['modulations'].append({'source':'macro_control_1','destination':'modulation_1_amount'})
    result=convert(vital,template);saved=SerumPreset.from_bytes(result.preset.to_bytes())
    route=next(z for z in result.report.modulations if z['destination']=='distortion_mix')
    assert route['status']=='mapped'
    controls=saved.state['FXRack0']['FX'][route['atomic_law_destinations'][0]['module_id']]['FXDistortion']['plainParams']
    wet_gain,pre_gain=d.mix_control_scales(settings(mix=0),soft_clip(6))
    mod=modulation.active_modulations(vital)[1];law=modulation.SOURCE_LAWS.get(source,lambda x:x)
    coefficient=.8 if aux else 1
    for i in range(4097):
        x=i/4096;positions={'kParamWet':controls['kParamWet']/100,'kParamLevelOut':controls['kParamLevelOut']}
        for entry in route['atomic_law_destinations']:
            slot=saved.state['ModSlot'+str(entry['slot'])]
            positions[entry['parameter']]+=plain(slot)['kParamAmount']/100*curve_position(slot['flex'][0],x)
        amplitude=(2*positions['kParamLevelOut'])**2;wet=math.sin(positions['kParamWet']*math.pi/2)**2
        mix=min(1,max(0,coefficient*mod.value(law(x))))
        assert amplitude*(1-wet*wet)*pre_gain==pytest.approx(1-mix,abs=2e-4)
        assert amplitude*wet/wet_gain==pytest.approx(mix,abs=2e-4)


def test_adaptive_points_find_signed_piece_between_default_samples():
    v=VitalPreset({'settings':{'distortion_mix':0}})
    m=modulation.Modulation(1,'env_2','distortion_mix',1,False,0,False)
    law=lambda x:.5*math.sqrt(d.mix_params(x,3,3)[1])
    default=modulation.Destination('FXDistortion',0,'kParamLevelOut',law=law)
    adaptive=modulation.Destination('FXDistortion',0,'kParamLevelOut',law=law,curve_policy=(2e-5,481))
    assert all(amount>0 for amount,_ in modulation._law_slots(v,m,default,1))
    assert any(amount<0 for amount,_ in modulation._law_slots(v,m,adaptive,1))


def test_adaptive_points_preserve_narrow_remap_knots():
    v=VitalPreset({'settings':{'distortion_mix':0}})
    m=modulation.Modulation(1,'env_1','distortion_mix',1,False,0,False,
                           ((0,1),(.0151,1),(.0152,0),(.0153,1),(1,1)),(0,0,0,0))
    dest=modulation.Destination('FXDistortion',0,'kParamWet',law=lambda x:x,curve_policy=(2e-5,481))
    pieces=modulation._law_slots(v,m,dest,1)
    assert len(pieces)==1 and pieces[0][0]==pytest.approx(100)
    assert curve_position(pieces[0][1],.0152**2)==pytest.approx(1)


@pytest.mark.parametrize('budget',[33,80])
def test_point_budget_failure_retains_all_static_mix_controls(template,doc,monkeypatch,budget):
    vital=moving_mix(doc);static=copy.deepcopy(vital);static.settings['modulations']=[]
    expected=convert(static,template)
    original=modulation._adaptive_law_points
    monkeypatch.setattr(modulation,'_adaptive_law_points',lambda function,m,policy:original(function,m,(1e-12,budget)))
    result=convert(vital,template)
    assert result.preset.state['FXRack0']==expected.preset.state['FXRack0']
    route=next(z for z in result.report.modulations if z['destination']=='distortion_mix')
    assert route['status']=='omitted' and not route.get('atomic_law_destinations')
    assert not any(value.get('destModuleTypeString')=='FXDistortion'
                   for name,value in result.preset.state.items() if name.startswith('ModSlot'))


@pytest.mark.parametrize('users,expected',[(61,'written'),(62,'omitted')])
def test_adaptive_hidden_extremum_reserves_its_extra_signed_slot(template,doc,users,expected):
    doc['settings'].update(distortion_mix=0,modulation_1_amount=1,
                           modulations=[{'source':'env_2','destination':'distortion_mix'}])
    for i in range(users):
        doc['settings']['modulations'].append({'source':'lfo_1','destination':'osc_1_pan'})
        doc['settings'][f'modulation_{i+2}_amount']=.01
    ctx=Context(VitalPreset(doc),copy.deepcopy(template),Report('test','test'))
    ctx.target('osc_1_pan','Oscillator',0,'kParamPan')
    wet=lambda x:d.mix_params(x,3,3)[0]/100
    level=lambda x:.5*math.sqrt(d.mix_params(x,3,3)[1])
    destinations=[modulation.Destination('FXDistortion',0,'kParamWet',law=wet,curve_policy=(2e-5,480)),
                  modulation.Destination('FXDistortion',0,'kParamLevelOut',law=level,curve_policy=(2e-5,480))]
    events=[];ctx.atomic_law_destinations={1:(destinations,lambda:events.append('written'),lambda:events.append('omitted'))}
    modulation.convert_modulations(ctx,modulation.KNOWN_IDS)
    assert events==[expected]
    slots=[v for k,v in ctx.s.state.items() if k.startswith('ModSlot') and v.get('destModuleTypeString')=='FXDistortion']
    assert len(slots)==(3 if expected=='written' else 0)
