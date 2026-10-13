import copy
import math
from bisect import bisect_right

import pytest

from vital2serum import distortion_model, effects, modulation
from vital2serum.effects import soft_clip
from vital2serum.mapping import convert
from vital2serum.serum import SerumPreset, plain
from vital2serum.vital import VitalPreset


TARGETS=('distortion_mix','delay_dry_wet')
MODULES={'distortion_mix':'FXDistortion','delay_dry_wet':'FXDelay'}


def setup(doc,target,source='env_2',amount=1,base=.5):
    s=doc['settings']
    s.update(distortion_on=int(target=='distortion_mix'),delay_on=int(target=='delay_dry_wet'),
             distortion_type=0,distortion_drive=6,distortion_filter_order=0,
             distortion_filter_cutoff=72,distortion_filter_blend=.4,distortion_filter_resonance=.4,
             delay_style=0,macro_control_1=.25,macro_control_2=.75,macro_control_3=.25,
             modulation_1_amount=amount,modulation_1_bipolar=1,modulation_1_power=0,
             modulations=[{'source':source,'destination':target}])
    s[target]=base
    return s


def curve_value(curve,x):
    xs,ys=curve['xVals'],curve['yVals']
    i=min(len(xs)-2,max(0,bisect_right(xs,x)-1))
    t=(x-xs[i])/(xs[i+1]-xs[i])
    return 1-(ys[i]+t*(ys[i+1]-ys[i]))


def saved_control(result,target):
    saved=SerumPreset.from_bytes(result.preset.to_bytes())
    route=next(m for m in result.report.modulations if m['destination']==target)
    assert route['status']=='mapped'
    module=MODULES[target];group=route['atomic_law_destinations']
    index=next(i for i,e in enumerate(saved.state['FXRack0']['FX']) if module in e)
    assert {item['module_id'] for item in group}=={index}
    assert {item['parameter'] for item in group}=={'kParamWet','kParamLevelOut'}
    return saved,group,saved.state['FXRack0']['FX'][index][module]['plainParams']


def positions(saved,group,params,x):
    values={'kParamWet':params['kParamWet']/100,'kParamLevelOut':params['kParamLevelOut']}
    for item in group:
        slot=saved.state[f"ModSlot{item['slot']}"]
        assert plain(slot)['kParamBipolar']==0 and plain(slot)['kParamMainCurveData']==1
        curve=slot['flex'][0]
        assert curve['numPoints']==len(curve['xVals'])-1 and len(curve['xVals'])<=480
        values[item['parameter']]+=plain(slot)['kParamAmount']/100*curve_value(curve,x)
    return values


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('source,amount,power,base,extra',[
    ('env_2',1,0,.5,None),('env_2',-1,0,.5,None),('env_1',1,0,.5,None),
    ('lfo_1',.8,3,.5,None),('velocity',.8,-3,.5,None),
    ('env_2',1,0,.1,None),('env_2',1,0,.9,None),
    ('env_2',.9,0,.4,'remap'),('env_2',.2,0,.5,'aux'),
    ('env_2',.2,0,.5,'nested_aux'),
])
def test_bipolar_serialized_coefficients_follow_center_shape_and_target_clamp(
        template,doc,target,source,amount,power,base,extra):
    s=setup(doc,target,source,amount,base);s['modulation_1_power']=power
    coefficient=amount
    if extra=='remap':
        s['modulations'][0]['line_mapping']={'num_points':3,'points':[0,1,.35,.1,1,1],
                                           'powers':[0,0,0],'smooth':False}
    if extra in ('aux','nested_aux'):
        s['modulations'].append({'source':'macro_control_2','destination':'modulation_1_amount'})
        s.update(modulation_2_amount=.4,modulation_2_bipolar=1)
        coefficient+=.4*(.75-.5)*2
        if extra=='nested_aux':
            s['modulations'].append({'source':'macro_control_3','destination':'modulation_2_amount'})
            s['modulation_3_amount']=.2
            coefficient=amount+(.4+.2*.25*2)*(.75-.5)*2
    vital=VitalPreset(doc);result=convert(vital,template);saved,group,params=saved_control(result,target)
    mod=modulation.active_modulations(vital)[1]
    source_law=modulation.SOURCE_LAWS.get(source,lambda x:x)
    wet_gain,pre_gain=distortion_model.mix_control_scales(
        dict(type=0,drive=6,mix=base,filter_order=0,filter_cutoff=72,
             filter_blend=.4,filter_resonance=.4),soft_clip(6))
    for i in range(4097):
        x=i/4096;value=positions(saved,group,params,x)
        mix=max(0,min(1,base+coefficient*(mod.shaped(source_law(x))-.5)))
        gain=(2*value['kParamLevelOut'])**2;share=math.sin(value['kParamWet']*math.pi/2)**2
        if target=='distortion_mix':
            dry,wet=gain*(1-share**2)*pre_gain,gain*share/wet_gain
            desired_dry,desired_wet=1-mix,mix
        else:
            dry,wet=gain*(1-share**2),gain*share
            desired_dry,desired_wet=math.cos(mix*math.pi/2),math.sin(mix*math.pi/2)
        assert dry==pytest.approx(desired_dry,abs=3e-4)
        assert wet==pytest.approx(desired_wet,abs=3e-4)


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('source',['macro_control_1','mod_wheel'])
@pytest.mark.parametrize('amount',[-1,1])
def test_bipolar_stored_control_preserves_source_and_signed_endpoint_offsets(template,doc,target,source,amount):
    setup(doc,target,source,amount);vital=VitalPreset(doc);result=convert(vital,template)
    saved,group,params=saved_control(result,target)
    assert vital.folded==set() and len(group)==4
    assert any(item['amount']<0 for item in group) and any(item['amount']>0 for item in group)
    for item in group:assert saved.state[f"ModSlot{item['slot']}"]['source']==[modulation.SOURCES[source],0]
    if source=='macro_control_1':assert plain(saved.state['Macro0'])['kParamValue']==25
    for x in (0,.5,1):
        value=positions(saved,group,params,x)
        reference=copy.deepcopy(doc);reference['settings'].update(modulations=[])
        reference['settings'][target]=.5+amount*(x-.5)
        expected=convert(VitalPreset(reference),template)
        module=MODULES[target];wanted=next(e[module]['plainParams'] for e in expected.preset.state['FXRack0']['FX'] if module in e)
        assert value['kParamWet']==pytest.approx(wanted['kParamWet']/100,abs=3e-5)
        assert value['kParamLevelOut']==pytest.approx(wanted['kParamLevelOut'],abs=3e-5)


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('source',['env_2','macro_control_1'])
@pytest.mark.parametrize('guard',['stereo','multiple','transfer','ramp','nested_ramp','signed','moving_aux','shared_transfer'])
def test_bipolar_mix_retains_domain_and_dependency_guards(template,doc,monkeypatch,target,source,guard):
    s=setup(doc,target,source)
    transfer='distortion_drive' if target=='distortion_mix' else 'delay_feedback'
    if guard=='stereo':s['modulation_1_stereo']=1
    elif guard=='ramp':s['modulation_1_ramp_up']=1
    elif guard=='signed':monkeypatch.setitem(modulation.RANGES,source,(-1.,1.))
    elif guard=='nested_ramp':
        s['modulations'] += [{'source':'macro_control_2','destination':'modulation_1_amount'},
                             {'source':'macro_control_3','destination':'modulation_2_amount'}]
        s.update(modulation_2_amount=.2,modulation_3_amount=.2,modulation_3_ramp_down=1)
    else:
        destination=target if guard=='multiple' else 'modulation_1_amount' if guard=='moving_aux' else transfer
        s['modulations'].append({'source':source if guard=='shared_transfer' else 'lfo_2','destination':destination})
        s['modulation_2_amount']=.2
    vital=VitalPreset(copy.deepcopy(doc));result=convert(vital,template)
    reference=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_atomic_mix_targets=frozenset({target}))
    assert result.preset.to_bytes()==reference.preset.to_bytes()
    assert [(a.data,a.gain) for a in result.assets]==[(a.data,a.gain) for a in reference.assets]
    primary=[m for m in result.report.modulations if m['vital_slot']==1]
    assert all(not m.get('atomic_law_destinations') for m in primary)
    if not primary:assert 1 in vital.folded


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('failure',['wet_id','level_id','slot','point','source_gain'])
def test_failed_bipolar_group_restores_fold_and_preserves_other_mix_and_trajectory(
        template,doc,monkeypatch,target,failure):
    s=setup(doc,target,'macro_control_1');other=next(t for t in TARGETS if t!=target)
    s.update(distortion_on=1,delay_on=1,osc_1_distortion_type=4,osc_1_distortion_amount=.2)
    s[other]=0
    for source,destination,amount in [('env_2',other,.2),('env_3','osc_1_wave_frame',.5),
                                     ('env_3','osc_1_distortion_amount',.4)]:
        s['modulations'].append({'source':source,'destination':destination})
        s[f'modulation_{len(s["modulations"])}_amount']=amount
    ids=dict(modulation.KNOWN_IDS)
    if failure.endswith('_id'):
        del ids[(MODULES[target],'kParamWet' if failure=='wet_id' else 'kParamLevelOut')]
    elif failure=='slot':monkeypatch.setattr(modulation,'MAX_SLOTS',3)
    elif failure=='point':
        original=modulation._adaptive_law_points
        monkeypatch.setattr(modulation,'_adaptive_law_points',
            lambda f,m,p:original(f,m,(1e-12,80) if m.target==target else p))
    else:
        owner,name=(effects,'delay_mix_control') if target=='delay_dry_wet' else (distortion_model,'register_mix_control')
        original=getattr(owner,name)
        def changed(ctx,*args):
            ctx.source_gain['macro_control_1']=.5
            return original(ctx,*args)
        monkeypatch.setattr(owner,name,changed)
    vital=VitalPreset(copy.deepcopy(doc));result=convert(vital,template,destination_ids=ids)
    repeated=convert(vital,template,destination_ids=ids)
    reference=convert(VitalPreset(copy.deepcopy(doc)),template,destination_ids=ids,
                      _disabled_atomic_mix_targets=frozenset({target}))
    assert result.preset.to_bytes()==repeated.preset.to_bytes()==reference.preset.to_bytes()
    assert [(a.data,a.gain) for a in result.assets]==[(a.data,a.gain) for a in reference.assets]
    assert next(m for m in result.report.modulations if m['destination']==other)['status']=='mapped'
    assert {m['joint_trajectory'] for m in result.report.modulations if m.get('joint_trajectory')}=={'osc_1_'}
    assert not any(m.get('destModuleTypeString')==MODULES[target] for k,m in result.preset.state.items() if k.startswith('ModSlot'))
