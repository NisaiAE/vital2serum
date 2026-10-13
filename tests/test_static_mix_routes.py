import copy
import math
from bisect import bisect_right

import pytest

from vital2serum import effects, distortion_model, modulation
from vital2serum.mapping import convert
from vital2serum.serum import plain, SerumPreset
from vital2serum.vital import VitalPreset


TARGETS=('distortion_mix','delay_dry_wet')
MODULES={'distortion_mix':'FXDistortion','delay_dry_wet':'FXDelay'}


def setup(doc,target,source='macro_control_1'):
    s=doc['settings'];s.update(distortion_on=float(target=='distortion_mix'),delay_on=float(target=='delay_dry_wet'),
                              distortion_type=0,distortion_drive=6,distortion_filter_order=0,
                              distortion_mix=.25,delay_dry_wet=.25,delay_style=0,
                              macro_control_1=.6,mod_wheel=.6,modulation_1_amount=.4,
                              modulations=[{'source':source,'destination':target}])
    return s


def curve_value(curve,x):
    xs,ys=curve['xVals'],curve['yVals'];i=min(len(xs)-2,max(0,bisect_right(xs,x)-1))
    t=(x-xs[i])/(xs[i+1]-xs[i])
    return 1-(ys[i]+t*(ys[i+1]-ys[i]))


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('source',['macro_control_1','mod_wheel'])
def test_primary_stored_mix_remains_operable_and_matches_stored_state(template,doc,target,source):
    setup(doc,target,source);v=VitalPreset(doc);result=convert(v,template)
    assert v.folded==set()
    route=result.report.modulations[0];assert route['status']=='mapped'
    saved=SerumPreset.from_bytes(result.preset.to_bytes());module=MODULES[target]
    index=next(i for i,e in enumerate(saved.state['FXRack0']['FX']) if module in e)
    params=saved.state['FXRack0']['FX'][index][module]['plainParams']
    for x in (0,.001,.2,.6,.999,1):
        position={'kParamWet':params['kParamWet']/100,'kParamLevelOut':params['kParamLevelOut']}
        for entry in route['atomic_law_destinations']:
            slot=saved.state[f"ModSlot{entry['slot']}"]
            assert slot['source']==[modulation.SOURCES[source],0]
            position[entry['parameter']]+=plain(slot)['kParamAmount']/100*curve_value(slot['flex'][0],x)
        reference=copy.deepcopy(doc);reference['settings'][target]=.25+.4*x;reference['settings']['modulations']=[]
        r=convert(VitalPreset(reference),template);p=r.preset.state['FXRack0']['FX'][index][module]['plainParams']
        assert position['kParamWet']==pytest.approx(p['kParamWet']/100,abs=3e-5)
        assert position['kParamLevelOut']==pytest.approx(p['kParamLevelOut'],abs=3e-5)


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('bipolar',[0,1])
@pytest.mark.parametrize('failure',['id','slot','point','registration','source_gain'])
def test_failed_primary_static_mix_returns_complete_stored_fold(template,doc,monkeypatch,target,bipolar,failure):
    setup(doc,target)['modulation_1_bipolar']=bipolar;options={}
    if failure=='id':
        ids=dict(modulation.KNOWN_IDS);del ids[(MODULES[target],'kParamLevelOut')];options['destination_ids']=ids
    elif failure=='slot':monkeypatch.setattr(modulation,'MAX_SLOTS',1)
    elif failure=='point':monkeypatch.setattr(modulation,'_adaptive_law_points',lambda *a:(_ for _ in ()).throw(ValueError('budget')))
    else:
        owner,function=(distortion_model,'register_mix_control') if target=='distortion_mix' else (effects,'delay_mix_control')
        original=getattr(owner,function)
        if failure=='registration':monkeypatch.setattr(owner,function,lambda *a:None)
        else:
            def changed(ctx,*args):
                ctx.source_gain['macro_control_1']=.5
                original(ctx,*args)
            monkeypatch.setattr(owner,function,changed)
    v=VitalPreset(copy.deepcopy(doc));result=convert(v,template,**options)
    ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_atomic_mix_targets=frozenset({target}),**options)
    assert result.preset.state==ref.preset.state
    assert [(a.data,a.gain) for a in result.assets]==[(a.data,a.gain) for a in ref.assets]
    assert result.report.modulations==ref.report.modulations
    assert v.folded=={1}
    assert not any(z.get('destModuleTypeString')==MODULES[target] for k,z in result.preset.state.items() if k.startswith('ModSlot'))


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('axis',['transfer','aux'])
def test_same_primary_macro_additional_axis_keeps_original_fold(template,doc,target,axis):
    s=setup(doc,target)
    destination=('distortion_drive' if target=='distortion_mix' else 'delay_feedback') if axis=='transfer' else 'modulation_1_amount'
    s['modulations'].append({'source':'macro_control_1','destination':destination});s['modulation_2_amount']=.2
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    reference=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_atomic_mix_targets=frozenset({target}))
    assert result.preset.state==reference.preset.state
    assert result.report.modulations==reference.report.modulations
    assert not any(z.get('destModuleTypeString')==MODULES[target] for k,z in result.preset.state.items() if k.startswith('ModSlot'))


@pytest.mark.parametrize('target',TARGETS)
def test_separate_stored_aux_is_folded_into_primary_mix_curve(template,doc,target):
    s=setup(doc,target);s.update(macro_control_2=.25,modulation_2_amount=.2)
    s['modulations'].append({'source':'macro_control_2','destination':'modulation_1_amount'})
    result=convert(VitalPreset(doc),template)
    assert result.report.modulations[0]['status']=='mapped'
    assert any(n['feature']=='modulation_1_static_aux' for n in result.report.notes)


@pytest.mark.parametrize('target',TARGETS)
def test_primary_macro_inside_nested_aux_keeps_original_fold(template,doc,target):
    s=setup(doc,target);s.update(macro_control_2=.25,modulation_2_amount=.2,modulation_3_amount=.1)
    s['modulations'] += [{'source':'macro_control_2','destination':'modulation_1_amount'},
                         {'source':'macro_control_1','destination':'modulation_2_amount'}]
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_atomic_mix_targets=frozenset({target}))
    assert result.preset.state==ref.preset.state
    assert result.report.modulations==ref.report.modulations


def test_failed_mix_preserves_other_mix_and_macro_trajectory(template,doc,monkeypatch):
    s=setup(doc,'distortion_mix');s.update(delay_on=1,macro_control_2=.3,macro_control_3=.4,
        osc_1_distortion_type=4,osc_1_distortion_amount=.2,osc_1_wave_frame=10,
        modulation_2_amount=.4,modulation_3_amount=.75,modulation_4_amount=.6)
    s['modulations'] += [{'source':'macro_control_2','destination':'delay_dry_wet'},
        {'source':'macro_control_3','destination':'osc_1_wave_frame'},
        {'source':'macro_control_3','destination':'osc_1_distortion_amount'}]
    monkeypatch.setattr(distortion_model,'register_mix_control',lambda *a:None)
    v=VitalPreset(copy.deepcopy(doc));first=convert(v,template);second=convert(v,template)
    ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_atomic_mix_targets=frozenset({'distortion_mix'}))
    assert first.preset.state==second.preset.state==ref.preset.state
    assert first.report.modulations==second.report.modulations==ref.report.modulations
    assert first.report.notes==second.report.notes
    assert [m['joint_trajectory'] for m in first.report.modulations if m.get('joint_trajectory')]==['osc_1_','osc_1_']
    assert next(m for m in first.report.modulations if m['destination']=='delay_dry_wet')['status']=='mapped'
    assert v.folded=={1}


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('source',['env_2','macro_control_1'])
@pytest.mark.parametrize('ramped_slot',[1,2,3])
@pytest.mark.parametrize('direction',['up','down'])
def test_mix_ramp_in_primary_or_amount_chain_keeps_stored_conversion(template,doc,target,source,ramped_slot,direction):
    s=setup(doc,target,source);s.update(macro_control_2=.25,modulation_2_amount=.2,
                                    macro_control_3=.5,modulation_3_amount=.1)
    s['modulations'] += [{'source':'macro_control_2','destination':'modulation_1_amount'},
                         {'source':'macro_control_3','destination':'modulation_2_amount'}]
    s[f'modulation_{ramped_slot}_ramp_{direction}']=1
    v=VitalPreset(copy.deepcopy(doc));result=convert(v,template)
    reference=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_atomic_mix_targets=frozenset({target}))
    assert result.preset.state==reference.preset.state
    assert [(a.data,a.gain) for a in result.assets]==[(a.data,a.gain) for a in reference.assets]
    assert not any(z.get('destModuleTypeString')==MODULES[target] for k,z in result.preset.state.items() if k.startswith('ModSlot'))
    assert any(n['kind']=='unsupported' and n['feature']==f'modulation_{ramped_slot}_ramp'
               and n.get('destination')==target for n in result.report.notes)
    if source=='macro_control_1':
        assert v.folded=={1,2,3}
    else:
        assert next(m for m in result.report.modulations if m['destination']==target)['status']=='omitted'


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('signed_slot',[1,2,3])
def test_explicit_signed_source_range_excludes_complete_mix(template,doc,monkeypatch,target,signed_slot):
    s=setup(doc,target);s.update(macro_control_2=.25,modulation_2_amount=.2,
                              macro_control_3=.5,modulation_3_amount=.1)
    s['modulations'] += [{'source':'macro_control_2','destination':'modulation_1_amount'},
                         {'source':'macro_control_3','destination':'modulation_2_amount'}]
    monkeypatch.setitem(modulation.RANGES,f'macro_control_{signed_slot}',(-1.,1.))
    v=VitalPreset(copy.deepcopy(doc));result=convert(v,template)
    ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_atomic_mix_targets=frozenset({target}))
    assert result.preset.state==ref.preset.state
    assert v.folded=={1,2,3}
    assert any(n['feature']==f'modulation_{signed_slot}_source_range' and n['kind']=='unsupported'
               and n.get('destination')==target for n in result.report.notes)


@pytest.mark.parametrize('target',TARGETS)
def test_signed_pitch_wheel_keeps_static_mix_and_reports_domain(template,doc,target):
    setup(doc,target,'pitch_wheel')
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    static=copy.deepcopy(doc);static['settings']['modulations']=[]
    reference=convert(VitalPreset(static),template)
    assert result.preset.state==reference.preset.state
    assert result.report.modulations[0]['status']=='omitted'
    assert any(n['feature']=='modulation_1_source_range' and n.get('source')=='pitch_wheel'
               for n in result.report.notes)


@pytest.mark.parametrize('target',TARGETS)
def test_ramp_off_boundary_keeps_supported_primary_mix(template,doc,target):
    s=setup(doc,target);s.update(modulation_1_ramp_up=modulation.RAMP_OFF,
                              modulation_1_ramp_down=modulation.RAMP_OFF+1e-7)
    result=convert(VitalPreset(doc),template)
    assert result.report.modulations[0]['status']=='mapped'


def test_ramp_rejection_preserves_other_mix_and_macro_trajectory(template,doc):
    s=setup(doc,'distortion_mix');s.update(delay_on=1,macro_control_2=.3,macro_control_3=.4,
        osc_1_distortion_type=4,osc_1_distortion_amount=.2,osc_1_wave_frame=10,
        modulation_1_ramp_up=1,modulation_2_amount=.4,modulation_3_amount=.75,modulation_4_amount=.6)
    s['modulations'] += [{'source':'macro_control_2','destination':'delay_dry_wet'},
        {'source':'macro_control_3','destination':'osc_1_wave_frame'},
        {'source':'macro_control_3','destination':'osc_1_distortion_amount'}]
    v=VitalPreset(doc);result=convert(v,template)
    assert v.folded=={1}
    assert next(m for m in result.report.modulations if m['destination']=='delay_dry_wet')['status']=='mapped'
    assert [m['joint_trajectory'] for m in result.report.modulations if m.get('joint_trajectory')]==['osc_1_','osc_1_']


@pytest.mark.parametrize('target',TARGETS)
def test_zero_amount_aux_leaf_does_not_restrict_mix_source(template,doc,target):
    s=setup(doc,target);s.update(modulation_2_amount=0,modulation_2_ramp_up=1)
    s['modulations'].append({'source':'pitch_wheel','destination':'modulation_1_amount'})
    v=VitalPreset(doc);mods=modulation.active_modulations(v);into=modulation.amount_modulators(mods)
    assert modulation.mix_route_supported(v,mods,into,1)
    result=convert(VitalPreset(doc),template)
    assert result.report.modulations[0]['status']=='mapped'
    assert not any(n['feature']=='modulation_2_ramp' for n in result.report.notes)


@pytest.mark.parametrize('target',TARGETS)
def test_zero_amount_aux_parent_with_active_child_still_restricts_mix(template,doc,target):
    s=setup(doc,target);s.update(modulation_2_amount=0,modulation_2_ramp_down=1,modulation_3_amount=.2)
    s['modulations'] += [{'source':'macro_control_2','destination':'modulation_1_amount'},
                         {'source':'macro_control_3','destination':'modulation_2_amount'}]
    result=convert(VitalPreset(doc),template)
    assert not any(z.get('destModuleTypeString')==MODULES[target] for k,z in result.preset.state.items() if k.startswith('ModSlot'))
    assert any(n['feature']=='modulation_2_ramp' and n['kind']=='unsupported' for n in result.report.notes)
