import copy
import math
from unittest.mock import patch
import pytest
from vital2serum import voice_gain
from vital2serum.mapping import convert
from vital2serum.modulation import KNOWN_IDS
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset
from test_voice_gain import source, previous


def setup(doc,template,factor=1.5,volume=6400,**kwargs):
    source(doc,factor);doc['settings']['volume']=volume
    original,old,ctx=previous(doc,template,**kwargs)
    note=next(n for n in ctx.r.notes if 'gain' in n and 'output_stage' in n)
    return original,old,ctx,note


def unchanged(ctx,original,**kwargs):
    preset,report,assets=ctx.s,ctx.r,ctx.assets
    state=copy.deepcopy(preset.state);details=copy.deepcopy(report.__dict__)
    assert not voice_gain.apply(ctx,original,**kwargs)
    assert ctx.s is preset and ctx.r is report and ctx.assets is assets
    assert ctx.s.state==state and ctx.r.__dict__==details


@pytest.mark.parametrize('factor',[1.5,3.,4.])
@pytest.mark.parametrize('volume',[6400.,7399.4404])
@pytest.mark.parametrize('embed',[False,True])
def test_complete_owned_static_suffix_uses_only_original_master_gain(template,doc,factor,volume,embed):
    original,old,ctx,owner=setup(doc,template,factor,volume,embed=embed)
    original_master=voice_gain.master_value(original.get('volume'))
    result=convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    expected=copy.deepcopy(old.preset.state)
    expected['Global0']['plainParams']['kParamVoiceAmp']=min(1.,factor/2)
    if factor>2:
        slots=[(k,m) for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')=='kParamVoiceAmp']
        assert len(slots)==1;key,slot=slots[0];expected[key]=copy.deepcopy(slot)
        assert plain(slot)['kParamAmount']==100*(factor/2-1)
        assert slot['flex'][0]['yVals']==[0.,0.]
    expected['Global0']['plainParams']['kParamMasterVolume']=min(1.,original_master)
    expected['FXRack0']['FX']=expected['FXRack0']['FX'][:owner['output_stage']]
    if original_master>1:
        expected['FXRack0']['FX'].append({'FXEQ':{'plainParams':{'kParamEnable':1.,'kParamLevelOut':.5*math.sqrt(original_master)}},'kUIParamMixOrGain':0.,'type':7})
        tail=next(n for n in result.report.notes if 'gain' in n and 'output_stage' in n)
        assert tail['gain']==original_master and tail['output_stage']==owner['output_stage']
        assert result.report.mapped==old.report.mapped
    else:
        assert 'master_output_gain' not in result.report.mapped
        assert result.report.mapped==[x for x in old.report.mapped if x!='master_output_gain']
        assert not any('output_stage' in n for n in result.report.notes)
    assert result.preset.state==expected
    output=plain(result.preset.state['Global0'])['kParamMasterVolume']
    output*=math.prod((2*plain(stage['FXEQ'])['kParamLevelOut'])**2 for stage in result.preset.state['FXRack0']['FX'][owner['output_stage']:])
    assert output==pytest.approx(original_master,rel=2e-16)
    assert result.assets==old.assets and result.report.assets==old.report.assets
    assert result.report.modulations==old.report.modulations
    assert result.report.as_dict()['counts']['approximation']==sum(n['kind']=='approximation' for n in result.report.notes)
    assert (ctx.master_boost,ctx.fx_gain)==(factor,1.)
    reused=VitalPreset(copy.deepcopy(doc));a=convert(reused,template,embed=embed);b=convert(reused,template,embed=embed)
    assert a.preset.to_bytes()==b.preset.to_bytes()==result.preset.to_bytes()
    assert a.report.as_dict()==b.report.as_dict()==result.report.as_dict() and reused.doc==doc


@pytest.mark.parametrize('gain',[math.nextafter(4.,-math.inf),4.,math.nextafter(4.,math.inf),5.65646355781488])
def test_minimal_old_cascade_boundary_retains_prefix_and_position_semantics(template,doc,gain):
    original,_,ctx,owner=setup(doc,template,3,7399.4404)
    original_master=voice_gain.master_value(original.get('volume'));factor=gain/original_master
    assert 1<factor<=4
    gain=voice_gain.master_value(original.get('volume'),factor)
    ctx.master_boost=factor;owner['gain']=gain
    prefix={'FXChorus':{'plainParams':{'kParamWet':.2}},'type':3,'kUIParamMixOrGain':0.}
    ctx.s.state['FXRack0']['FX']=[prefix]+[{'FXEQ':{'plainParams':{'kParamEnable':1.,'kParamLevelOut':.5*math.sqrt(x)}},'type':7,'kUIParamMixOrGain':0.} for x in voice_gain.gain_stages(gain,9)]
    owner['output_stage']=1
    before=copy.deepcopy(ctx.s.state);flags=(ctx.master_boost,ctx.fx_gain);assets=ctx.assets
    assert voice_gain.apply(ctx,original)
    assert ctx.s.state['FXRack0']['FX'][0]==prefix
    assert len(ctx.s.state['FXRack0']['FX'])==2
    for key in before:
        if key not in ('Global0','FXRack0','ModSlot0'):assert ctx.s.state[key]==before[key]
    assert ctx.assets is assets and (ctx.master_boost,ctx.fx_gain)==flags
    assert next(n for n in ctx.r.notes if 'output_stage' in n)['output_stage']==1


@pytest.mark.parametrize('gain,count',[(math.nextafter(4.,-math.inf),1),(4.,1),(math.nextafter(4.,math.inf),2)])
def test_stage_product_boundary_is_exact(gain,count):
    stages=voice_gain.gain_stages(gain,2)
    assert len(stages)==count and math.prod(stages)==gain


@pytest.mark.parametrize('failure',[
    'note_missing','note_duplicate','gain','gain_nan','stage_bool','stage_index','note_extra',
    'wrong_type','type_bool','enable_bool','level_bool','ui_bool','level','extra_field',
    'extra_stage','incomplete_cascade','rack_shape','unknown_prefix','null_body',
    'tail_active','tail_inactive','tail_zero_aux','tail_report','tail_nested_report',
    'unknown_report_ref','bad_report_id','second_rack','old_volume','unknown_slot','slot_exhausted',
])
def test_unknown_owner_structure_or_any_tail_reference_is_whole_fallback(template,doc,failure):
    original,_,ctx,owner=setup(doc,template,3.,7399.4404)
    rack=ctx.s.state['FXRack0']['FX'];stage=rack[owner['output_stage']];params=plain(stage['FXEQ'])
    if failure=='note_missing':ctx.r.notes.remove(owner)
    elif failure=='note_duplicate':ctx.r.notes.append(copy.deepcopy(owner))
    elif failure=='gain':owner['gain']+=.01
    elif failure=='gain_nan':owner['gain']=math.nan
    elif failure=='stage_bool':owner['output_stage']=True
    elif failure=='stage_index':owner['output_stage']+=1
    elif failure=='note_extra':owner['unknown_ref']=0
    elif failure=='wrong_type':stage['type']=8
    elif failure=='type_bool':stage['type']=True
    elif failure=='enable_bool':params['kParamEnable']=True
    elif failure=='level_bool':params['kParamLevelOut']=True
    elif failure=='ui_bool':stage['kUIParamMixOrGain']=False
    elif failure=='level':params['kParamLevelOut']+=1e-12
    elif failure=='extra_field':params['kParamFreq1']=100.
    elif failure=='extra_stage':rack.append(copy.deepcopy(stage))
    elif failure=='incomplete_cascade':rack.pop()
    elif failure=='rack_shape':ctx.s.state['FXRack0']['FX']=None
    elif failure=='unknown_prefix':rack.insert(0,{'FXUnknown':{},'type':99});owner['output_stage']+=1
    elif failure=='null_body':stage['FXEQ']=None
    elif failure in ('tail_active','tail_inactive','tail_zero_aux'):
        ctx.s.state['ModSlot63']=dict(source=[0,0] if failure=='tail_inactive' else [25,6] if failure=='tail_zero_aux' else [25,0],destModuleTypeString='FXEQ',destModuleID=owner['output_stage'],destModuleParamName='kParamLevelOut',destModuleParamID=KNOWN_IDS['FXEQ','kParamLevelOut'],plainParams={'kParamAmount':0.})
    elif failure in ('tail_report','tail_nested_report'):
        ref=dict(module='FXEQ',module_id=owner['output_stage'],parameter='kParamLevelOut',parameter_id=KNOWN_IDS['FXEQ','kParamLevelOut'],slot=0)
        ctx.r.modulations.append({'atomic_law_destinations':[ref]} if failure=='tail_nested_report' else {'serum_module':'FXEQ','serum_module_id':owner['output_stage'],'serum_parameter':'kParamLevelOut'})
    elif failure=='unknown_report_ref':ctx.r.notes.append(dict(kind='info',feature='unknown',message='unknown schema',module_id=0))
    elif failure=='bad_report_id':ctx.r.modulations.append(dict(serum_module='Global',serum_module_id=0,serum_parameter='kParamVoiceAmp',serum_parameter_id=None))
    elif failure=='second_rack':ctx.s.state['FXRack1']['FX']=[copy.deepcopy(stage)]
    elif failure=='old_volume':original.settings.update(modulations=[dict(source='env_2',destination='volume')],modulation_1_amount=0.)
    elif failure=='unknown_slot':ctx.s.state['ModSlot63'].update(unknownReference=0)
    elif failure=='slot_exhausted':
        for key,module in ctx.s.state.items():
            if key.startswith('ModSlot'):module['plainParams']={'kParamAmount':0.}
    unchanged(ctx,original)


def test_master_without_owned_tail_never_enters_cleanup(template,doc):
    source(doc,1.5);original,_,ctx=previous(doc,template)
    before=copy.deepcopy(ctx.s.state)
    with patch.object(voice_gain,'_tail_cleanup',side_effect=AssertionError('no cleanup needed')):
        assert voice_gain.apply(ctx,original)
    expected=copy.deepcopy(before);expected['Global0']['plainParams']['kParamVoiceAmp']=.75;expected['Global0']['plainParams']['kParamMasterVolume']/=1.5
    assert ctx.s.state==expected and 'master_output_gain' not in ctx.r.mapped
