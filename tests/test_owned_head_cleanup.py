import copy
import pytest
from test_voice_gain import source,previous,assert_fallback
from vital2serum import voice_gain


def routed(doc,template):
    source(doc,effects=True)
    original,old,ctx=previous(doc,template)
    rack=ctx.s.state['FXRack0']['FX']
    assert rack[1]['type']==0
    for _ in range(2):
        rack.append(dict(FXFilter=dict(plainParams=dict(kParamEnable=1.,kParamFreq=500.,kParamReso=25.)),type=8,kUIParamMixOrGain=0.))
    rack.append(dict(FXEQ=dict(plainParams=dict(kParamEnable=1.,kParamLevelOut=.6)),type=7,kUIParamMixOrGain=0.))
    for slot,kind,index,parameter,number in [(0,'FXDistortion',1,'kParamFreq',5),
                                          (1,'FXFilter',2,'kParamFreq',3),(2,'FXFilter',3,'kParamReso',4),
                                          (3,'FXEQ',4,'kParamLevelOut',9)]:
        ctx.s.state[f'ModSlot{slot}']=dict(source=[3,0],destModuleTypeString=kind,destModuleID=index,
            destModuleParamName=parameter,destModuleParamID=number,
            plainParams=dict(kParamAmount=13.,kParamBipolar=0.))
        ctx.r.modulations.append(dict(vital_slot=slot+1,status='mapped',serum_slot=slot,serum_slots=[slot],
            serum_module=kind,serum_module_id=index,serum_parameter=parameter,serum_parameter_id=number))
    item=dict(slot=3,module='FXEQ',module_id=4,parameter='kParamLevelOut',parameter_id=9,amount=13.)
    ctx.r.modulations[0]['atomic_law_destinations']=[copy.deepcopy(item)]
    ctx.r.modulations[1]['supplementary_destinations']=[copy.deepcopy(item)]
    ctx.r.notes.append(dict(kind='approximation',feature='master_volume',output_stage=4))
    ctx.r.notes.append(dict(kind='approximation',feature='filter_1_structure',stages=['clean two-pole','driven two-pole']))
    return original,ctx


def test_distinct_same_kind_stages_and_report_paths_keep_their_meaning(template,doc):
    original,ctx=routed(doc,template)
    before=copy.deepcopy(ctx.s.state);details=copy.deepcopy(ctx.r.as_dict());assets=ctx.assets
    flags=(ctx.fx_gain,ctx.master_boost)
    assert voice_gain.apply(ctx,original)
    assert ctx.s.state['FXRack0']['FX']==before['FXRack0']['FX'][1:]
    assert ctx.assets is assets and (ctx.fx_gain,ctx.master_boost)==flags
    for slot in range(4):
        old=before[f'ModSlot{slot}'];new=ctx.s.state[f'ModSlot{slot}']
        expected=copy.deepcopy(old);expected['destModuleID']-=1
        assert new==expected
        assert ctx.s.state['FXRack0']['FX'][new['destModuleID']]==before['FXRack0']['FX'][old['destModuleID']]
        assert ctx.r.modulations[slot]['serum_module_id']==details['modulations'][slot]['serum_module_id']-1
    for name,index in [('atomic_law_destinations',0),('supplementary_destinations',1)]:
        assert ctx.r.modulations[index][name][0]['module_id']==3
    assert next(n for n in ctx.r.notes if 'output_stage' in n)['output_stage']==3
    assert next(n for n in ctx.r.notes if 'stages' in n)['stages']==['clean two-pole','driven two-pole']
    for key,value in before.items():
        if key not in {'Global0','FXRack0','ModSlot0','ModSlot1','ModSlot2','ModSlot3'}:assert ctx.s.state[key]==value


@pytest.mark.parametrize('change',[
    'unknown_kind','container','modslot_container','inactive_dest','index_bool','index_float','index_negative',
    'index_range','head_target','param_bool','param_unknown','type_bool','type_mismatch','kind_mismatch',
    'ui_nonzero','ui_bool','enable_bool','level_bool','slot_unknown_field','head_type_bool','head_extra','other_rack','unknown_rack','unknown_slot_name',
    'report_kind','report_index_bool','report_index_range','report_slot','report_param','report_unknown_path',
    'output_head','output_bool','output_kind','output_feature','source_bool','source_unknown','missing_dest',
])
def test_unknown_or_inconsistent_graph_has_whole_fallback(template,doc,change):
    original,ctx=routed(doc,template)
    slot=ctx.s.state['ModSlot0'];rack=ctx.s.state['FXRack0']['FX'];record=ctx.r.modulations[0]
    if change=='unknown_kind':slot['destModuleTypeString']='FXUnknown'
    elif change=='container':slot['destModuleTypeString']='FXRack'
    elif change=='modslot_container':slot['destModuleTypeString']='ModSlot'
    elif change=='inactive_dest':slot['source']=[0,0]
    elif change=='index_bool':slot['destModuleID']=True
    elif change=='index_float':slot['destModuleID']=1.
    elif change=='index_negative':slot['destModuleID']=-1
    elif change=='index_range':slot['destModuleID']=len(rack)
    elif change=='head_target':slot.update(destModuleTypeString='FXEQ',destModuleID=0,destModuleParamName='kParamLevelOut',destModuleParamID=9)
    elif change=='param_bool':slot['destModuleParamID']=True
    elif change=='param_unknown':slot['destModuleParamID']=99
    elif change=='type_bool':rack[1]['type']=False
    elif change=='type_mismatch':rack[1]['type']=8
    elif change=='kind_mismatch':slot['destModuleTypeString']='FXFilter'
    elif change=='ui_nonzero':rack[0]['kUIParamMixOrGain']=1.
    elif change=='ui_bool':rack[0]['kUIParamMixOrGain']=False
    elif change=='enable_bool':rack[0]['FXEQ']['plainParams']['kParamEnable']=True
    elif change=='level_bool':rack[0]['FXEQ']['plainParams']['kParamLevelOut']=True
    elif change=='slot_unknown_field':slot['destModuleRackID']=0
    elif change=='head_type_bool':rack[0]['type']=True
    elif change=='head_extra':rack[0]['FXEQ']['plainParams']['kParamGain2']=0.
    elif change=='other_rack':ctx.s.state['FXRack1']['FX']=[copy.deepcopy(rack[1])]
    elif change=='unknown_rack':ctx.s.state['FXRack3']={'FX':[]}
    elif change=='unknown_slot_name':ctx.s.state['ModSlot000']=ctx.s.state.pop('ModSlot0')
    elif change=='report_kind':record['serum_module']='FXFilter'
    elif change=='report_index_bool':record['serum_module_id']=True
    elif change=='report_index_range':record['serum_module_id']=len(rack)
    elif change=='report_slot':record['serum_slot']=1
    elif change=='report_param':record['serum_parameter_id']=99
    elif change=='report_unknown_path':ctx.r.notes.append(dict(kind='info',feature='unknown',module='FXEQ',module_id=4))
    elif change=='output_head':ctx.r.notes[-2]['output_stage']=0
    elif change=='output_bool':ctx.r.notes[-2]['output_stage']=True
    elif change=='output_kind':ctx.r.notes[-2]['output_stage']=2
    elif change=='output_feature':ctx.r.notes[-2]['feature']='unknown'
    elif change=='source_bool':slot['source']=[True,0]
    elif change=='source_unknown':slot['source']=[999,0]
    elif change=='missing_dest':slot.pop('destModuleTypeString')
    assert_fallback(ctx,original)


@pytest.mark.parametrize('kind,index,parameter,number',[('WTOsc',0,'kParamWarp',0),('WTOsc',2,'kParamTablePos',6),('NoiseOsc',3,'kParamColor',0)])
@pytest.mark.parametrize('wrong_owner',[False,True])
def test_non_fx_nested_owner_uses_the_destination_index(template,doc,kind,index,parameter,number,wrong_owner):
    original,ctx=routed(doc,template)
    owner=ctx.s.state[f'Oscillator{index}'];child=f'{kind}{index}'
    assert child in owner
    ctx.s.state['ModSlot8']=dict(source=[3,0],destModuleTypeString=kind,destModuleID=index,
        destModuleParamName=parameter,destModuleParamID=number,plainParams=dict(kParamAmount=17.,kParamBipolar=0.))
    if wrong_owner:
        owner[f'{kind}{(index+1)%4}']=owner.pop(child)
        assert_fallback(ctx,original)
    else:
        old=copy.deepcopy(ctx.s.state['ModSlot8']);body=copy.deepcopy(owner)
        assert voice_gain.apply(ctx,original)
        assert ctx.s.state['ModSlot8']==old and ctx.s.state[f'Oscillator{index}']==body


@pytest.mark.parametrize('name,index',[('atomic_law_destinations',0),('supplementary_destinations',1)])
@pytest.mark.parametrize('number',[None,True,9.,'9',99])
def test_secondary_report_ids_require_verified_integers(template,doc,name,index,number):
    original,ctx=routed(doc,template)
    ctx.r.modulations[index][name][0]['parameter_id']=number
    assert_fallback(ctx,original)


@pytest.mark.parametrize('number',[None,True,5.,'5',99])
def test_explicit_primary_report_id_cannot_be_null_or_unknown(template,doc,number):
    original,ctx=routed(doc,template)
    ctx.r.modulations[0]['serum_parameter_id']=number
    assert_fallback(ctx,original)


def test_primary_without_optional_report_number_matches_the_actual_slot(template,doc):
    original,ctx=routed(doc,template)
    ctx.r.modulations[0].pop('serum_parameter_id')
    assert voice_gain.apply(ctx,original)
    assert ctx.r.modulations[0]['serum_module_id']==0


@pytest.mark.parametrize('kind,index',[('Global',1),('Oscillator',4),('WTOsc',3),('NoiseOsc',0),('VoiceFilter',2),('Env',4),('LFO',8),('Macro',8)])
def test_dummy_module_does_not_expand_the_verified_non_fx_domain(template,doc,kind,index):
    original,ctx=routed(doc,template)
    if kind in ('WTOsc','NoiseOsc'):
        ctx.s.state[f'Oscillator{index}'][f'{kind}{index}']={'plainParams':'default'}
    else:ctx.s.state[f'{kind}{index}']={'plainParams':'default'}
    from vital2serum.modulation import KNOWN_IDS
    parameter,number=next((p,n) for (k,p),n in KNOWN_IDS.items() if k==kind)
    ctx.s.state['ModSlot8']=dict(source=[3,0],destModuleTypeString=kind,destModuleID=index,
        destModuleParamName=parameter,destModuleParamID=number,plainParams=dict(kParamAmount=17.))
    assert_fallback(ctx,original)


@pytest.mark.parametrize('change',['dummy_owner','slot_conflict','bad_parameter','null_id','bool_id','float_id','unknown_id','absent_id'])
def test_non_fx_report_and_native_slot_must_describe_the_same_owner(template,doc,change):
    original,ctx=routed(doc,template)
    record=ctx.r.modulations[0]
    if change=='dummy_owner':
        ctx.s.state['Global999']={'plainParams':'default'}
        record.update(serum_module='Global',serum_module_id=999,serum_parameter='kParamVoiceAmp',serum_parameter_id=2)
    elif change=='slot_conflict':record.update(serum_module='Global',serum_module_id=0,serum_parameter='kParamVoiceAmp',serum_parameter_id=2)
    else:
        slot=ctx.s.state['ModSlot0'];slot.update(destModuleTypeString='Global',destModuleID=0,destModuleParamName='kParamPortamentoTime',destModuleParamID=3)
        record.update(serum_module='Global',serum_module_id=0,serum_parameter='kParamPortamentoTime',serum_parameter_id=3)
        if change=='bad_parameter':record['serum_parameter']='notKnown'
        elif change=='null_id':record['serum_parameter_id']=None
        elif change=='bool_id':record['serum_parameter_id']=True
        elif change=='float_id':record['serum_parameter_id']=3.
        elif change=='unknown_id':record['serum_parameter_id']=65535
        elif change=='absent_id':record.pop('serum_parameter_id')
    if change=='absent_id':assert voice_gain.apply(ctx,original)
    else:assert_fallback(ctx,original)


@pytest.mark.parametrize('field',['atomic_law_destinations','supplementary_destinations'])
def test_shared_writer_secondary_dict_is_shifted_once_for_all_records(template,doc,field):
    original,ctx=routed(doc,template)
    shared=ctx.r.modulations[0]['atomic_law_destinations']
    for record in ctx.r.modulations[:3]:record[field]=shared
    assert shared[0]['module_id']==4
    assert voice_gain.apply(ctx,original)
    records=ctx.r.modulations[:3]
    assert records[0][field] is records[1][field] is records[2][field]
    for record in records:
        item=record[field][0]
        assert item['module_id']==ctx.s.state[f"ModSlot{item['slot']}"]['destModuleID']==3
        assert ctx.s.state['FXRack0']['FX'][item['module_id']]['type']==7
    assert [r['serum_module_id'] for r in records]==[0,1,2]


def test_shared_secondary_item_across_list_kinds_retains_single_stage_identity(template,doc):
    original,ctx=routed(doc,template)
    item=ctx.r.modulations[0]['atomic_law_destinations'][0]
    ctx.r.modulations[1]['supplementary_destinations']=[item]
    assert voice_gain.apply(ctx,original)
    first=ctx.r.modulations[0]['atomic_law_destinations'][0]
    second=ctx.r.modulations[1]['supplementary_destinations'][0]
    assert first is second and first['module_id']==3
    assert ctx.s.state[f"ModSlot{first['slot']}"]['destModuleID']==first['module_id']
