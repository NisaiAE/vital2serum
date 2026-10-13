import pytest
from vital2serum import modulation
from vital2serum.curves import sampled_curve
from vital2serum.mapping import Context
from vital2serum.modulation import Destination, KNOWN_IDS, convert_modulations
from vital2serum.report import Report
from vital2serum.serum import clean_template, plain
from vital2serum.vital import VitalPreset


def context(doc,template,filler=0):
    doc['settings'].update(modulations=[dict(source='env_2',destination='distortion_drive'),
        dict(source='env_3',destination='modulation_1_amount')],
        modulation_1_amount=0,modulation_2_amount=.5)
    for index in range(3,min(filler,62)+3):
        doc['settings']['modulations'].append(dict(source='lfo_1',destination='osc_1_pan'))
        doc['settings'][f'modulation_{index}_amount']=.01
    ctx=Context(VitalPreset(doc),clean_template(template),Report('',''))
    if filler>62:
        ctx.fixed_routings=[('env_2',Destination('Oscillator',0,'kParamPan'),1,False)]
    ctx.target('osc_1_pan','Oscillator',0,'kParamPan')
    callbacks=[]
    output=Destination('FXEQ',0,'kParamLevelOut')
    main=sampled_curve(lambda x:x)
    auxiliary=sampled_curve(lambda x:min(1,2*x))
    result=sampled_curve(lambda x:x*x)
    ctx.complete_control_buses={1:(.25,'env_2','env_3',main,auxiliary,output,30,result,
                                  lambda:callbacks.append('written'),lambda:callbacks.append('omitted'))}
    return ctx,callbacks


def test_complete_control_uses_one_macro_and_keeps_auxiliary_records(doc,template):
    ctx,callbacks=context(doc,template)
    convert_modulations(ctx,KNOWN_IDS)
    assert callbacks==['written']
    inputs=ctx.s.state['ModSlot0'];output=ctx.s.state['ModSlot1']
    assert inputs['source']==[3,4]
    assert inputs['destModuleTypeString']=='Macro' and inputs['destModuleID']==4
    assert plain(inputs)['kParamAmount']==100
    assert plain(ctx.s.state['Macro4'])['kParamValue']==25
    assert output['source']==[29,0]
    assert (output['destModuleTypeString'],output['destModuleParamName'],output['destModuleParamID'])==('FXEQ','kParamLevelOut',9)
    assert plain(output)['kParamAmount']==30
    assert all(record['status']=='mapped' for record in ctx.r.modulations)
    assert ctx.r.modulations[1]['aux_for_slot']==1
    assert ctx.r.modulations[0]['complete_control_bus']==dict(macro=4,input_slot=0,output_slot=1)


@pytest.mark.parametrize('filler,expected',[(62,'written'),(63,'omitted')])
def test_complete_control_reserves_both_slots_after_existing_routes(doc,template,filler,expected):
    ctx,callbacks=context(doc,template,filler)
    convert_modulations(ctx,KNOWN_IDS)
    assert callbacks==[expected]
    primary=next(record for record in ctx.r.modulations if record['vital_slot']==1)
    assert primary['status']==('mapped' if expected=='written' else 'omitted')
    assert sum(m.get('destModuleTypeString')=='Oscillator' for name,m in ctx.s.state.items() if name.startswith('ModSlot'))==filler
    if expected=='written':
        assert primary['complete_control_bus']['input_slot']==62
        assert primary['complete_control_bus']['output_slot']==63
    else:
        assert not any(m.get('destModuleTypeString')=='Macro' for name,m in ctx.s.state.items() if name.startswith('ModSlot'))
        assert plain(ctx.s.state['Macro4'])=={}


def test_complete_control_without_free_macro_preserves_all_state(doc,template,monkeypatch):
    ctx,callbacks=context(doc,template)
    monkeypatch.setattr(modulation,'BUS_MACROS',())
    convert_modulations(ctx,KNOWN_IDS)
    assert callbacks==['omitted']
    assert not any(m.get('destModuleTypeString') for name,m in ctx.s.state.items() if name.startswith('ModSlot'))
    assert plain(ctx.s.state['Macro4'])=={}
