import copy
import pytest
from vital2serum.mapping import Context, convert
from vital2serum.modulation import (_Writer, _order_phase_buses, Destination, KNOWN_IDS)
from vital2serum.report import Report
from vital2serum.serum import clean_template, plain
from vital2serum.vital import VitalPreset


def setup(doc,template):
    ctx=Context(VitalPreset(copy.deepcopy(doc)),clean_template(template),Report('',''))
    return ctx,_Writer(ctx,KNOWN_IDS)


def test_phase_updates_precede_products_and_their_consumers_stably(doc,template):
    ctx,w=setup(doc,template)
    w.slot(6,3,Destination('Macro',4,'kParamValue'),0,100,False)
    w.slot(29,4,Destination('Macro',5,'kParamValue'),0,100,False)
    w.slot(8,30,Destination('Oscillator',0,'kParamFine'),5,25,False)
    w.slot(16,0,Destination('Oscillator',0,'kParamPan'),2,10,False)
    w.slot(7,0,Destination('LFO',0,'kParamPhase'),4,100,False)
    old=[ctx.s.state[f'ModSlot{i}'] for i in range(5)]
    extras=[{'slot':2}];bus={'macro':4,'input_slot':0,'output_slot':2}
    ctx.r.modulations=[dict(serum_slot=2,serum_slots=[0,2],supplementary_destinations=extras,complete_control_bus=bus),
                       dict(serum_slot=2,serum_slots=[2],supplementary_destinations=extras,complete_control_bus=bus)]
    _order_phase_buses(w)
    assert [ctx.s.state[f'ModSlot{i}'] for i in range(5)]==[old[i] for i in (3,4,0,1,2)]
    assert ctx.r.modulations[0]['serum_slot']==4
    assert ctx.r.modulations[0]['serum_slots']==[2,4]
    assert extras==[{'slot':4}] and bus=={'macro':4,'input_slot':2,'output_slot':4}
    assert w.next==5


def test_phase_macro_feedback_retains_its_original_order(doc,template):
    ctx,w=setup(doc,template)
    w.slot(6,3,Destination('Macro',4,'kParamValue'),0,100,False)
    w.slot(29,0,Destination('LFO',0,'kParamPhase'),4,100,False)
    old=copy.deepcopy(ctx.s.state)
    _order_phase_buses(w)
    assert ctx.s.state==old
    assert any('feedback cycle' in n['message'] for n in ctx.r.notes)


def test_unrelated_phase_routes_keep_existing_order(doc,template):
    ctx,w=setup(doc,template)
    w.slot(6,3,Destination('Macro',4,'kParamValue'),0,100,False)
    w.slot(29,0,Destination('Oscillator',0,'kParamPan'),2,10,False)
    w.slot(7,0,Destination('LFO',2,'kParamPhase'),4,100,False)
    old=copy.deepcopy(ctx.s.state)
    _order_phase_buses(w)
    assert ctx.s.state==old and not ctx.r.notes


def test_phase_reordering_does_not_allocate_or_omit_at_the_slot_limit(doc,template):
    ctx,w=setup(doc,template)
    w.slot(6,3,Destination('Macro',4,'kParamValue'),0,100,False)
    w.slot(29,0,Destination('Oscillator',0,'kParamFine'),5,25,False)
    for _ in range(61):w.slot(16,0,Destination('Oscillator',0,'kParamPan'),2,10,False)
    w.slot(7,0,Destination('LFO',0,'kParamPhase'),4,100,False)
    _order_phase_buses(w)
    assert w.next==64
    assert ctx.s.state['ModSlot61']['destModuleTypeString']=='LFO'
    assert ctx.s.state['ModSlot62']['destModuleTypeString']=='Macro'
    assert ctx.s.state['ModSlot63']['source']==[29,0]
    assert sum(m.get('destModuleTypeString')=='Oscillator' for n,m in ctx.s.state.items() if n.startswith('ModSlot'))==62


@pytest.mark.parametrize('poly',[1,8])
def test_frozen_envelope_phase_is_written_before_an_amount_product(doc,template,poly):
    s=doc['settings'];s.update(polyphony=poly,lfo_1_sync=1,lfo_1_tempo=0,lfo_1_sync_type=2,
        lfo_2_sync=0,lfo_2_frequency=0,lfo_2_sync_type=2)
    s['modulations']=[dict(source='lfo_3',destination='osc_1_tune'),
        dict(source='lfo_1',destination='modulation_1_amount'),
        dict(source='env_2',destination='modulation_2_amount'),
        dict(source='lfo_2',destination='lfo_1_phase')]
    for i,amount in enumerate((0,0,.25,.5),1):s[f'modulation_{i}_amount']=amount
    result=convert(VitalPreset(doc),template)
    active=[(i,result.preset.state[f'ModSlot{i}']) for i in range(64) if result.preset.state[f'ModSlot{i}'].get('destModuleTypeString')]
    phase=next(i for i,m in active if m.get('destModuleTypeString')=='LFO')
    product=next(i for i,m in active if m.get('destModuleTypeString')=='Macro')
    destination=next(i for i,m in active if m.get('destModuleTypeString')=='Oscillator')
    assert phase<product<destination
    assert plain(result.preset.state['LFO0'])['kParamMode']=='Envelope'
    assert plain(result.preset.state['LFO0'])['kParamRate']==0
    for record in result.report.modulations:
        if record['status']=='mapped':
            for index in record['serum_slots']:
                assert result.preset.state[f'ModSlot{index}']['destModuleParamName']==record['serum_parameter']
