import copy
import pytest
from vital2serum.mapping import Context
from vital2serum.mapping import convert
from vital2serum import modulation
from vital2serum.modulation import (_Writer, _plan, Destination, KNOWN_IDS, Modulation)
from vital2serum.report import Report
from vital2serum.serum import clean_template, plain
from vital2serum.vital import VitalPreset


def setup(doc,template):
    ctx=Context(VitalPreset(copy.deepcopy(doc)),clean_template(template),Report('',''))
    return ctx,_Writer(ctx,KNOWN_IDS)


def factor(index,source,*,bipolar=False,points=()):
    return Modulation(index,source,'modulation_1_amount',.5,bipolar,0,False,points)


def products():
    return {i:factor(i,f'lfo_{i}') for i in range(1,5)}


def test_failed_deep_product_does_not_allocate_a_prefix(doc,template):
    ctx,writer=setup(doc,template);mods=products();writer.free=[4,5]
    assert writer.bus(mods,(1,2,3,4)) is None
    assert writer.next==0 and writer.free==[4,5] and writer.buses=={}
    assert plain(ctx.s.state['Macro4'])=={}


def test_failed_extension_preserves_an_existing_product(doc,template):
    ctx,writer=setup(doc,template);mods=products()
    assert writer.bus(mods,(1,2))==29
    saved=copy.deepcopy(ctx.s.state['ModSlot0']);cache=dict(writer.buses);writer.free=[5]
    assert writer.bus(mods,(1,2,3,4)) is None
    assert writer.next==1 and writer.free==[5] and writer.buses==cache
    assert ctx.s.state['ModSlot0']==saved


@pytest.mark.parametrize('first,success',[(61,True),(62,False)])
def test_product_reserves_a_slot_for_its_final_destination(doc,template,first,success):
    ctx,writer=setup(doc,template);mods=products();writer.next=first
    bus=writer.bus(mods,(1,2,3))
    assert (bus is not None)==success
    if success:
        assert writer.next==63
        assert writer.slot(6,bus,Destination('Oscillator',0,'kParamPan'),2,10,False)==63
    else:
        assert writer.next==first and writer.free==[4,5,6,7] and writer.buses=={}
        assert plain(ctx.s.state['Macro4'])=={}


def test_cached_product_needs_no_new_macro_at_the_last_slot(doc,template):
    ctx,writer=setup(doc,template);mods=products()
    bus=writer.bus(mods,(1,2));writer.free=[];writer.next=63
    assert writer.can_bus(mods,[(1,2)])
    assert writer.bus(mods,(1,2))==bus
    assert writer.slot(6,bus,Destination('Oscillator',0,'kParamPan'),2,10,False)==63


def test_fixed_zero_term_does_not_consume_the_complementary_product(doc,template):
    ctx,writer=setup(doc,template);writer.free=[4,5]
    mods={1:factor(1,'lfo_1'),2:factor(2,'lfo_2'),3:factor(3,'mod_wheel'),
          4:factor(4,'mod_wheel',points=((0,0),(1,1)))}
    planned=_plan(ctx.v,mods,[(1,(1,2,3)),(1,(1,2,4))],set(),writer,ctx.r,'amount')
    assert planned==[(1,29,None)]
    assert writer.next==1 and writer.free==[5]
    assert plain(ctx.s.state['Macro4'])['kParamValue']==0
    assert plain(ctx.s.state['ModSlot0'])['kParamAmount']==100
    assert ctx.s.state['ModSlot0']['source']==[6,7]
    assert any('changing those controls' in n['message'] for n in ctx.r.notes)


def test_bipolar_fixed_factor_is_folded_after_its_constant_is_split(doc,template):
    doc['settings']['mod_wheel']=.25
    ctx,writer=setup(doc,template);writer.free=[4]
    mods={1:factor(1,'lfo_1'),2:factor(2,'lfo_2'),3:factor(3,'mod_wheel',bipolar=True)}
    assert _plan(ctx.v,mods,[(1,(1,2,3))],set(),writer,ctx.r,'amount')==[(-.25,29,None)]
    assert writer.next==1


def test_a_moving_control_is_not_folded_when_the_budget_is_short(doc,template):
    ctx,writer=setup(doc,template);writer.free=[4]
    mods={1:factor(1,'lfo_1'),2:factor(2,'lfo_2'),3:factor(3,'mod_wheel')}
    assert _plan(ctx.v,mods,[(1,(1,2,3))],{'mod_wheel'},writer,ctx.r,'amount')==[]
    assert writer.next==0 and writer.free==[4] and writer.buses=={}


def test_available_budget_preserves_fixed_control_operation(doc,template):
    ctx,writer=setup(doc,template)
    mods={1:factor(1,'lfo_1'),2:factor(2,'lfo_2'),3:factor(3,'mod_wheel')}
    assert _plan(ctx.v,mods,[(1,(1,2,3))],set(),writer,ctx.r,'amount')==[(1,30,None)]
    assert writer.next==2
    assert ctx.s.state['ModSlot1']['source']==[29,1]
    assert not ctx.r.notes


def test_zero_coefficients_allocate_no_slots_or_macros(doc,template):
    ctx,writer=setup(doc,template);mods=products()
    assert _plan(ctx.v,mods,[(0,(1,2,3))],set(),writer,ctx.r,'amount')==[]
    assert writer.next==0 and writer.buses=={}


def test_fixed_zero_preserves_an_oscillator_volume_endpoint(doc,template,monkeypatch):
    monkeypatch.setattr(modulation,'BUS_MACROS',(4,))
    s=doc['settings'];s['osc_1_level']=0
    s['modulations']=[dict(source='env_2',destination='osc_1_level'),
        dict(source='lfo_1',destination='modulation_1_amount'),
        dict(source='lfo_2',destination='modulation_2_amount'),
        dict(source='mod_wheel',destination='modulation_3_amount')]
    for i,amount in enumerate((0,0,0,.125),1):s[f'modulation_{i}_amount']=amount
    result=convert(VitalPreset(doc),template)
    assert plain(result.preset.state['Oscillator0'])['kParamVolume']==0
    assert not any(m.get('destModuleTypeString') for n,m in result.preset.state.items() if n.startswith('ModSlot'))
    assert plain(result.preset.state['Macro4'])=={}
