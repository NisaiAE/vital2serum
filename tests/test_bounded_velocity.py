import copy
import math
from unittest.mock import patch
import pytest
from vital2serum import voice_gain
from vital2serum.mapping import convert
from vital2serum.modulation import KNOWN_IDS
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset
from test_voice_gain import source,previous,velocity_route,assert_fallback

@pytest.mark.parametrize('track',[-.001,-.01,-.1])
@pytest.mark.parametrize('factor',[1.1,1.25,1.5])
@pytest.mark.parametrize('effects',[False,True])
@pytest.mark.parametrize('embed',[False,True])
def test_bounded_negative_tracking_preserves_normalized_curve_and_existing_route(template,doc,track,factor,effects,embed):
    source(doc,factor,effects)
    doc['settings'].update(velocity_track=track,macro_control_1=.37,filter_1_on=1,
        filter_1_model=0,filter_1_style=0,filter_1_drive=20,filter_1_resonance=0,
        modulations=[dict(source='env_2',destination='filter_1_cutoff')],modulation_1_amount=.1)
    original,old,ctx=previous(doc,template,embed=embed);key,route=velocity_route(old.preset)
    result=convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    # The source PCM stores binary32; use its actual common gain rather than
    # assuming a decimal coefficient survives serialization exactly.
    factor=ctx.fx_gain if effects else ctx.master_boost
    top=max((1-track)**2,1);assert factor*top<=2
    expected=copy.deepcopy(old.preset.state);amount=plain(route)['kParamAmount']*factor;expected[key]['plainParams']['kParamAmount']=amount
    if effects:expected['FXRack0']['FX'].pop(0)
    else:expected['Global0']['plainParams']['kParamMasterVolume']/=factor
    assert result.preset.state==expected
    assert result.preset.state[key]['flex']==route['flex'] and result.preset.state[key]['source']==[16,0]
    assert plain(result.preset.state['Global0'])['kParamVoiceAmp']==0
    assert amount<=100 and amount/100<=1
    note=next(n for n in result.report.notes if n['feature']=='levels')
    assert note['velocity_amplitude_depth']==amount/100 and note['velocity_normalization_top']==top
    assert 'normalized velocity curve' in note['message']
    assert result.assets==old.assets and result.report.assets==old.report.assets
    assert result.report.mapped==old.report.mapped and result.report.modulations==old.report.modulations
    assert [n for n in result.report.notes if n['feature']!='levels']==[n for n in old.report.notes if n['feature']!='levels']
    curve=route['flex'][0]
    for i in range(4097):
        v=i/4096;k=min(31,int(v*32));u=v*32-k;g=1-((1-u)*curve['yVals'][k]+u*curve['yVals'][k+1])
        assert 1/top-2e-16<=g<=1
        old_amp=plain(route)['kParamAmount']/100*g;new_amp=amount/100*g
        assert 0<=new_amp<=1 and new_amp/factor==pytest.approx(old_amp,abs=2e-16)
    reused=VitalPreset(copy.deepcopy(doc));a=convert(reused,template,embed=embed);b=convert(reused,template,embed=embed)
    assert a.preset.to_bytes()==b.preset.to_bytes()==result.preset.to_bytes()
    assert a.report.as_dict()==b.report.as_dict()==result.report.as_dict() and reused.doc==doc

@pytest.mark.parametrize('track',[-.01,-.1,-.25])
@pytest.mark.parametrize('branch',['master','effects'])
def test_complete_coefficient_boundary_has_no_extra_tolerance(template,doc,track,branch):
    source(doc,effects=branch=='effects');doc['settings']['velocity_track']=track
    original,_,ctx=previous(doc,template);top=(1-track)**2;limit=2/top
    high=math.nextafter(limit,math.inf)
    while high*top<=2:high=math.nextafter(high,math.inf)
    for factor in (high,limit+1e-8):
        rejected=copy.deepcopy(ctx)
        if branch=='master':rejected.master_boost=factor
        else:rejected.fx_gain=factor
        assert factor*top>2
        assert_fallback(rejected,original)
    # Below the bound, supply the corresponding complete old output gain;
    # changing Context alone would instead test the restoration-ownership guard.
    factor=math.nextafter(limit,-math.inf)
    _,route=velocity_route(ctx.s)
    while factor*top>2 or plain(route)['kParamAmount']*factor>100:factor=math.nextafter(factor,-math.inf)
    if branch=='master':
        ctx.master_boost=factor;ctx.s.state['Global0']['plainParams']['kParamMasterVolume']=voice_gain.master_value(original.get('volume'),factor)
    else:
        ctx.fx_gain=factor;ctx.s.state['FXRack0']['FX'][0]['FXEQ']['plainParams']['kParamLevelOut']=.5*math.sqrt(factor)
    assert voice_gain.apply(ctx,original)
    _,route=velocity_route(ctx.s);assert plain(route)['kParamAmount']<=100

@pytest.mark.parametrize('failure',['id','bool_id','base','curve','amount','aux_source','aux_curve','bipolar','other_amp','incoming_zero','unknown_incoming','original_volume','moving_track','negative_amp','multislot'])
def test_negative_owned_response_failure_keeps_all_objects_assets_and_report(template,doc,failure):
    source(doc,effects=True);doc['settings']['velocity_track']=-.1
    original,_,ctx=previous(doc,template);key,route=velocity_route(ctx.s);options={}
    if failure in ('id','bool_id'):
        ids=dict(KNOWN_IDS);ids[('Global','kParamVoiceAmp')]=3 if failure=='id' else True;options['destination_ids']=ids
    elif failure=='base':plain(ctx.s.state['Global0'])['kParamVoiceAmp']=.01
    elif failure=='curve':route['flex'][0]['yVals'][2]+=.001
    elif failure=='amount':plain(route)['kParamAmount']=50.
    elif failure=='aux_source':route['source'][1]=6
    elif failure=='aux_curve':route['flex'][1]={'numPoints':1}
    elif failure=='bipolar':plain(route)['kParamBipolar']=1.
    elif failure in ('other_amp','multislot'):ctx.s.state['ModSlot63']=copy.deepcopy(route)
    elif failure in ('incoming_zero','unknown_incoming'):
        ctx.s.state['ModSlot63']=dict(source=[6,0],destModuleTypeString='ModSlot' if failure=='incoming_zero' else 'ModSlot0',destModuleID=int(key[7:]),destModuleParamName='kParamAmount',plainParams={'kParamAmount':0.})
    elif failure=='negative_amp':plain(route)['kParamAmount']=-plain(route)['kParamAmount']
    else:
        target='volume' if failure=='original_volume' else 'velocity_track'
        original.settings.update(modulations=[dict(source='unknown_source',destination=target),dict(source='lfo_1',destination='modulation_1_amount')],modulation_1_amount=0,modulation_2_amount=.5)
    assert_fallback(ctx,original,**options)

@pytest.mark.parametrize('track',[-1.,-.5,-.25])
def test_multiple_original_routes_or_amplitude_above_one_is_not_repaired(template,doc,track):
    source(doc,1.5);doc['settings']['velocity_track']=track
    original,old,ctx=previous(doc,template)
    assert max((1-track)**2,1)*ctx.master_boost>2
    assert_fallback(ctx,original)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==old.preset.to_bytes() and result.report.as_dict()==old.report.as_dict()


def test_representable_product_does_not_allow_an_amount_above_one_hundred(template,doc):
    source(doc);track=-1e-5;doc['settings']['velocity_track']=track
    original,_,ctx=previous(doc,template);top=(1-track)**2;factor=2/top
    key,route=velocity_route(ctx.s)
    assert factor*top==2 and plain(route)['kParamAmount']*factor>100
    ctx.master_boost=factor
    ctx.s.state['Global0']['plainParams']['kParamMasterVolume']=voice_gain.master_value(original.get('volume'),factor)
    assert_fallback(ctx,original)
