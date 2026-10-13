import copy,math
from bisect import bisect_right
from types import SimpleNamespace
import pytest
from vital2serum import mapping,modulation,post_master
from vital2serum.mapping import convert
from vital2serum.vital import VitalPreset
from vital2serum.serum import SerumPreset,clean_template


def setup(doc,source='macro_control_1'):
    doc['settings'].update(volume=4900,velocity_track=0,macro_control_1=.6,mod_wheel=.6,
        modulations=[{'source':source,'destination':'volume'}],modulation_1_amount=-.1)
    return doc['settings']

def value(curve,x):
    xs,ys=curve['xVals'],curve['yVals'];i=min(len(xs)-2,max(0,bisect_right(xs,x)-1));t=(x-xs[i])/(xs[i+1]-xs[i]);return 1-(ys[i]+t*(ys[i+1]-ys[i]))

@pytest.mark.parametrize('source',['macro_control_1','macro_control_4','mod_wheel'])
@pytest.mark.parametrize('base,amount',[(4900,-.1),(6400,.25),(7399.4404,-.2)])
def test_output_master_curve_and_unowned_global_params(template,doc,source,base,amount):
    s=setup(doc,source);s.update(volume=base,modulation_1_amount=amount);result=convert(VitalPreset(doc),template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    route=result.report.modulations[0];assert route['status']=='mapped' and route['serum_module']=='FXEQ'
    assert not any(k.startswith('ModSlot') and d.get('destModuleTypeString')=='Global' and d.get('destModuleParamName')=='kParamVoiceAmp' for k,d in result.preset.state.items())
    actual=SerumPreset.from_bytes(result.preset.to_bytes());index=route['serum_module_id'];p=actual.state['FXRack0']['FX'][index]['FXEQ']['plainParams'];master=actual.state['Global0']['plainParams']['kParamMasterVolume']
    assert {k:v for k,v in actual.state['Global0']['plainParams'].items() if k!='kParamMasterVolume'}=={k:v for k,v in ref.preset.state['Global0']['plainParams'].items() if k!='kParamMasterVolume'}
    for x in (0,.000001,.001,.2,.6,.999999,1):
        position=p['kParamLevelOut']
        for entry in route['atomic_law_destinations']:
            d=actual.state[f"ModSlot{entry['slot']}"];position+=d['plainParams']['kParamAmount']/100*value(d['flex'][0],x)
        raw=max(0,min(7399.4404,base+7399.4404*amount*x));wanted=post_master.master_value(raw)
        assert master*(2*position)**2==pytest.approx(wanted,abs=2e-6)
    assert len(actual.state['FXRack0']['FX'])==len(ref.preset.state['FXRack0']['FX'])+1

@pytest.mark.parametrize('bad',['lfo','envelope','velocity','bipolar','stereo','moving_source','same_aux','moving_aux','primary_ramp','aux_ramp','multiple'])
def test_unsupported_response_retains_original_conversion(template,doc,bad):
    s=setup(doc)
    if bad=='lfo':s['modulations'][0]['source']='lfo_1'
    elif bad=='envelope':s['modulations'][0]['source']='env_2'
    elif bad=='velocity':s['velocity_track']=.5
    elif bad in ('bipolar','stereo'):s['modulation_1_'+bad]=1
    elif bad=='primary_ramp':s['modulation_1_ramp_up']=0
    elif bad=='multiple':s['modulations'].append({'source':'macro_control_2','destination':'volume'});s['modulation_2_amount']=.1
    elif bad=='moving_source':s['modulations'].append({'source':'env_2','destination':'macro_control_1'});s['modulation_2_amount']=.1
    else:
        s.update(macro_control_2=.5,modulation_2_amount=.2)
        s['modulations'].append({'source':'macro_control_1' if bad=='same_aux' else 'lfo_1' if bad=='moving_aux' else 'macro_control_2','destination':'modulation_1_amount'})
        if bad=='aux_ramp':s['modulation_2_ramp_down']=0
    result=convert(VitalPreset(copy.deepcopy(doc)),template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    assert result.preset.to_bytes()==ref.preset.to_bytes();assert result.report.as_dict()==ref.report.as_dict()

@pytest.mark.parametrize('failure',['id','slot','curve','source_gain','registration','direct'])
def test_failed_complete_response_returns_pure_original(template,doc,monkeypatch,failure):
    s=setup(doc);s.update(delay_on=1,delay_style=0,delay_dry_wet=.25,macro_control_2=.4,modulation_2_amount=.3)
    s['modulations'].append({'source':'macro_control_2','destination':'delay_dry_wet'});options={}
    if failure=='id':
        ids=dict(modulation.KNOWN_IDS);ids.pop(('FXEQ','kParamLevelOut'));options['destination_ids']=ids
    elif failure=='slot':monkeypatch.setattr(modulation,'MAX_SLOTS',0)
    elif failure=='curve':
        original=modulation._adaptive_law_points
        def no_master(fn,m,p):
            if m.target=='volume':raise ValueError('budget')
            return original(fn,m,p)
        monkeypatch.setattr(modulation,'_adaptive_law_points',no_master)
    elif failure in ('source_gain','registration'):
        original=post_master.register
        def rejected(c,ids,disabled=False):
            if failure=='source_gain':c.source_gain['macro_control_1']=.5
            if failure=='registration' and not disabled:c.post_master_failed=True;c.post_master_failure='Resources unavailable';return False
            return original(c,ids,disabled)
        monkeypatch.setattr(post_master,'register',rejected)
    elif failure=='direct':s['osc_1_destination']=4
    result=convert(VitalPreset(copy.deepcopy(doc)),template,**options);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True,**options)
    assert result.preset.to_bytes()==ref.preset.to_bytes();assert result.report.modulations==ref.report.modulations
    if failure!='slot':assert [r for r in result.report.modulations if r.get('destination')=='delay_dry_wet'][0]['status']=='mapped'
    assert [(a.data,a.gain) for a in result.assets]==[(a.data,a.gain) for a in ref.assets]


def test_fixed_aux_and_context_boost_share_one_complete_curve(template,doc):
    s=setup(doc);s.update(macro_control_2=.25,modulation_2_amount=.1);s['modulations'].append({'source':'macro_control_2','destination':'modulation_1_amount'})
    result=convert(VitalPreset(doc),template);route=result.report.modulations[0];assert route['serum_module']=='FXEQ'
    p=result.preset.state['FXRack0']['FX'][-1]['FXEQ']['plainParams'];master=result.preset.state['Global0']['plainParams']['kParamMasterVolume']
    for x in (0,.2,.6,1):
        position=p['kParamLevelOut']+sum(e['amount']/100*value(result.preset.state[f"ModSlot{e['slot']}"]['flex'][0],x) for e in route['atomic_law_destinations'])
        assert master*(2*position)**2==pytest.approx(post_master.master_value(4900+7399.4404*(-.05)*x),abs=2e-6)
    assert any(n['feature']=='modulation_1_static_aux' for n in result.report.notes)

@pytest.mark.parametrize('volume',[0,4900,6400])
def test_unneeded_static_stage_is_byte_identical(template,doc,volume):
    doc['settings'].update(volume=volume,velocity_track=0)
    result=convert(VitalPreset(copy.deepcopy(doc)),template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    assert result.preset.to_bytes()==ref.preset.to_bytes();assert result.report.as_dict()==ref.report.as_dict()


def test_static_stage_restores_gain_and_removes_only_recovered_clip(template,doc):
    doc['settings'].update(volume=7399.4404,velocity_track=0)
    result=convert(VitalPreset(doc),template);p=result.preset.state['FXRack0']['FX'][-1]['FXEQ']['plainParams']
    assert result.preset.state['Global0']['plainParams']['kParamMasterVolume']==1
    assert (2*p['kParamLevelOut'])**2==pytest.approx(post_master.master_value(7399.4404))
    assert not any(n['feature']=='master_volume' and n['kind']=='clipped' for n in result.report.notes)

@pytest.mark.parametrize('axis',['primary_range','aux_range','remap_y','remap_x'])
def test_unit_domain_guards_keep_original_response(template,doc,monkeypatch,axis):
    s=setup(doc)
    if axis=='primary_range':monkeypatch.setitem(modulation.RANGES,'macro_control_1',(-1,1))
    elif axis=='aux_range':
        s.update(macro_control_2=.3,modulation_2_amount=.1);s['modulations'].append({'source':'macro_control_2','destination':'modulation_1_amount'});monkeypatch.setitem(modulation.RANGES,'macro_control_2',(-1,1))
    else:
        s['modulations'][0]['line_mapping']={'points':([0,1,.5,-.1,1,0] if axis=='remap_y' else [-.1,1,.5,.5,1,0]),'powers':[0,0]}
    result=convert(VitalPreset(copy.deepcopy(doc)),template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    assert result.preset.to_bytes()==ref.preset.to_bytes();assert result.report.as_dict()==ref.report.as_dict()

@pytest.mark.parametrize('power',[-1.0,0.0,1.0])
@pytest.mark.parametrize('remap',[False,True])
def test_shaped_control_gain_stays_within_reachable_bound(template,doc,power,remap):
    s=setup(doc);s.update(volume=6400,modulation_1_amount=.1,modulation_1_power=power)
    if remap:s['modulations'][0]['line_mapping']={'points':[0,.8,.4,.1,.7,.6,1,.2],'powers':[0,0,0]}
    v=VitalPreset(doc);ctrl,reason=post_master.control(v,modulation.KNOWN_IDS);assert ctrl and not reason
    mod=modulation.active_modulations(v)[1];hi=7399.4404;maximum=max(v.get('volume'),min(hi,v.get('volume')+hi*ctrl.coefficient))
    for i in range(4097):
        shaped=mod.value(i/4096);assert 0<=shaped<=1
        raw=max(0,min(hi,v.get('volume')+hi*ctrl.coefficient*shaped));assert raw<=maximum
        assert post_master.master_value(raw)<=post_master.master_value(maximum)
    result=convert(v,template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    if any(r.get('serum_module')=='FXEQ' for r in result.report.modulations):
        route=result.report.modulations[0];index=route['serum_module_id'];base=result.preset.state['FXRack0']['FX'][index]['FXEQ']['plainParams']['kParamLevelOut'];master=result.preset.state['Global0']['plainParams']['kParamMasterVolume']
        for i in range(4097):
            x=i/4096;position=base+sum(e['amount']/100*value(result.preset.state[f"ModSlot{e['slot']}"]['flex'][0],x) for e in route['atomic_law_destinations']);raw=max(0,min(hi,6400+hi*.1*mod.value(x)))
            assert master*(2*position)**2==pytest.approx(post_master.master_value(raw),abs=3e-6)
    else:assert result.preset.to_bytes()==ref.preset.to_bytes()

@pytest.mark.parametrize('boost,applied',[(1,True),(2.8,True),(2.9,True)])
def test_static_context_compensation_and_headroom(template,doc,boost,applied):
    from vital2serum.report import Report
    doc['settings'].update(volume=7399.4404,velocity_track=0);v=VitalPreset(doc);state=SimpleNamespace(state={'Global0':{'plainParams':{'kParamMasterVolume':1.0}},'FXRack0':{'FX':[]}})
    report=Report('test');report.note('clipped','master_volume','clip');report.note('clipped','osc_1_level','clip')
    c=SimpleNamespace(v=v,s=state,r=report,levels={},master_boost=boost,fx_gain=2.5,source_gain={},destinations={},static_aux_law_targets=set())
    original=copy.deepcopy(state.state);assert post_master.register(c,modulation.KNOWN_IDS)==applied
    if applied:
        gain=math.prod((2*e['FXEQ']['plainParams']['kParamLevelOut'])**2 for e in state.state['FXRack0']['FX']);assert gain==pytest.approx(post_master.master_value(7399.4404,boost));assert not any(n['feature']=='master_volume' and n['kind']=='clipped' for n in report.notes)
    else:assert state.state==original;assert any(n['feature']=='master_volume' and n['kind']=='clipped' for n in report.notes)
    assert any(n['feature']=='osc_1_level' and n['kind']=='clipped' for n in report.notes)


def test_reused_input_restores_original_offsets_and_report_after_fallback(template,doc,monkeypatch):
    setup(doc);original=modulation._adaptive_law_points
    monkeypatch.setattr(modulation,'_adaptive_law_points',lambda *args:(_ for _ in ()).throw(ValueError('budget')))
    v=VitalPreset(copy.deepcopy(doc));first=convert(v,template);second=convert(v,template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    assert first.preset.to_bytes()==second.preset.to_bytes()==ref.preset.to_bytes();assert first.report.as_dict()==second.report.as_dict();assert v.offsets=={} and v.folded==set()
    monkeypatch.setattr(modulation,'_adaptive_law_points',original);success=convert(v,template);fresh=convert(VitalPreset(copy.deepcopy(doc)),template);assert success.preset.to_bytes()==fresh.preset.to_bytes();assert success.report.as_dict()==fresh.report.as_dict()

@pytest.mark.parametrize('raw',[-1,7399.4405,9000])
def test_outside_requested_source_range_keeps_original_conversion(template,doc,raw):
    doc['settings'].update(volume=raw,velocity_track=0)
    result=convert(VitalPreset(copy.deepcopy(doc)),template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    assert result.preset.to_bytes()==ref.preset.to_bytes();assert result.report.as_dict()==ref.report.as_dict()


def test_fold_beyond_source_range_keeps_original_state_and_offset(template,doc):
    s=setup(doc);s.update(volume=7000,velocity_track=.5,modulation_1_amount=.2)
    v=VitalPreset(copy.deepcopy(doc));result=convert(v,template);refv=VitalPreset(copy.deepcopy(doc));ref=convert(refv,template,_disabled_post_master=True)
    assert v.offsets==refv.offsets and v.folded==refv.folded=={1}
    assert result.preset.to_bytes()==ref.preset.to_bytes();assert result.report.as_dict()==ref.report.as_dict()

@pytest.mark.parametrize('gain,count',[(2,1),(4,1),(4+1e-12,2),(8,2),(16,2),(16+1e-12,3),(20,3),(4**10,10)])
def test_cascade_minimal_stage_product_and_native_bound(gain,count):
    stages=post_master.gain_stages(gain,10)
    assert len(stages)==count and all(0<g<=4 for g in stages)
    assert stages[:-1]==(4.,)*(count-1)
    assert math.prod(stages)==pytest.approx(gain,rel=2e-15)
    assert post_master.gain_stages(gain,count-1) is None

@pytest.mark.parametrize('gain',[0,-1,float('inf'),float('-inf'),float('nan'),4**10+1])
def test_cascade_nonfinite_or_unavailable_product_is_rejected(gain):
    assert post_master.gain_stages(gain,10) is None


def boosted_levels(monkeypatch,gain):
    original=mapping._levels
    def apply(c):
        original(c);c.master_boost=gain/post_master.master_value(6400.)
    monkeypatch.setattr(mapping,'_levels',apply)


def existing_rack(monkeypatch,count):
    original=mapping.convert_effects
    def apply(c):
        original(c);rack=c.s.state['FXRack0']['FX']
        while len(rack)<count:rack.append({'FXEQ':{'plainParams':{'kParamEnable':1.,'kParamLevelOut':.5}},'kUIParamMixOrGain':0.,'type':7})
    monkeypatch.setattr(mapping,'convert_effects',apply)

@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('gain,prefix,stages',[(20,7,3),(8,8,2),(4**10,0,10)])
def test_complete_cascade_uses_rack_boundary_and_last_control_only(template,doc,monkeypatch,dynamic,gain,prefix,stages):
    s=setup(doc) if dynamic else doc['settings'];s.update(volume=6400,velocity_track=0)
    if dynamic:s['modulation_1_amount']=-.1
    boosted_levels(monkeypatch,gain);existing_rack(monkeypatch,prefix)
    r=convert(VitalPreset(copy.deepcopy(doc)),template);fx=r.preset.state['FXRack0']['FX'];assert len(fx)==prefix+stages==10
    assert all(x['FXEQ']['plainParams']['kParamLevelOut']==1 for x in fx[prefix:-1])
    if dynamic:
        route=[x for x in r.report.modulations if x['destination']=='volume'][0];assert route['status']=='mapped' and route['serum_module_id']==9
        assert {x['destModuleID'] for k,x in r.preset.state.items() if k.startswith('ModSlot') and x.get('destModuleTypeString')=='FXEQ'}=={9}
        base=fx[-1]['FXEQ']['plainParams']['kParamLevelOut'];master=r.preset.state['Global0']['plainParams']['kParamMasterVolume'];fixed=4**(stages-1)
        for i in range(257):
            x=i/256;position=base+sum(e['amount']/100*value(r.preset.state[f"ModSlot{e['slot']}"]['flex'][0],x) for e in route['atomic_law_destinations'])
            wanted=post_master.master_value(6400-7399.4404*.1*x,gain/post_master.master_value(6400))
            assert master*fixed*(2*position)**2==pytest.approx(wanted,rel=2e-6)
    else:assert math.prod((2*x['FXEQ']['plainParams']['kParamLevelOut'])**2 for x in fx[prefix:])==pytest.approx(gain)

@pytest.mark.parametrize('failure',['rack','curve','slot','source_gain','direct'])
def test_cascade_complete_failure_restores_other_mix_and_trajectory(template,doc,monkeypatch,failure):
    s=setup(doc);s.update(volume=6400,modulation_1_amount=-.1,delay_on=1,delay_style=0,delay_dry_wet=.25,macro_control_2=.4,modulation_2_amount=.3,modulation_3_amount=.5,modulation_4_amount=.4,osc_1_distortion_type=4,osc_1_distortion_amount=.2,osc_1_wave_frame=0)
    s['modulations'] += [{'source':'macro_control_2','destination':'delay_dry_wet'},{'source':'env_2','destination':'osc_1_wave_frame'},{'source':'env_2','destination':'osc_1_distortion_amount'}]
    boosted_levels(monkeypatch,20.)
    if failure=='rack':existing_rack(monkeypatch,8)
    elif failure=='curve':
        old=modulation._adaptive_law_points
        def fail(fn,m,p):
            if m.target=='volume':raise ValueError('curve budget')
            return old(fn,m,p)
        monkeypatch.setattr(modulation,'_adaptive_law_points',fail)
    elif failure=='slot':monkeypatch.setattr(modulation,'MAX_SLOTS',0)
    elif failure=='source_gain':
        old=post_master.register
        def fail(c,ids,disabled=False):c.source_gain['macro_control_1']=.5;return old(c,ids,disabled)
        monkeypatch.setattr(post_master,'register',fail)
    elif failure=='direct':s['osc_1_destination']=4
    v=VitalPreset(copy.deepcopy(doc));r=convert(v,template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    assert r.preset.to_bytes()==ref.preset.to_bytes() and r.report.modulations==ref.report.modulations
    assert [(x.data,x.gain) for x in r.assets]==[(x.data,x.gain) for x in ref.assets]
    if failure not in ('slot','direct'):
        assert next(x for x in r.report.modulations if x['destination']=='delay_dry_wet')['status']=='mapped'
        assert any(x.get('serum_parameter')=='kParamTablePos' and x.get('status')=='mapped' for x in r.report.modulations)
    assert any(x['feature']=='master_volume' and x['kind']=='clipped' for x in r.report.notes)

@pytest.mark.parametrize('prefix',[8,9])
def test_static_cascade_rack_shortage_has_no_partial_stages(template,doc,monkeypatch,prefix):
    doc['settings'].update(volume=6400,velocity_track=0);boosted_levels(monkeypatch,20.);existing_rack(monkeypatch,prefix)
    r=convert(VitalPreset(copy.deepcopy(doc)),template);ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True)
    assert len(r.preset.state['FXRack0']['FX'])==prefix
    assert r.preset.to_bytes()==ref.preset.to_bytes() and r.report.modulations==ref.report.modulations
