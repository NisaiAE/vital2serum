import pytest
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def setup_tempo(doc):
    s=doc['settings']
    s.update(lfo_1_sync=1,lfo_1_tempo=8,lfo_1_sync_type=0,
             modulations=[{'source':'env_2','destination':'lfo_1_tempo'}],
             modulation_1_amount=1/3,modulation_1_bipolar=0)
    return s


def tempo_slots(result):
    return [m for name,m in result.preset.state.items() if name.startswith('ModSlot')
            and m.get('destModuleTypeString')=='LFO' and m.get('destModuleParamName')=='kParamRate']


def test_bounded_linear_tempo_uses_native_boundaries_and_continuous_phase(template,doc):
    setup_tempo(doc)
    result=convert(VitalPreset(doc),template)
    p=plain(result.preset.state['LFO0']);slots=tempo_slots(result)
    assert p['kParamBeatSync']==1 and p['kParamAnchored']==0
    assert p['kParamRate']==pytest.approx(100*((7+.3-1/6)/14)**4)
    assert len(slots)==1 and slots[0]['destModuleParamID']==0
    assert plain(slots[0])['kParamAmount']==pytest.approx(100*4/14)
    assert not slots[0].get('flex')
    note=next(n for n in result.report.notes if n['feature']=='lfo_1_tempo')
    assert note['raw_base']==8 and note['raw_end']==12


def test_tempo_keeps_fraction_from_folded_fixed_control(template,doc):
    s=setup_tempo(doc)
    s['modulation_1_amount']=.2
    s['modulations'].append({'source':'macro_control_1','destination':'lfo_1_tempo'})
    s.update(macro_control_1=.45,modulation_2_amount=1/12)
    result=convert(VitalPreset(doc),template)
    p=plain(result.preset.state['LFO0'])
    assert p['kParamRate']==pytest.approx(100*((8.45-1+.3-1/6)/14)**4)
    assert len(tempo_slots(result))==1


@pytest.mark.parametrize('base,amount',[(1,.1),(12,-.1)])
def test_tempo_range_endpoints_without_clipping(template,doc,base,amount):
    s=setup_tempo(doc);s.update(lfo_1_tempo=base,modulation_1_amount=amount)
    result=convert(VitalPreset(doc),template)
    assert len(tempo_slots(result))==1
    assert plain(tempo_slots(result)[0])['kParamAmount']==pytest.approx(100*amount*12/14)


@pytest.mark.parametrize('condition',[
    'free','freeze','dotted','triplet','upper_clamp','lower_clamp','aux','multiple',
    'stereo','bipolar','power','curve','dynamic_sync','dynamic_mode','source_gain',
])
def test_tempo_excludes_unverified_control_combinations(template,doc,condition):
    s=setup_tempo(doc)
    if condition=='free':s['lfo_1_sync']=0
    elif condition=='freeze':s['lfo_1_tempo']=0
    elif condition=='dotted':s['lfo_1_sync']=2
    elif condition=='triplet':s['lfo_1_sync']=3
    elif condition=='upper_clamp':s['modulation_1_amount']=.34
    elif condition=='lower_clamp':s.update(lfo_1_tempo=2,modulation_1_amount=-.2)
    elif condition=='aux':
        s['modulations'].append({'source':'macro_control_1','destination':'modulation_1_amount'})
        s['modulation_2_amount']=.2
    elif condition=='multiple':
        s['modulations'].append({'source':'env_3','destination':'lfo_1_tempo'})
        s['modulation_2_amount']=.05
    elif condition=='stereo':s['modulation_1_stereo']=1
    elif condition=='bipolar':s['modulation_1_bipolar']=1
    elif condition=='power':s['modulation_1_power']=1
    elif condition=='curve':
        s['modulations'][0]['line_mapping']={'num_points':2,'points':[0,0,1,1],'powers':[0,0]}
    elif condition in ('dynamic_sync','dynamic_mode'):
        s['modulations'].append({'source':'env_3','destination':'lfo_1_'+('sync' if condition=='dynamic_sync' else 'sync_type')})
        s['modulation_2_amount']=.1
    else:
        s['modulations'][0]['source']='lfo_2'
        s.update(lfo_2_sync=0,lfo_2_frequency=6,lfo_2_smooth_mode=1,lfo_2_smooth_time=-7.5)
    result=convert(VitalPreset(doc),template)
    assert tempo_slots(result)==[]
    assert 'kParamAnchored' not in plain(result.preset.state['LFO0'])


def test_unmodulated_sync_preserves_original_rate_and_anchor_default(template,doc):
    s=setup_tempo(doc);s['modulations']=[]
    result=convert(VitalPreset(doc),template)
    p=plain(result.preset.state['LFO0'])
    assert p['kParamRate']==6.25 and 'kParamAnchored' not in p
