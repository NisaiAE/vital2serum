import math
import pytest
from vital2serum.curves import envelope_tail_curve, lfo_curve
from vital2serum.mapping import convert
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def test_envelope_tail_preserves_point_times_and_terminal_value():
    points=[(0.,1.),(.8,0.),(1.,0.),(1.,1.)]
    curve,q,tail=envelope_tail_curve(points,[0.,2.,0.,0.],False,6.,.0025)
    assert tail>12*.0025
    assert q==pytest.approx(.998/(1+12*.0025*6))
    assert curve['yVals'][-2:]==[0.,0.]
    assert curve['xVals'][-2:]==[q,1.]
    # The x=0 wrap entry has no duration; every real point retains its original time.
    assert curve['xVals'][2]/(6*q)==pytest.approx(.8/6)
    assert curve['curveVals'][2]==lfo_curve(points[:-1],[0.,2.,0.],False)['curveVals'][2]
    assert curve['numPoints']==len(curve['xVals'])-1
    assert len(curve['xVals'])==len(curve['yVals'])==len(curve['curveVals'])


def test_envelope_tail_retains_sampled_shape_smoothing():
    points=[(0.,1.),(.5,0.),(1.,1.)]
    original=lfo_curve(points,[0.,0.,0.],True)
    curve,q,_=envelope_tail_curve(points,[0.,0.,0.],True,2.,0.)
    assert curve['xVals'][:-1]==pytest.approx([x*q for x in original['xVals']])
    assert curve['yVals'][:-1]==original['yVals']
    assert curve['curveVals'][:-1]==original['curveVals']
    assert curve['yVals'][-1]==1.


def setup_envelope(doc):
    s=doc['settings']
    s.update(lfo_1_sync=0,lfo_1_sync_type=2,lfo_1_frequency=math.log2(8),lfo_1_phase=0,
             lfo_1_smooth_mode=1,lfo_1_smooth_time=-7.5,
             modulations=[{'source':'lfo_1','destination':'osc_1_transpose'}],modulation_1_amount=.1)
    s['lfos']=[{'num_points':4,'points':[0.,1.,.8,0.,1.,0.,1.,1.],'powers':[0.,0.,0.,0.]}]
    return s


def test_envelope_tail_uses_effective_rate_and_omits_periodic_depth_gain(template,doc):
    s=setup_envelope(doc)
    s.update(macro_control_1=.5,modulation_2_amount=.1)
    s['modulations'].append({'source':'macro_control_1','destination':'lfo_1_frequency'})
    v=VitalPreset(doc);result=convert(v,template)
    note=next(n for n in result.report.notes if n['feature']=='lfo_1_terminal_hold')
    assert note['original_hz']==pytest.approx(2**(3+.5*.1*16))
    p=plain(result.preset.state['LFO0']);q=note['point_scale']
    assert p['kParamRate']==pytest.approx(note['original_hz']*q)
    assert p['kParamMode']=='Envelope' and p['kParamSmooth']>=.5
    assert plain(result.preset.state['ModSlot0'])['kParamAmount']==pytest.approx(.1*100*96/128)
    assert not any(n['feature']=='lfo_1_smooth_time' for n in result.report.notes)


@pytest.mark.parametrize('excluded',['trigger','sync','freeze','phase','dynamic_rate','dynamic_phase','dynamic_smooth'])
def test_envelope_tail_excludes_moving_or_other_clock_modes(template,doc,excluded):
    s=setup_envelope(doc)
    if excluded=='trigger':s['lfo_1_sync_type']=0
    elif excluded=='sync':s.update(lfo_1_sync=1,lfo_1_tempo=7)
    elif excluded=='freeze':s.update(lfo_1_sync=1,lfo_1_tempo=0)
    elif excluded=='phase':s['lfo_1_phase']=.25
    else:
        dest={'dynamic_rate':'frequency','dynamic_phase':'phase','dynamic_smooth':'smooth_time'}[excluded]
        s['modulations'].append({'source':'env_2','destination':'lfo_1_'+dest});s['modulation_2_amount']=.2
    result=convert(VitalPreset(doc),template)
    assert not any(n['feature']=='lfo_1_terminal_hold' for n in result.report.notes)
    assert result.preset.state['LFO0']['curveData']['xVals'][-2:]==[1.,1.]


@pytest.mark.parametrize('chain',['direct','aux','macro','macro_aux','lfo_control'])
def test_phase_consumers_exclude_envelope_tail_through_indirect_paths(template,doc,chain):
    s=setup_envelope(doc)
    if chain=='direct':
        extra=[{'source':'lfo_1','destination':'lfo_2_phase'}]
    elif chain=='aux':
        extra=[{'source':'lfo_3','destination':'lfo_2_phase'},
               {'source':'lfo_1','destination':'modulation_2_amount'}]
    elif chain=='macro':
        extra=[{'source':'lfo_1','destination':'macro_control_1'},
               {'source':'macro_control_1','destination':'lfo_2_phase'}]
    elif chain=='macro_aux':
        extra=[{'source':'lfo_1','destination':'macro_control_1'},
               {'source':'lfo_3','destination':'lfo_2_phase'},
               {'source':'macro_control_1','destination':'modulation_3_amount'}]
    else:
        extra=[{'source':'lfo_1','destination':'lfo_3_frequency'},
               {'source':'lfo_3','destination':'lfo_2_phase'}]
    s['modulations']+=extra
    for j in range(2,2+len(extra)):s[f'modulation_{j}_amount']=.2
    result=convert(VitalPreset(doc),template)
    assert not any(n['feature']=='lfo_1_terminal_hold' for n in result.report.notes)
    assert plain(result.preset.state['LFO0'])['kParamRate']==8.


def test_bypassed_phase_route_does_not_exclude_envelope_tail(template,doc):
    s=setup_envelope(doc)
    s['modulations'].append({'source':'lfo_1','destination':'lfo_2_phase'})
    s.update(modulation_2_amount=1,modulation_2_bypass=1)
    result=convert(VitalPreset(doc),template)
    assert any(n['feature']=='lfo_1_terminal_hold' for n in result.report.notes)
