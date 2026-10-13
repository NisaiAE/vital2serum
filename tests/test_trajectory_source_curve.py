import copy
import math

import numpy as np
import pytest

from vital2serum import modulation
from vital2serum.mapping import convert
from vital2serum.serum import SerumPreset
from vital2serum.vital import VitalPreset


def raw(source='env_1', **options):
    values=dict(slot=1,source=source,target='osc_1_wave_frame',amount=1.,bipolar=False,power=0.,stereo=False)
    values.update(options)
    return modulation.Modulation(**values)


def joint(doc, source='env_1', osc=1):
    s=doc['settings'];p=f'osc_{osc}_'
    s.update({p+'on':1,p+'distortion_type':4,p+'distortion_amount':.2,p+'wave_frame':0})
    for target,amount in [('wave_frame',.75),('distortion_amount',.6)]:
        s['modulations'].append(dict(source=source,destination=p+target))
        s[f'modulation_{len(s["modulations"])}_amount']=amount


def test_raw_projection_bounds_every_serialized_segment(template):
    curve=modulation._trajectory_source_curve(raw())
    template.state['ModSlot0']['flex']=[curve]
    loaded=SerumPreset.from_bytes(template.to_bytes()).state['ModSlot0']['flex'][0]
    assert loaded==curve
    assert len(curve['xVals'])<=480 and curve['numPoints']==len(curve['xVals'])-1
    assert curve['xVals'][0]==0 and curve['xVals'][-1]==1
    assert curve['yVals'][0]==1 and curve['yVals'][-1]==0
    assert all(a<b for a,b in zip(curve['xVals'],curve['xVals'][1:]))
    for a,b,ya,yb in zip(curve['xVals'],curve['xVals'][1:],curve['yVals'],curve['yVals'][1:]):
        # The derivative of sqrt(x) equals this secant slope at this point.
        x=((math.sqrt(a)+math.sqrt(b))/2)**2
        represented=1-ya+(ya-yb)*(x-a)/(b-a)
        assert 0<=math.sqrt(x)-represented<=2e-5+1e-15
    x=np.linspace(0,1,100001)
    error=np.sqrt(x)-np.interp(x,curve['xVals'],1-np.array(curve['yVals']))
    assert np.max(np.abs(error))<=2e-5
    assert np.interp(1/4096,curve['xVals'],1-np.array(curve['yVals']))==pytest.approx(1/64)


@pytest.mark.parametrize('source',['env_2','lfo_1','macro_control_1'])
@pytest.mark.parametrize('options',[{},dict(power=2,bipolar=True),dict(points=((0,1),(.5,0),(1,1)),powers=(1,-1))])
def test_sources_without_law_keep_existing_curves(source,options):
    m=raw(source,**options)
    assert modulation._trajectory_source_curve(m)==modulation._source_curve(m)


@pytest.mark.parametrize('options',[dict(power=1),dict(bipolar=True),dict(points=((0,1),(1,0)))])
def test_projection_requires_a_raw_source(options):
    with pytest.raises(ValueError):modulation._trajectory_source_curve(raw(**options))


def test_unverified_future_source_law_is_not_treated_as_square_root(monkeypatch):
    monkeypatch.setitem(modulation.SOURCE_LAWS,'env_2',lambda x:x)
    with pytest.raises(ValueError):modulation._trajectory_source_curve(raw('env_2'))


@pytest.mark.parametrize('policy',[dict(tolerance=0),dict(tolerance=float('nan')),dict(limit=481),dict(limit=32),dict(limit=100.5),dict(limit=33)])
def test_invalid_or_insufficient_budget_fails_before_return(policy):
    with pytest.raises(ValueError):modulation._trajectory_source_curve(raw(),**policy)


@pytest.mark.parametrize('source',['env_2','lfo_1','macro_control_1'])
def test_unaffected_joint_sources_keep_complete_payload(template,doc,monkeypatch,source):
    joint(doc,source)
    before=convert(VitalPreset(copy.deepcopy(doc)),template)
    monkeypatch.setattr(modulation,'_trajectory_source_curve',modulation._source_curve)
    after=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert before.preset.to_bytes()==after.preset.to_bytes()
    assert before.report.modulations==after.report.modulations
    assert before.report.notes==after.report.notes


@pytest.mark.parametrize('target',['osc_1_pan','osc_1_distortion_amount'])
def test_ordinary_and_single_axis_law_routes_keep_complete_payload(template,doc,monkeypatch,target):
    s=doc['settings'];s.update(osc_1_distortion_type=4,modulation_1_amount=.6,
                             modulations=[dict(source='env_1',destination=target)])
    before=convert(VitalPreset(copy.deepcopy(doc)),template)
    monkeypatch.setattr(modulation,'_trajectory_source_curve',lambda m:pytest.fail('No joint trajectory'))
    after=convert(VitalPreset(copy.deepcopy(doc)),template)
    assert before.preset.to_bytes()==after.preset.to_bytes()
    assert before.report.notes==after.report.notes


@pytest.mark.parametrize('oscillators',[1,2,3])
def test_budget_failure_restores_whole_conversion_and_other_groups(template,doc,monkeypatch,oscillators):
    joint(doc)
    s=doc['settings'];s['wavetables']=[copy.deepcopy(s['wavetables'][0]) for _ in range(3)]
    for osc in range(2,oscillators+1):joint(doc,'env_2' if osc==2 else 'lfo_1',osc)
    helper=modulation._trajectory_source_curve
    monkeypatch.setattr(modulation,'_trajectory_source_curve',lambda m:helper(m,limit=33))
    v=VitalPreset(copy.deepcopy(doc));first=convert(v,template);again=convert(v,template)
    ref=convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_trajectories=frozenset({'osc_1_'}))
    assert first.preset.to_bytes()==again.preset.to_bytes()==ref.preset.to_bytes()
    assert [(a.data,a.gain) for a in first.assets]==[(a.data,a.gain) for a in ref.assets]
    assert first.report.modulations==again.report.modulations==ref.report.modulations
    assert first.report.notes==again.report.notes
    assert [n for n in first.report.notes if 'original conversion is retained' not in n['message']]==ref.report.notes
    assert {m['joint_trajectory'] for m in first.report.modulations if m.get('joint_trajectory')}=={
        f'osc_{osc}_' for osc in range(2,oscillators+1)}


def test_only_joint_source_curve_changes(template,doc,monkeypatch):
    joint(doc)
    improved=convert(VitalPreset(copy.deepcopy(doc)),template)
    monkeypatch.setattr(modulation,'_trajectory_source_curve',modulation._source_curve)
    old=convert(VitalPreset(copy.deepcopy(doc)),template)
    a,b=copy.deepcopy(improved.preset.state),copy.deepcopy(old.preset.state)
    keys=[k for k,v in a.items() if k.startswith('ModSlot') and v.get('destModuleParamName')]
    assert len(keys)==1
    key=keys[0]
    assert len(a[key]['flex'][0]['xVals'])>len(b[key]['flex'][0]['xVals'])
    a[key]['flex'][0]=b[key]['flex'][0]=None
    assert a==b
    assert [(x.data,x.gain) for x in improved.assets]==[(x.data,x.gain) for x in old.assets]
    assert improved.report.modulations==old.report.modulations
