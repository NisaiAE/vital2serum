import math

import pytest

from vital2serum.mapping import Context, _global, convert
from vital2serum.report import Report
from vital2serum.serum import clean_template, plain
from vital2serum.vital import VitalPreset


@pytest.mark.parametrize('volume',[1000.0,4000.0,5473.0404])
def test_master_volume_modulation_has_exact_mute_endpoint(template,doc,volume):
    doc['settings'].update(volume=volume,velocity_track=0.0)
    ctx=Context(VitalPreset(doc),clean_template(template),Report('<memory>'))
    _global(ctx)
    law=ctx.destinations['volume'].law
    assert law(-1.0)==law(0.0)==0.0
    assert law(volume)==pytest.approx(0.5)
    assert law(volume/2)==pytest.approx(0.5*10**((math.sqrt(volume/2)-math.sqrt(volume))/20))


def test_full_master_volume_cut_writes_exact_voice_mute(template,doc):
    doc['settings'].update(volume=4000.0,velocity_track=0.0,macro_control_1=1.0,
        modulation_1_amount=-1.0,
        modulations=[{'source':'macro_control_1','destination':'volume'}])
    result=convert(VitalPreset(doc),template)
    route=next(m for m in result.report.modulations if m.get('destination')=='volume')
    assert route['serum_parameter']=='kParamVoiceAmp'
    assert route['serum_amount']==pytest.approx(-50.0)


@pytest.mark.parametrize('tracking',[-1.0,0.5,1.0])
def test_master_volume_endpoint_does_not_replace_velocity_tracking(template,doc,tracking):
    doc['settings'].update(velocity_track=tracking)
    ctx=Context(VitalPreset(doc),clean_template(template),Report('<memory>'))
    _global(ctx)
    assert 'volume' not in ctx.destinations
    assert plain(ctx.s.state['Global0'])['kParamVoiceAmp']==0.0
    assert ctx.fixed_routings and all(route[0]=='velocity' for route in ctx.fixed_routings)
