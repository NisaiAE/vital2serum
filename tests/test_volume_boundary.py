import copy
import math
import struct
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from conftest import wave
from vital2serum import mapping,post_master,modulation,voice_gain
from vital2serum.vital import VitalPreset
from vital2serum.ranges import RANGES

LO,HI=RANGES['volume']
ALIAS=struct.unpack('<f',struct.pack('<f',HI))[0]
BITS=struct.unpack('<I',struct.pack('<f',ALIAS))[0]
NEXT=struct.unpack('<f',struct.pack('<I',BITS+1))[0]
PREVIOUS=struct.unpack('<f',struct.pack('<I',BITS-1))[0]

def old_domain(v):
    raw=v.settings.get('volume',post_master.DEFAULTS['volume'])+v.offsets.get('volume',0.)
    return isinstance(raw,(int,float)) and math.isfinite(raw) and LO<=raw<=HI

@pytest.mark.parametrize('stored,offset,accepted',[
    (LO,0,True),(HI,0,True),(PREVIOUS,0,True),(ALIAS,0,True),
    (math.nextafter(HI,math.inf),0,False),((HI+ALIAS)/2,0,False),
    (math.nextafter(ALIAS,-math.inf),0,False),(math.nextafter(ALIAS,math.inf),0,False),
    (NEXT,0,False),(ALIAS,1e-10,False),(HI,ALIAS-HI,False),
    (ALIAS,-1,True),(ALIAS,-1e-10,False),(HI,1e-10,False),
    (-1e-10,0,False),(math.nan,0,False),(math.inf,0,False),(-math.inf,0,False),
    (True,0,True),(False,0,True),(9000,-2000,True),
])
def test_only_exact_stored_alias_adds_an_outside_value(stored,offset,accepted):
    v=SimpleNamespace(settings={'volume':stored},offsets={'volume':offset})
    assert post_master.bounded_volume(v)==accepted
    if stored!=ALIAS or offset!=0:assert post_master.bounded_volume(v)==old_domain(v)

@pytest.mark.parametrize('stored',[None,'7399.4404',[],{}])
def test_unknown_storage_retains_the_previous_exception(stored):
    v=SimpleNamespace(settings={'volume':stored},offsets={})
    with pytest.raises(TypeError):old_domain(v)
    with pytest.raises(TypeError):post_master.bounded_volume(v)

@pytest.mark.parametrize('missing',[True,False])
def test_default_and_bool_storage_semantics_are_unchanged(missing):
    v=SimpleNamespace(settings={} if missing else {'volume':True},offsets={})
    assert post_master.bounded_volume(v)==old_domain(v)

@pytest.mark.parametrize('moving',[False,True])
@pytest.mark.parametrize('lowered',[False,True])
@pytest.mark.parametrize('embed',[False,True])
def test_literal_upper_endpoint_and_its_alias_have_identical_full_results(template,doc,moving,lowered,embed):
    doc['settings'].update(volume=HI,velocity_track=0,osc_1_level=1)
    if lowered:doc['settings']['wavetables'][0]=wave([1.5*math.sin(2*math.pi*i/2048) for i in range(2048)])
    if moving:
        doc['settings'].update(modulations=[dict(source='macro_control_1',destination='volume')],
                               modulation_1_amount=-.1,macro_control_1=.5)
    literal=mapping.convert(VitalPreset(copy.deepcopy(doc)),template,embed=embed)
    alias_doc=copy.deepcopy(doc);alias_doc['settings']['volume']=ALIAS
    reused=VitalPreset(copy.deepcopy(alias_doc));alias=mapping.convert(reused,template,embed=embed)
    assert alias.preset.to_bytes()==literal.preset.to_bytes()
    assert alias.preset.state==literal.preset.state and alias.report.as_dict()==literal.report.as_dict()
    assert alias.assets==literal.assets and alias.report.assets==literal.report.assets
    second=mapping.convert(reused,template,embed=embed)
    assert second.preset.to_bytes()==alias.preset.to_bytes() and second.report.as_dict()==alias.report.as_dict()
    assert reused.doc==alias_doc

@pytest.mark.parametrize('stored',[NEXT,math.nextafter(HI,math.inf),(HI+ALIAS)/2,9000,-1])
def test_other_outside_storage_retains_exact_previous_conversion(template,doc,stored):
    doc['settings'].update(volume=stored,velocity_track=0)
    result=mapping.convert(VitalPreset(copy.deepcopy(doc)),template)
    with patch.object(post_master,'bounded_volume',side_effect=old_domain):
        previous=mapping.convert(VitalPreset(copy.deepcopy(doc)),template)
    assert result.preset.to_bytes()==previous.preset.to_bytes()
    assert result.report.as_dict()==previous.report.as_dict() and result.assets==previous.assets

@pytest.mark.parametrize('failure',['direct','rack','slot','curve','id','source_gain','moving_aux'])
def test_alias_does_not_bypass_owned_response_or_pure_fallback(template,doc,monkeypatch,failure):
    doc['settings'].update(volume=ALIAS,velocity_track=0,macro_control_1=.5,
        modulations=[dict(source='macro_control_1',destination='volume')],modulation_1_amount=-.1)
    options={}
    if failure=='direct':doc['settings']['osc_1_destination']=4
    elif failure=='rack':
        original=mapping.convert_effects
        def full(c):
            original(c)
            c.s.state['FXRack0']['FX']=[{'FXEQ':{'plainParams':{'kParamEnable':1.,'kParamLevelOut':.5}},'type':7,'kUIParamMixOrGain':0.} for _ in range(10)]
        monkeypatch.setattr(mapping,'convert_effects',full)
    elif failure=='slot':monkeypatch.setattr(modulation,'MAX_SLOTS',0)
    elif failure=='curve':
        original=modulation._adaptive_law_points
        def reject(fn,m,policy):
            if m.target=='volume':raise ValueError('budget')
            return original(fn,m,policy)
        monkeypatch.setattr(modulation,'_adaptive_law_points',reject)
    elif failure=='id':
        ids=dict(modulation.KNOWN_IDS);del ids[('FXEQ','kParamLevelOut')];options['destination_ids']=ids
    elif failure=='source_gain':
        original=post_master.register
        def unsupported(c,ids,disabled=False):
            c.source_gain['macro_control_1']=.5
            return original(c,ids,disabled)
        monkeypatch.setattr(post_master,'register',unsupported)
    else:
        doc['settings']['modulations'].append(dict(source='env_2',destination='modulation_1_amount'))
        doc['settings']['modulation_2_amount']=.1
    reused=VitalPreset(copy.deepcopy(doc));result=mapping.convert(reused,template,**options)
    previous=mapping.convert(VitalPreset(copy.deepcopy(doc)),template,_disabled_post_master=True,**options)
    assert result.preset.to_bytes()==previous.preset.to_bytes()
    assert result.report.modulations==previous.report.modulations
    assert result.assets==previous.assets and reused.doc==doc
    second=mapping.convert(reused,template,**options)
    assert second.preset.to_bytes()==result.preset.to_bytes() and second.report.as_dict()==result.report.as_dict()
    assert not any(slot.get('destModuleTypeString')=='FXEQ' and slot.get('destModuleParamName')=='kParamLevelOut'
                   for key,slot in result.preset.state.items() if key.startswith('ModSlot'))
