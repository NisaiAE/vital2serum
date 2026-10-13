import copy
import math
import pytest
from vital2serum.effects import filter_dry_gain, filter_mix_controls, interp
from vital2serum.mapping import convert
from vital2serum.modulation import Destination, Modulation, _law_slots
from vital2serum.serum import plain
from vital2serum.vital import VitalPreset


def configure(doc,control='mix',**settings):
    doc['settings'].update(filter_1_on=1,filter_1_model=0,filter_1_style=0,
                           filter_1_cutoff=84,filter_1_resonance=0,filter_1_drive=0,
                           filter_1_blend=0,filter_1_mix=1 if control=='resonance' else 0,
                           osc_1_level=.70710678119)
    doc['settings']['modulations']=[dict(source='macro_control_1',destination='filter_1_'+control)]
    doc['settings']['modulation_1_amount']=1
    doc['settings'].update(settings)
    return doc


def source_value(curve,value):
    return 1-interp(value,list(zip(curve['xVals'],curve['yVals'])))


def effective(result,parameter,value):
    scale=100 if parameter in ('kParamWet','kParamReso','kParamDrive') else 1
    base=plain(result.preset.state['VoiceFilter0'])[parameter]/scale
    for key,module in result.preset.state.items():
        if not key.startswith('ModSlot') or module.get('destModuleParamName')!=parameter:continue
        if module.get('destModuleTypeString')!='VoiceFilter' or module.get('destModuleID')!=0:continue
        curve=module.get('flex',[None])[0]
        base+=plain(module)['kParamAmount']/100*(source_value(curve,value) if curve else value)
    return base


@pytest.mark.parametrize('control,mix',[('mix',0),('resonance',1),('resonance',.5)])
def test_joint_controls_reproduce_static_endpoints(template,doc,control,mix):
    configure(doc,control,filter_1_mix=mix)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    for source in (0,.25,.5,.75,1):
        fixed=copy.deepcopy(doc);fixed['settings']['modulations']=[]
        fixed['settings']['filter_1_'+control]=source
        expected=plain(convert(VitalPreset(fixed),template).preset.state['VoiceFilter0'])
        for parameter in ('kParamWet','kParamReso','kParamLevelOut'):
            if parameter=='kParamReso' and control=='mix':continue
            if parameter=='kParamLevelOut' and control=='mix' and source==0:
                # A fixed dry control skips drive; a moving one cancels its small-signal gain.
                params=plain(result.preset.state['VoiceFilter0'])
                dry_gain=filter_dry_gain(params['kParamType'],params['kParamDrive'])
                assert (2*effective(result,parameter,source))**2*dry_gain==pytest.approx(1)
                continue
            scale=100 if parameter in ('kParamWet','kParamReso') else 1
            assert effective(result,parameter,source)==pytest.approx(expected[parameter]/scale,abs=1e-10)
    assert result.report.modulations[0]['supplementary_destinations']


def test_bipolar_resonance_makeup_tracks_both_sides(template,doc):
    configure(doc,'resonance',filter_1_resonance=.5,filter_1_mix=.5,modulation_1_bipolar=1)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    for source in (0,.25,.5,.75,1):
        fixed=copy.deepcopy(doc);fixed['settings'].update(modulations=[],filter_1_resonance=source)
        expected=plain(convert(VitalPreset(fixed),template).preset.state['VoiceFilter0'])
        assert effective(result,'kParamLevelOut',source)==pytest.approx(expected['kParamLevelOut'],abs=1e-10)
        assert effective(result,'kParamWet',source)==pytest.approx(expected['kParamWet']/100,abs=1e-10)


@pytest.mark.parametrize('model,style,blend',[(0,0,0),(0,1,0),(1,0,0),(2,0,0),(3,1,0),(4,0,0)])
def test_full_wet_drive_tracks_static_saturation_and_output(template,doc,model,style,blend):
    configure(doc,'drive',filter_1_model=model,filter_1_style=style,filter_1_blend=blend,
              filter_1_mix=1,filter_1_drive=0,filter_1_resonance=.5)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    for index in range(33):
        source=index/32
        fixed=copy.deepcopy(doc);fixed['settings'].update(modulations=[],filter_1_drive=20*source)
        expected=plain(convert(VitalPreset(fixed),template).preset.state['VoiceFilter0'])
        assert effective(result,'kParamDrive',source)==pytest.approx(expected['kParamDrive']/100,abs=1e-10)
        assert effective(result,'kParamLevelOut',source)==pytest.approx(expected['kParamLevelOut'],abs=1e-10)
    assert result.report.modulations[0]['supplementary_destinations']


@pytest.mark.parametrize('extra',[
    {'filter_1_mix':.5}, {'modulation_1_stereo':1},
])
def test_drive_joint_excludes_partial_mix_and_stereo(template,doc,extra):
    configure(doc,'drive',**dict({'filter_1_mix':1,'filter_1_drive':0},**extra))
    result=convert(VitalPreset(doc),template)
    assert not result.report.modulations[0].get('supplementary_destinations')
    assert not any(m.get('destModuleParamName')=='kParamLevelOut' for k,m in result.preset.state.items() if k.startswith('ModSlot'))


@pytest.mark.parametrize('bipolar',[0,1])
def test_negative_drive_joint_matches_static_values_at_clamped_edges(template,doc,bipolar):
    configure(doc,'drive',filter_1_mix=1,filter_1_drive=12.4,
              modulation_1_amount=-1,modulation_1_bipolar=bipolar)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    for index in range(33):
        source=index/32;raw=12.4-20*(source-.5 if bipolar else source)
        fixed=copy.deepcopy(doc);fixed['settings'].update(modulations=[],filter_1_drive=min(20,max(0,raw)))
        expected=plain(convert(VitalPreset(fixed),template).preset.state['VoiceFilter0'])
        assert effective(result,'kParamDrive',source)==pytest.approx(expected['kParamDrive']/100,abs=1e-10)
        assert effective(result,'kParamLevelOut',source)==pytest.approx(expected['kParamLevelOut'],abs=1e-10)


@pytest.mark.parametrize('other',[('env_2','filter_1_resonance'),('env_2','filter_1_mix'),('env_2','filter_1_drive')])
def test_multiple_moving_controls_or_sources_keep_fixed_makeup(template,doc,other):
    configure(doc)
    doc['settings']['modulations'].append(dict(source=other[0],destination=other[1]))
    doc['settings']['modulation_2_amount']=.2
    result=convert(VitalPreset(doc),template)
    assert not any(m.get('destModuleParamName')=='kParamLevelOut' for k,m in result.preset.state.items() if k.startswith('ModSlot'))
    assert any(n['feature']=='filter_1_makeup' for n in result.report.notes)


def test_amount_modulation_keeps_fixed_makeup(template,doc):
    configure(doc)
    doc['settings']['modulations'].append(dict(source='env_2',destination='modulation_1_amount'))
    doc['settings']['modulation_2_amount']=.2
    result=convert(VitalPreset(doc),template)
    assert not any(m.get('destModuleParamName')=='kParamLevelOut' for k,m in result.preset.state.items() if k.startswith('ModSlot'))
    assert any(n['feature']=='filter_1_makeup' for n in result.report.notes)


@pytest.mark.parametrize('filler,expected',[(61,2),(62,0),(63,0)])
def test_supplementary_routes_fit_as_a_complete_group(template,doc,filler,expected):
    configure(doc,'resonance',filter_1_mix=.5,modulation_1_amount=.5)
    for index in range(2,filler+2):
        doc['settings']['modulations'].append(dict(source='env_2',destination='osc_1_pan'))
        doc['settings'][f'modulation_{index}_amount']=.01
    result=convert(VitalPreset(doc),template)
    extras=result.report.modulations[0].get('supplementary_destinations',[])
    assert len(extras)==expected
    assert sum(m['status']=='mapped' for m in result.report.modulations)==filler+1
    if not expected:assert any(n['feature']=='modulation_1_makeup' for n in result.report.notes)


@pytest.mark.parametrize('law,expected_slots',[
    (lambda x: .25+2*x*(1-x),1),
    (lambda x: .5+x*(1-x)*(x-.5),2),
])
def test_nonmonotonic_law_retains_interior_motion(doc,law,expected_slots):
    doc['settings']['filter_1_mix']=0
    vital=VitalPreset(doc)
    modulation=Modulation(1,'env_2','filter_1_mix',1,False,0,False)
    destination=Destination('VoiceFilter',0,'kParamLevelOut',law=law)
    pieces=_law_slots(vital,modulation,destination,1)
    assert len(pieces)==expected_slots
    for source in (0,.25,.5,.75,1):
        actual=law(0)+sum(amount/100*source_value(curve,source) for amount,curve in pieces)
        assert actual==pytest.approx(law(source))


@pytest.mark.parametrize('coefficient',[.5,-.5])
def test_remapped_source_uses_only_one_curve_per_sign(doc,coefficient):
    doc['settings']['filter_1_mix']=.5
    vital=VitalPreset(doc)
    modulation=Modulation(1,'env_2','filter_1_mix',coefficient,False,0,False,((0,.8),(1,.2)),(0,))
    destination=Destination('VoiceFilter',0,'kParamLevelOut',law=lambda value:value*value)
    pieces=_law_slots(vital,modulation,destination,coefficient)
    assert len(pieces)==1
    for index in range(33):
        source=index/32
        wanted=(.5+coefficient*modulation.value(source))**2
        actual=.25+sum(amount/100*source_value(curve,source) for amount,curve in pieces)
        assert actual==pytest.approx(wanted)


@pytest.mark.parametrize('fx',[False,True])
@pytest.mark.parametrize('gain_db',[-12,0,9])
def test_filter_mix_reconstructs_both_signal_coefficients(fx,gain_db):
    gain=10**(gain_db/20)
    dry_gain=1 if fx else (1+.54)**4
    for mix in (0,.05,.25,.5,.9,1):
        position,level=filter_mix_controls(mix,gain_db,dry_gain,fx=fx)
        wet=math.sin(position*math.pi/2)**2 if fx else position
        dry=1-wet*wet if fx else 1-wet
        amplitude=(2*level)**2
        assert amplitude*dry*dry_gain==pytest.approx(1-mix,abs=1e-12)
        assert amplitude*wet/gain==pytest.approx(mix,abs=1e-12)


@pytest.mark.parametrize('drive,expected',[(0,1),(25,2.44140625),(50,5.0625),(75,9.37890625),(100,16)])
def test_voice_dry_drive_scale_and_ladder_exception(drive,expected):
    for kind in ('LBH12','LBH24','MgL12'):
        assert filter_dry_gain(kind,drive)==pytest.approx(expected)
    assert filter_dry_gain('LadderEMS',drive)==1


def test_fixed_dry_voice_filter_keeps_unity_and_ignores_drive_route(template,doc):
    configure(doc,'resonance',filter_1_mix=0,filter_1_drive=20)
    doc['settings']['modulations']=[dict(source='env_2',destination='filter_1_drive')]
    result=convert(VitalPreset(doc),template)
    params=plain(result.preset.state['VoiceFilter0'])
    assert (params['kParamDrive'],params['kParamWet'],params['kParamLevelOut'])==(0,0,.5)
    assert result.report.modulations[0]['status']=='omitted'


def test_moving_mix_retains_drive_and_reports_dry_saturation(template,doc):
    configure(doc,'mix',filter_1_drive=5)
    result=convert(VitalPreset(doc),template)
    params=plain(result.preset.state['VoiceFilter0'])
    assert params['kParamDrive']>0
    assert params['kParamLevelOut']<.5
    assert any(n['feature']=='filter_1_mix' and 'saturation' in n['message'] for n in result.report.notes)


def test_drive_limit_partial_mix_retains_legacy_input_fit(template,doc):
    configure(doc,'mix',filter_1_model=1,filter_1_style=1,filter_1_mix=.885,filter_1_drive=20)
    doc['settings'].update(modulations=[],osc_2_on=1,osc_2_destination=4,osc_2_level=1)
    doc['settings']['wavetables'][1]=copy.deepcopy(doc['settings']['wavetables'][0])
    result=convert(VitalPreset(doc),template)
    partial=plain(result.preset.state['VoiceFilter0'])
    # The legacy input fit counts the bypassing source, so obtain that reference explicitly.
    reference=copy.deepcopy(doc);reference['settings'].update(filter_1_mix=1,osc_2_destination=0)
    full=plain(convert(VitalPreset(reference),template).preset.state['VoiceFilter0'])
    gain=(2*full['kParamLevelOut'])**2;amplitude=1-.885+.885*gain
    assert partial['kParamDrive']==100
    assert partial['kParamWet']==pytest.approx(100*.885*gain/amplitude)
    assert (2*partial['kParamLevelOut'])**2==pytest.approx(amplitude)
    assert any(n['feature']=='filter_1_mix' and 'previous mix' in n['message'] for n in result.report.notes)


def test_drive_limit_moving_mix_does_not_add_uncalibrated_makeup(template,doc):
    configure(doc,'mix',filter_1_drive=20)
    result=convert(VitalPreset(doc),template)
    assert not any(m.get('destModuleParamName')=='kParamLevelOut' for k,m in result.preset.state.items() if k.startswith('ModSlot'))
    assert any(n['feature']=='filter_1_makeup' for n in result.report.notes)


def test_filter_auxiliary_law_is_not_folded_by_drive_opt_in(template,doc):
    configure(doc,'mix',macro_control_2=.75,modulation_1_amount=0)
    doc['settings']['modulations'].append(dict(source='macro_control_2',destination='modulation_1_amount'))
    doc['settings']['modulation_2_amount']=.5
    result=convert(VitalPreset(doc),template)
    slots=[m for k,m in result.preset.state.items() if k.startswith('ModSlot') and m.get('destModuleParamName')=='kParamWet']
    assert slots and all(m['source'][1]==26 for m in slots)
    assert not any(n['feature']=='modulation_1_static_aux' for n in result.report.notes)


def test_fx_filter_uses_its_rack_mix_weights(template,doc):
    doc['settings'].update(filter_fx_on=1,filter_fx_model=0,filter_fx_resonance=.5,
                           filter_fx_drive=5,filter_fx_mix=1)
    full=convert(VitalPreset(copy.deepcopy(doc)),template)
    params=next(plain(e['FXFilter']) for e in full.preset.state['FXRack0']['FX'] if 'FXFilter' in e)
    gain=(2*params['kParamLevelOut'])**2
    doc['settings']['filter_fx_mix']=.25
    result=convert(VitalPreset(doc),template)
    params=next(plain(e['FXFilter']) for e in result.preset.state['FXRack0']['FX'] if 'FXFilter' in e)
    wet=math.sin(params['kParamWet']*math.pi/200)**2
    amplitude=(2*params['kParamLevelOut'])**2
    assert amplitude*(1-wet*wet)==pytest.approx(.75)
    assert amplitude*wet/gain==pytest.approx(.25)
