import copy
import pytest

from vital2serum.mapping import convert
from vital2serum.report import Report
from vital2serum.vital import VitalPreset


def test_actionable_notes_are_selected_without_message_guessing():
    report=Report('synthetic')
    report.note('approximation','envelopes','Times use a measured conversion.')
    report.note('approximation','plain','A folded response is mentioned without an impact classification.')
    report.note('info','template','The remaining defaults are retained.')
    report.note('unsupported','env_2_delay','Delay is omitted.')
    report.note('clipped','osc_1_level','Gain is limited.',original=2.0,converted=1.0)
    assert [n['category'] for n in report.user_notices()]==['unsupported','range_limited']
    assert report.user_notices()[1]['original']==2.0


@pytest.mark.parametrize('impact',['control_fixed','control_partial','baked','alternative','unsupported'])
def test_explicit_impact_is_independent_of_conversion_kind(impact):
    report=Report('synthetic')
    report.note('approximation','osc_1_warp','A response is retained.',user_impact=impact)
    before=copy.deepcopy(report.notes)
    notice=report.user_notices()[0]
    assert notice['category']==impact
    assert notice['kind']==('unsupported' if impact=='unsupported' else 'approximation')
    assert report.notes==before and report.notes[0]['kind']=='approximation'
    assert report.lossy==(impact=='unsupported')
    assert report.as_dict()['counts']['unsupported']==(impact=='unsupported')


def test_notice_infers_the_original_route_and_copies_nested_details():
    report=Report('synthetic')
    report.modulations=[dict(vital_slot=2,source='macro_control_1',destination='osc_1_unison_voices',status='mapped')]
    report.note('approximation','modulation_2_control_bus','Stored value only.',
                user_impact='control_fixed',controls=['osc_1_unison_voices'])
    before=copy.deepcopy(report.as_dict())
    notice=report.user_notices()[0]
    assert notice['source']=='macro_control_1'
    assert notice['destination']=='osc_1_unison_voices'
    notice['controls'].append('other')
    assert report.as_dict()==before


def test_explicit_aux_source_is_not_replaced_by_the_primary_source():
    report=Report('synthetic')
    report.modulations=[dict(vital_slot=0,source='env_2',destination='osc_1_pan')]
    report.note('unsupported','modulation_0_','Auxiliary source omitted.',sources=['random_1'])
    notice=report.user_notices()[0]
    assert notice['sources']==['random_1'] and 'source' not in notice
    assert notice['destination']=='osc_1_pan'


def test_ambiguous_destination_is_not_guessed():
    report=Report('synthetic')
    report.modulations=[dict(vital_slot=1,source='env_2',destination='osc_1_pan'),
                        dict(vital_slot=1,source='env_2',destination='osc_2_pan')]
    report.note('unsupported','modulation_1_','Routing omitted.')
    assert 'destination' not in report.user_notices()[0]


def test_owned_output_note_shape_is_not_augmented():
    report=Report('synthetic')
    report.note('approximation','master_volume','Flat output stage.',gain=2.0,output_stage=0)
    before=copy.deepcopy(report.notes)
    assert report.user_notices()==[]
    assert report.notes==before
    assert set(report.notes[0])=={'kind','feature','message','gain','output_stage'}


def test_static_macro_folding_is_visible_without_adding_a_live_route(template,doc):
    doc['settings'].update(macro_control_1=.5,modulation_1_amount=.25,
                           modulations=[{'source':'macro_control_1','destination':'osc_1_unison_voices'}])
    result=convert(VitalPreset(doc),template)
    notice=next(n for n in result.report.user_notices() if n['feature']=='modulation_1_')
    assert notice['category']=='control_fixed'
    assert notice['source']=='macro_control_1' and notice['destination']=='osc_1_unison_voices'
    assert not any(record.get('vital_slot')==1 for record in result.report.modulations)


def test_unapplied_phase_disperse_is_reported_without_changing_the_fallback(template,doc):
    doc['settings'].update(osc_1_spectral_morph_type=9,osc_1_spectral_morph_amount=.7,
                           osc_1_spectral_morph_phase=0.0)
    result=convert(VitalPreset(copy.deepcopy(doc)),template)
    off=copy.deepcopy(doc);off['settings']['osc_1_spectral_morph_type']=0
    reference=convert(VitalPreset(off),template)
    assert result.preset.state==reference.preset.state
    notice=next(n for n in result.report.user_notices() if n['feature']=='osc_1_spectral_morph')
    assert notice['category']=='unsupported' and 'not applied' in notice['message']


def test_baked_warp_identifies_the_replaced_control(template,doc):
    doc['settings'].update(osc_1_distortion_type=1,osc_1_distortion_amount=.3)
    result=convert(VitalPreset(doc),template)
    notice=next(n for n in result.report.user_notices() if n['feature']=='osc_1_warp')
    assert notice['category']=='baked' and notice['controls']==['osc_1_distortion_amount']


def test_strict_blocks_what_the_notices_call_unsupported(template,doc,tmp_path):
    import json
    from vital2serum.cli import main
    doc['settings'].update(osc_1_spectral_morph_type=9,osc_1_spectral_morph_amount=.7,osc_1_spectral_morph_phase=0.0)
    report=convert(VitalPreset(copy.deepcopy(doc)),template).report
    assert [n['category'] for n in report.user_notices() if n['feature']=='osc_1_spectral_morph']==['unsupported']
    assert report.lossy and report.as_dict()['counts']['unsupported']>=1
    source=tmp_path/'source.vital';source.write_text(json.dumps(doc))
    out=tmp_path/'out'
    assert main(['convert',str(source),'-o',str(out),'--strict'])==2
    assert (out/'source.report.json').is_file() and not (out/'source.SerumPreset').exists()
    assert main(['convert',str(source),'-o',str(out),'--overwrite'])==0
    assert (out/'source.SerumPreset').is_file()

