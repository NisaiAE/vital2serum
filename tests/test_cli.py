import json
from pathlib import Path
import pytest
from vital2serum.cli import main,safe_name,secure_path
from vital2serum.errors import ConversionError
from vital2serum.serum import SerumPreset
from vital2serum.vital import VitalPreset


def test_cli_full_conversion(doc,tmp_path,capsys):
    source=tmp_path/'source.vital';source.write_text(json.dumps(doc))
    out=tmp_path/'out'
    assert main(['convert',str(source),'-o',str(out)])==0
    path=out/'source.SerumPreset'
    assert path.is_file()
    p=SerumPreset.read(path)
    wt=p.state['Oscillator0']['WTOsc0']
    assert 'relativePathToWT' not in wt and len(wt['embeddedWTData'])==wt['numFrames']==2048
    assert not (out/'Tables').exists()
    wav_out=tmp_path/'wav'
    assert main(['convert',str(source),'-o',str(wav_out),'--wav-tables'])==0
    ref=SerumPreset.read(wav_out/'source.SerumPreset').state['Oscillator0']['WTOsc0']['relativePathToWT']
    assert (wav_out/'Tables'/ref).is_file()
    report=json.loads((out/'source.report.json').read_text())
    assert report['converter']=='vital2serum' and set(report['counts'])=={'approximation','unsupported','clipped','info'}
    assert main(['inspect',str(path)])==0
    assert main(['convert',str(source),'-o',str(out)])==2
    assert main(['convert',str(source),'-o',str(out),'--overwrite'])==0


def test_dry_run_no_files(doc,tmp_path,capsys):
    source=tmp_path/'test.vital';source.write_text(json.dumps(doc))
    before=set(tmp_path.iterdir())
    assert main(['convert',str(source),'--dry-run'])==0
    assert set(tmp_path.iterdir())==before
    assert json.loads(capsys.readouterr().out)['target']=='Serum 2'


def test_strict_reports_only(doc,tmp_path,capsys):
    doc['settings']['osc_1_frame_spread']=1
    source=tmp_path/'test.vital';source.write_text(json.dumps(doc));out=tmp_path/'out'
    assert main(['convert',str(source),'-o',str(out),'--strict'])==2
    assert (out/'test.report.json').is_file()
    assert not list(out.rglob('*.SerumPreset'))
    assert not (out/'Tables').exists()


def test_recursive_paths(doc,tmp_path,capsys):
    folder=tmp_path/'in';(folder/'sub').mkdir(parents=True)
    (folder/'sub'/'a.vital').write_text(json.dumps(doc));out=tmp_path/'out'
    assert main(['convert',str(folder),'-o',str(out),'--recursive'])==0
    assert (out/'sub'/'a.SerumPreset').exists()


def test_batch_collision_no_writes(doc,tmp_path,capsys):
    folder=tmp_path/'in';folder.mkdir()
    for name in ('A.vital','a.vital'): (folder/name).write_text(json.dumps(doc))
    if len(list(folder.iterdir()))<2:
        pytest.skip('case-insensitive file system: both names are the same file')
    out=tmp_path/'out'
    assert main(['convert',str(folder),'-o',str(out)])==2
    assert not out.exists()


def test_asset_collision_not_overwritten(doc,tmp_path,capsys):
    source=tmp_path/'test.vital';source.write_text(json.dumps(doc));out=tmp_path/'out'
    assert main(['convert',str(source),'-o',str(out),'--wav-tables'])==0
    asset=next((out/'Tables').rglob('*.wav'));asset.write_bytes(b'conflicting file')
    assert main(['convert',str(source),'-o',str(out),'--overwrite','--wav-tables'])==2
    assert asset.read_bytes()==b'conflicting file'


def test_bad_input_no_traceback(tmp_path,capsys):
    source=tmp_path/'bad.vital';source.write_text('{"settings":{"volume":NaN}}')
    assert main(['convert',str(source),'--dry-run'])==2
    assert 'Traceback' not in capsys.readouterr().err


def test_path_security(tmp_path):
    assert safe_name('../../evil')=='.._.._evil'
    assert safe_name('CON')=='_CON'
    with pytest.raises(ConversionError): secure_path(tmp_path,Path('../escape'))
    outside=tmp_path.parent/'outside-test-v2s';outside.mkdir(exist_ok=True)
    (tmp_path/'link').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ConversionError): secure_path(tmp_path,Path('link/a'))

@pytest.mark.parametrize('version',['0.9.1','2.0.0','x.y'])
def test_version_guard(doc,version):
    doc['synth_version']=version
    with pytest.raises(ConversionError): VitalPreset(doc)

@pytest.mark.parametrize('version',['1.0.0','1.0.7','1.5.5'])
def test_vital_1x_accepted(doc,version):
    doc['synth_version']=version
    assert VitalPreset(doc).doc['synth_version']==version
