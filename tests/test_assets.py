import base64
import struct
import pytest
from vital2serum.assets import wavetable,sample
from vital2serum.errors import ConversionError
from vital2serum.report import Report
from conftest import wave


def chunks(data):
    assert data[:4]==b'RIFF' and data[8:12]==b'WAVE'
    assert struct.unpack_from('<I',data,4)[0]==len(data)-8
    out={};i=12
    while i<len(data):
        tag=data[i:i+4];n=struct.unpack_from('<I',data,i+4)[0]
        out[tag]=data[i+8:i+8+n];i+=8+n+(n&1)
    return out


def test_single_wave_preserves_floats():
    values=[i/2048 for i in range(2048)]
    a=wavetable(wave(values),Report('test'),'osc_1')
    c=chunks(a.data)
    assert a.frames==2048
    assert c[b'data']==struct.pack('<2048f',*values)
    assert c[b'clm '].startswith(b'<!>2048')
    assert a.serum_reference.startswith('vital2serum/')
    assert struct.unpack('<HHIIHH',c[b'fmt '])[:2]==(3,1)


def test_wave_linear_interpolation():
    table=wave([.25]*2048)
    component=table['groups'][0]['components'][0]
    component['keyframes'].append({'position':255,'wave_data':base64.b64encode(struct.pack('<2048f',*([.75]*2048))).decode()})
    a=wavetable(table,Report('x'),'osc')
    raw=chunks(a.data)[b'data']
    assert a.frames==256*2048
    # Serum frame k holds Vital position k (k < 255) and frame 255 holds position 256.
    assert struct.unpack_from('<f',raw,128*2048*4)[0]==pytest.approx(.25+.5*128/255,abs=1e-6)
    assert struct.unpack_from('<f',raw,255*2048*4)[0]==.75
    assert len(a.pcm[0])==256*2048


def test_unsupported_wave_requires_explicit_placeholder():
    table=wave();table['groups'][0]['components'].append({'type':'Frequency Filter'})
    r=Report('x')
    with pytest.raises(ConversionError): wavetable(table,r,'osc')
    result=wavetable(table,r,'osc',True)
    assert result.frames==2048 and r.lossy

@pytest.mark.parametrize('change',[{'interpolation':2},{'interpolation_style':7},{'keyframes':[]},
                                   {'keyframes':[{'position':257,'wave_data':wave()['groups'][0]['components'][0]['keyframes'][0]['wave_data']}]},
                                   {'keyframes':[{'position':0,'wave_data':'bad!'}]}])
def test_bad_wave(change):
    table=wave();table['groups'][0]['components'][0].update(change)
    with pytest.raises(ConversionError): wavetable(table,Report('x'),'osc')


def test_stereo_sample_is_bit_exact():
    left=struct.pack('<3h',-32768,0,32767);right=struct.pack('<3h',100,-100,42)
    a=sample({'length':3,'sample_rate':48000,'samples':base64.b64encode(left).decode(),
              'samples_stereo':base64.b64encode(right).decode()})
    assert a.channels==2 and a.frames==3
    assert chunks(a.data)[b'data']==struct.pack('<6h',-32768,100,0,-100,32767,42)


def test_bad_sample():
    with pytest.raises(ConversionError): sample({'length':4,'sample_rate':44100,'samples':'AA=='})
