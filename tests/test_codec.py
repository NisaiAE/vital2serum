import hashlib
import struct
from importlib.resources import files
import pytest
from vital2serum import codec
from vital2serum.errors import ConversionError
from vital2serum.serum import SerumPreset,parse_json

@pytest.mark.parametrize('value',[None,False,True,0,23,24,255,256,65535,65536,2**32,2**63,-1,-25,-2**40,
                                    .0,1.5,-23.25,'','日本語',b'\x00\xff',[1,'x'],{'foo':[1,2,3]},
                                    codec.Tag(1000,'opaque'),codec.Simple(23)])
def test_cbor_roundtrip(value): assert codec.loads(codec.dumps(value))==value

@pytest.mark.parametrize('raw,expected',[
    (b'\x9f\x01\x02\xff',[1,2]),(b'\xbf\x61a\x01\xff',{'a':1}),
    (b'\x7f\x61a\x61b\xff','ab'),(b'\x5f\x41a\x41b\xff',b'ab'),
    (b'\xf9\x3c\x00',1.0),(b'\xfa\x3f\x80\x00\x00',1.0)])
def test_cbor_forms(raw,expected): assert codec.loads(raw)==expected

@pytest.mark.parametrize('raw',[b'',b'\x1a\x00',b'\x01\x01',b'\xff',b'\x9f\x01',b'\xbf\x61a\xff',
                                 b'\xa2\x61a\x01\x61a\x02',b'\x7f\x41a\xff',b'\xf9\x7e\x00'])
def test_cbor_reject(raw):
    with pytest.raises(ConversionError): codec.loads(raw)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),2**65])
def test_bad_cbor_value(value):
    with pytest.raises(ConversionError): codec.dumps(value)

@pytest.mark.parametrize('payload',[b'x',b'0'*100000,bytes(range(256))])
def test_zstd_roundtrip(payload):
    packed=codec.compress(payload)
    assert codec.decompress(packed,len(payload))==payload
    with pytest.raises(ConversionError): codec.decompress(packed,len(payload)+1)
    with pytest.raises(ConversionError): codec.decompress(packed+b'x',len(payload))


def test_real_upstream_fixture(template):
    raw=files('vital2serum').joinpath('data/Init-2.0.24.SerumPreset').read_bytes()
    git_hash=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert git_hash=='e68328daa7fb52892ebff9cad28cc512f31e5913'
    assert template.meta['productVersion']=='2.0.24'
    rebuilt=template.to_bytes()
    p=SerumPreset.from_bytes(rebuilt)
    assert p.state==template.state
    assert p.meta['presetName']=='Default'
    end=17+struct.unpack_from('<Q',rebuilt,9)[0]
    assert p.meta['hash']==hashlib.md5(rebuilt[end+8:]).hexdigest()


def test_serum_rejects_corruption(template):
    raw=bytearray(template.to_bytes());raw[-1]^=1
    with pytest.raises(ConversionError,match='hash'): SerumPreset.from_bytes(raw)
    with pytest.raises(ConversionError): SerumPreset.from_bytes(b'XferJson\0'+b'\xff'*16)

@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','[] trailing'])
def test_json_rejects_bad(text):
    with pytest.raises(ConversionError): parse_json(text)


def test_non_string_serum_root_key_rejected(template):
    template.state[1]='malformed module name'
    with pytest.raises(ConversionError,match='root state keys'):
        SerumPreset.from_bytes(template.to_bytes())
