"""Serum 2 container writer/reader, based on btesser/serum2vital's schema.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from . import codec
from .errors import ConversionError

MAGIC = b'XferJson\x00'


def unique_object(pairs: list) -> dict:
    result = {}
    for k, v in pairs:
        if k in result:
            raise ConversionError(f"Duplicate JSON key: {k}")
        result[k] = v
    return result


def parse_json(data: str | bytes) -> Any:
    def bad_float(text: str):
        raise ConversionError(f"Invalid JSON number: {text}")
    try:
        return json.loads(data, object_pairs_hook=unique_object, parse_constant=bad_float)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ConversionError(f"Invalid JSON: {exc}") from exc


def plain(module: dict) -> dict:
    p = module.get('plainParams', 'default')
    if p == 'default':
        return {}
    if not isinstance(p, dict):
        raise ConversionError("plainParams must be a map or 'default'")
    return p


@dataclass
class SerumPreset:
    meta: dict
    state: dict

    def to_bytes(self) -> bytes:
        raw = codec.dumps(self.state)
        payload = codec.compress(raw)
        meta = copy.deepcopy(self.meta)
        meta['hash'] = hashlib.md5(payload).hexdigest()
        header = json.dumps(meta, ensure_ascii=False, allow_nan=False,
                            separators=(',', ':')).encode('utf-8')
        raw_size = len(raw)
        return MAGIC + struct.pack('<Q', len(header)) + header + struct.pack('<II', raw_size, 2) + payload

    @classmethod
    def from_bytes(cls, data: bytes) -> 'SerumPreset':
        if len(data) > codec.MAX_BYTES + 1024*1024:
            raise ConversionError("Serum preset is too large")
        if not data.startswith(MAGIC) or len(data) < len(MAGIC)+16:
            raise ConversionError("Not a Serum 2 .SerumPreset (XferJson header missing)")
        n = struct.unpack_from('<Q',data,len(MAGIC))[0]
        start = len(MAGIC)+8
        end = start+n
        if n > 1024*1024 or end+8 >= len(data):
            raise ConversionError("Invalid Serum metadata length")
        meta = parse_json(data[start:end])
        if not isinstance(meta,dict): raise ConversionError("Serum metadata is not a map")
        if meta.get('product', 'Serum2') != 'Serum2':
            raise ConversionError('Only Serum 2 templates are supported')
        if meta.get('fileType', 'SerumPreset') != 'SerumPreset':
            raise ConversionError("Template is not a SerumPreset container")
        size, encoding = struct.unpack_from('<II',data,end)
        if encoding != 2: raise ConversionError(f"Unsupported Serum encoding: {encoding}")
        payload = data[end+8:]
        claimed = meta.get('hash')
        if claimed and str(claimed).lower() != hashlib.md5(payload).hexdigest():
            raise ConversionError("Serum metadata hash does not match its payload")
        state = codec.loads(codec.decompress(payload,size))
        if not isinstance(state,dict): raise ConversionError("Serum CBOR state is not a map")
        if not all(isinstance(key,str) for key in state):
            raise ConversionError("Serum root state keys must be strings")
        return cls(meta,state)

    @classmethod
    def read(cls, path: str | Path) -> 'SerumPreset':
        path = Path(path)
        if path.stat().st_size > codec.MAX_BYTES + 1024*1024:
            raise ConversionError("Serum preset is too large")
        return cls.from_bytes(path.read_bytes())

    def module(self, name: str) -> dict:
        value = self.state.setdefault(name, {'plainParams':'default'})
        if not isinstance(value,dict):
            raise ConversionError(f"Template module {name} is not a map")
        return value

    def set_params(self, name: str, values: dict) -> dict:
        m = self.module(name)
        m['plainParams'] = values or 'default'
        return m


def clean_template(template: SerumPreset) -> SerumPreset:
    """Require a saved Init-like preset; never inherit active modulation/FX.

    Unknown UI and version data is retained. This is deliberately not a
    general 'erase any Serum patch' function.
    """
    for key in ('Oscillator0','Global0','Env0'):
        if not isinstance(template.state.get(key),dict):
            raise ConversionError(f"Init template is missing {key}")
    for name,m in template.state.items():
        if not isinstance(m,dict): continue
        p = plain(m) if 'plainParams' in m else {}
        if name.startswith('ModSlot') and any(m.get('source', [0,0])):
            raise ConversionError("Template contains modulation. Save an initialized Serum 2 preset.")
        if name.startswith('FXRack') and m.get('FX'):
            raise ConversionError("Template contains effects. Save an initialized Serum 2 preset.")
        if re.fullmatch(r'Oscillator[1-4]', name) and p.get('kParamEnable',0):
            raise ConversionError("Template has extra oscillators enabled. Use Serum 2 Init.")
        if name.startswith(('Arp','ClipPlayer','PitchQuantizer','VoicePanel')) and p:
            raise ConversionError(f"Non-default {name} in template. Use Serum 2 Init.")
    out = copy.deepcopy(template)
    for name,m in out.state.items():
        if not isinstance(m,dict): continue
        if re.fullmatch(r'(Env|LFO|Macro|VoiceFilter|RoutingSlot|Oscillator)\d+|Global0',name):
            m['plainParams'] = 'default'
        if name.startswith('Oscillator'):
            for child in m.values():
                if isinstance(child,dict) and 'plainParams' in child:
                    child['plainParams'] = 'default'
        if name.startswith('ModSlot'):
            # Retain any structural/UI keys, but remove obsolete destinations.
            for k in list(m):
                if k.startswith('destModule'): del m[k]
            m['source'] = [0,0]
            m['plainParams'] = 'default'
        if name.startswith('FXRack'):
            m['FX'] = []
            m['plainParams'] = 'default'
    return out
