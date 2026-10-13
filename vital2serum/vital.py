"""Validated Vital JSON input with defaults and scalar-use accounting."""
from __future__ import annotations
import math
from pathlib import Path
from .defaults import DEFAULTS
from .errors import ConversionError
from .ranges import RANGES
from .serum import parse_json

MAX_VITAL_BYTES = 64*1024*1024

class VitalPreset:
    def __init__(self, doc: dict, source: str = '<memory>'):
        if not isinstance(doc,dict) or not isinstance(doc.get('settings'),dict):
            raise ConversionError("Vital input must be an object containing settings")
        version = doc.get('synth_version', '')
        if version and not isinstance(version,str):
            raise ConversionError("synth_version must be text")
        if version:
            major = version.split('.')[0]
            if not major.isdigit():
                raise ConversionError(f"Unrecognised synth_version {version!r}")
            if int(major) < 1:
                raise ConversionError(f"Vital {version} predates the 1.0 preset format; open and re-save it in Vital 1.0.7 or later first")
            if int(major) > 1:
                raise ConversionError(f"Unsupported Vital version {version}")
        self.doc, self.settings, self.source = doc, doc['settings'], source
        self.used: set[str] = set()
        # Static modulation folded into the stored values (see mapping.fold_static_modulations).
        self.offsets: dict[str,float] = {}
        self.folded: set[int] = set()
        structured = {'wavetables','lfos','modulations','custom_warps','random_values','sample'}
        for key,value in self.settings.items():
            if key in structured: continue
            if not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ConversionError(f"Vital setting {key} must be a finite number")
        for key,limit in [('wavetables',3),('lfos',8),('modulations',64)]:
            value = self.settings.get(key,[])
            if not isinstance(value,list) or len(value)>limit:
                raise ConversionError(f"Vital {key} must be a list with at most {limit} entries")
            if any(x is not None and not isinstance(x,dict) for x in value):
                raise ConversionError(f"Invalid {key} entry")
        for field in ['author','comments','preset_name','preset_style','macro1','macro2','macro3','macro4']:
            if field in doc and not isinstance(doc[field],str):
                raise ConversionError(f"Vital metadata {field} must be text")

    @classmethod
    def read(cls,path: str | Path) -> 'VitalPreset':
        path = Path(path)
        if path.stat().st_size > MAX_VITAL_BYTES:
            raise ConversionError("Vital preset exceeds 64 MiB")
        try:
            text = path.read_text(encoding='utf-8-sig').rstrip('\x00 \t\r\n')
        except UnicodeError as exc:
            raise ConversionError("Vital input is not UTF-8 JSON") from exc
        return cls(parse_json(text),str(path))

    @property
    def name(self):
        return self.doc.get('preset_name') or Path(self.source).stem

    def get(self,key: str) -> float:
        self.used.add(key)
        if key not in DEFAULTS and key not in self.settings:
            raise ConversionError(f"No known Vital default for {key}")
        value = self.settings.get(key,DEFAULTS.get(key))
        if not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ConversionError(f"Invalid numerical parameter {key}")
        value = float(value)+self.offsets.get(key,0.0)
        lo,hi = RANGES.get(key,(None,None))
        if lo is not None:
            value = max(lo,min(hi,value))
        return value

    def integer(self,key: str,lo: int,hi: int) -> int:
        v = self.get(key)
        if key in self.offsets:
            # A folded modulation leaves a fraction; Vital rounds to the nearest step
            # (measured for unison voices: 7.4 -> 7, 7.6 -> 8, 2.5 -> 3).
            v = float(math.floor(v+0.5))
        if abs(v-round(v)) > 1e-5 or not lo <= round(v) <= hi:
            raise ConversionError(f"{key} must be an integer from {lo} through {hi}")
        return round(v)

    def array(self,key: str) -> list:
        self.used.add(key)
        return self.settings.get(key,[])

    def changed(self,key: str) -> bool:
        if key not in self.settings: return False
        value = self.settings[key]
        if not isinstance(value,(int,float)): return bool(value)
        return key not in DEFAULTS or not math.isclose(value,DEFAULTS[key],rel_tol=1e-6,abs_tol=1e-7)

    def ignore_prefix(self,prefix: str) -> None:
        self.used.update(k for k in self.settings if k.startswith(prefix))
