"""Extract simple Vital wave sources/sample PCM to Serum-compatible WAV assets.
Only Wave Source components are rendered; procedural components are NOT guessed.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import base64
import binascii
import hashlib
import math
import struct
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import numpy as np
from .errors import ConversionError
from .spectral import STATIC_MORPHS, selected_positions, spectral_frames
from .report import Report
from .wavetable import FRAME_SIZE, morph_sweep, phase_warp, render_wavetable, spectral_morph, warp_sweep

@dataclass
class Asset:
    relative_path: str
    data: bytes
    frames: int
    channels: int = 1
    sample_rate: int = 44100
    pcm: list = field(default_factory=list)      # per-channel float32 samples for embedding
    display_name: str = ''
    gain: float = 1.0                            # amplitude removed from the audio to keep |x| <= 1

    @property
    def serum_reference(self) -> str:
        return str(PurePosixPath(self.relative_path).relative_to(PurePosixPath(self.relative_path).parts[0]))


def _b64(text: str, limit: int) -> bytes:
    if not isinstance(text,str) or len(text)>limit*2:
        raise ConversionError("Invalid/excessive base64 audio")
    try:
        out = base64.b64decode(text,validate=True)
    except (ValueError,binascii.Error) as exc:
        raise ConversionError("Malformed base64 audio") from exc
    if len(out)>limit: raise ConversionError("Decoded audio is too large")
    return out


def _chunk(tag: bytes,data: bytes) -> bytes:
    return tag+struct.pack('<I',len(data))+data+(b'\0' if len(data)&1 else b'')


def wav_bytes(pcm: bytes, channels: int, sample_rate: int, floating: bool,
              wavetable: bool = False) -> bytes:
    width = 4 if floating else 2
    if channels not in (1,2) or not 8000 <= sample_rate <= 384000 or len(pcm)%(channels*width):
        raise ConversionError("Invalid PCM/WAV format")
    fmt = struct.pack('<HHIIHH',3 if floating else 1,channels,sample_rate,
                      sample_rate*channels*width,channels*width,width*8)
    body = b'WAVE'+_chunk(b'fmt ',fmt)
    if floating: body += _chunk(b'fact',struct.pack('<I',len(pcm)//(channels*width)))
    if wavetable: body += _chunk(b'clm ',b'<!>2048 0 0 0\0')
    body += _chunk(b'data',pcm)
    return b'RIFF'+struct.pack('<I',len(body))+body


def _asset(data: bytes, directory: str, frames: int,channels: int=1,sample_rate: int=44100,
           pcm: list | None=None, display_name: str='') -> Asset:
    digest = hashlib.sha256(data).hexdigest()[:24]
    return Asset(f'{directory}/vital2serum/{digest}.wav',data,frames,channels,sample_rate,pcm or [],display_name)


def wavetable(table: dict | None, report: Report, feature: str, allow_placeholder: bool = False,
              morph: tuple[int,float] | None = None, warp: tuple[int,float,float] | None = None,
              sweep: str | None = None, position: float = 0.0, headroom: float = 1.0) -> Asset:
    """`morph` is (Vital spectral morph type, amount) and `warp` (distortion type, amount, phase);
    both are rendered into the frames. `sweep` ('morph' or 'warp') lays that amount out along the
    table instead, starting from the frame Vital plays at wave_frame `position`. `headroom`
    divides the table and goes to the oscillator level with the rest of the gain."""
    try:
        return _wavetable(table,report,feature,morph,warp,sweep,position,headroom)
    except ConversionError as exc:
        if not allow_placeholder: raise ConversionError(f'{feature}: {exc}') from exc
        # Explicit opt-in only: never silently pretend a missing table is original audio.
        report.note('unsupported',feature,'Wavetable replaced with a generated sine by explicit --allow-placeholder.',reason=str(exc))
        values = [math.sin(2*math.pi*i/FRAME_SIZE) for i in range(FRAME_SIZE)]
        data = wav_bytes(struct.pack('<2048f',*values),1,44100,True,True)
        return _asset(data,'Tables',FRAME_SIZE,pcm=[values],display_name='vital2serum sine')


def _wavetable(table: dict | None,report: Report,feature: str,morph: tuple[int,float] | None = None,
               warp: tuple[int,float,float] | None = None,sweep: str | None = None,position: float = 0.0,
               headroom: float = 1.0) -> Asset:
    new_morph=morph is not None and morph[0] in STATIC_MORPHS
    frames, notes = render_wavetable(table, feature, full=new_morph)
    for note in notes:
        report.note('approximation',feature,note)
    if new_morph:
        frames=selected_positions(spectral_frames(frames,*morph))
        morph=None
    # Vital morphs the frame it plays, then reads it at the warped phase.
    if sweep:
        base = frames[min(int(max(position,0.0)),frames.shape[0]-1)]
        if sweep == 'morph':
            frames = morph_sweep(base, morph[0])
            if warp is not None: frames = phase_warp(frames, *warp)
        else:
            if morph is not None: base = spectral_morph(base.reshape(1,FRAME_SIZE), *morph)[0]
            frames = warp_sweep(base, warp[0], warp[2])
    else:
        if morph is not None: frames = spectral_morph(frames, *morph)
        if warp is not None: frames = phase_warp(frames, *warp)
    count = int(frames.shape[0])
    if count > 1 and not sweep:
        report.note('approximation',feature,'Wavetable morph is sampled onto 256 time-domain frames; engine interpolation may differ.')
    flat = frames.reshape(-1).astype(np.float64)
    # Serum clips table samples to +-1 (measured on 2.1.5); Vital does not. Scale the table down
    # and hand the gain to the oscillator level.
    peak = float(np.abs(flat).max()) if flat.size else 0.0
    gain = max(peak, headroom)
    flat = (flat / gain).astype(np.float32)
    data = wav_bytes(flat.astype('<f4').tobytes(),1,44100,True,True)
    name = table.get('name') if isinstance(table.get('name'),str) else ''
    a = _asset(data,'Tables',count*FRAME_SIZE,pcm=[flat.tolist()],display_name=name or 'Vital table')
    a.gain = gain
    return a


def sample(obj: dict) -> Asset:
    if not isinstance(obj,dict): raise ConversionError('Missing sample object')
    n = obj.get('length',0)
    rate = obj.get('sample_rate',44100)
    if not isinstance(n,int) or not 0<n<=10_000_000:
        raise ConversionError('Invalid sample length (limit: 10 million samples per channel)')
    if not isinstance(rate,int) or not 8000<=rate<=384000:
        raise ConversionError('Invalid sample rate')
    left = _b64(obj.get('samples'),n*2)
    if len(left)!=n*2: raise ConversionError('Sample length does not match PCM data')
    channels = 1
    if obj.get('samples_stereo'):
        right = _b64(obj['samples_stereo'],n*2)
        if len(right)!=n*2: raise ConversionError('Stereo channel lengths differ')
        # bytes-level interleaving avoids changing signed int16 PCM values.
        inter = bytearray(n*4)
        inter[0::4],inter[1::4]=left[0::2],left[1::2]
        inter[2::4],inter[3::4]=right[0::2],right[1::2]
        left,channels=bytes(inter),2
    data=wav_bytes(left,channels,rate,False)
    pcm=[(np.frombuffer(left,dtype='<i2')[c::channels]/32768.0).tolist() for c in range(channels)]
    name=obj.get('name') if isinstance(obj.get('name'),str) else ''
    return _asset(data,'Noises',n,channels,rate,pcm=pcm,display_name=name or 'Vital sample')
