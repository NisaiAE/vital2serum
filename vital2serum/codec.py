"""Bounded CBOR codec and Zstandard adapter.

No executable serialization formats are accepted. Unknown CBOR tags are
preserved, not interpreted. zstandard is preferred; system libzstd is an
offline fallback. Intended for Serum's maps, arrays, strings and numbers.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import struct
from dataclasses import dataclass
from typing import Any
from .errors import ConversionError

MAX_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 16_000_000
MAX_DEPTH = 64

@dataclass(frozen=True)
class Tag:
    number: int
    value: Any

@dataclass(frozen=True)
class Simple:
    value: int


def _pack_float32(values) -> bytes:
    """Items as CBOR float32 (0xfa + big-endian IEEE single), vectorised."""
    import numpy as np
    arr = np.asarray(values, dtype='>f4')
    if not np.all(np.isfinite(arr)):
        raise ConversionError("Non-finite CBOR float")
    out = np.empty((arr.size, 5), dtype=np.uint8)
    out[:, 0] = 0xfa
    out[:, 1:] = arr.view(np.uint8).reshape(-1, 4)
    return out.tobytes()


def _unpack_float32(data: bytes, pos: int, n: int):
    """Return values if data[pos:] holds n consecutive CBOR float32 items, else None."""
    import numpy as np
    end = pos + 5 * n
    if end > len(data):
        return None
    block = np.frombuffer(data, dtype=np.uint8, count=5 * n, offset=pos).reshape(n, 5)
    if not np.all(block[:, 0] == 0xfa):
        return None
    values = np.ascontiguousarray(block[:, 1:]).view('>f4').reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ConversionError("Non-finite CBOR float")
    return values.astype(np.float64).tolist()


class Float32List(list):
    """A list of floats serialised as CBOR float32 items (0xfa), as Serum writes embedded audio.
    Values are rounded to float32 on construction so a decode->encode round trip is exact."""
    def __init__(self, values=()):
        values = list(values)
        super().__init__(struct.unpack(f'<{len(values)}f', struct.pack(f'<{len(values)}f', *values)))

_BREAK = object()

def _head(major: int, n: int) -> bytes:
    if n < 0 or n > 0xffffffffffffffff:
        raise ConversionError("CBOR integer/length is outside uint64")
    if n < 24:
        return bytes([(major << 5) | n])
    for ai, fmt, limit in ((24, '>B', 2**8), (25, '>H', 2**16),
                           (26, '>I', 2**32), (27, '>Q', 2**64)):
        if n < limit:
            return bytes([(major << 5) | ai]) + struct.pack(fmt, n)
    raise AssertionError(n)


def dumps(value: Any) -> bytes:
    budget = [MAX_ITEMS]
    def encode(obj: Any, depth: int) -> bytes:
        budget[0] -= 1
        if budget[0] < 0 or depth > MAX_DEPTH:
            raise ConversionError("CBOR complexity limit exceeded")
        if obj is None:
            return b'\xf6'
        if isinstance(obj, bool):
            return b'\xf5' if obj else b'\xf4'
        if isinstance(obj, int):
            return _head(0, obj) if obj >= 0 else _head(1, -1 - obj)
        if isinstance(obj, float):
            if not math.isfinite(obj):
                raise ConversionError("Non-finite CBOR float")
            return b'\xfb' + struct.pack('>d', obj)
        if isinstance(obj, (bytes, bytearray)):
            return _head(2, len(obj)) + bytes(obj)
        if isinstance(obj, str):
            data = obj.encode('utf-8')
            return _head(3, len(data)) + data
        if isinstance(obj, Float32List):
            budget[0] -= len(obj)
            if budget[0] < 0:
                raise ConversionError("CBOR complexity limit exceeded")
            return _head(4, len(obj)) + _pack_float32(obj)
        if isinstance(obj, (list, tuple)):
            return _head(4, len(obj)) + b''.join(encode(x, depth+1) for x in obj)
        if isinstance(obj, dict):
            return _head(5, len(obj)) + b''.join(
                encode(k, depth+1) + encode(v, depth+1) for k, v in obj.items())
        if isinstance(obj, Tag):
            return _head(6, obj.number) + encode(obj.value, depth+1)
        if isinstance(obj, Simple) and 0 <= obj.value <= 255:
            return bytes([0xe0 | obj.value]) if obj.value < 20 else bytes([0xf8, obj.value])
        raise ConversionError(f"Unsupported CBOR value: {type(obj).__name__}")
    data = encode(value, 0)
    if len(data) > MAX_BYTES:
        raise ConversionError("CBOR payload exceeds 64 MiB")
    return data


def loads(data: bytes) -> Any:
    if len(data) > MAX_BYTES:
        raise ConversionError("CBOR payload exceeds 64 MiB")
    pos, items = 0, 0
    def take(n: int) -> bytes:
        nonlocal pos
        if n < 0 or pos+n > len(data):
            raise ConversionError("Truncated CBOR document")
        result = data[pos:pos+n]
        pos += n
        return result
    def uint(ai: int) -> int:
        if ai < 24:
            return ai
        spec = {24:(1, '>B'),25:(2, '>H'),26:(4, '>I'),27:(8, '>Q')}.get(ai)
        if not spec:
            raise ConversionError("Invalid CBOR additional information")
        return struct.unpack(spec[1], take(spec[0]))[0]
    def read(depth: int, allow_break: bool = False) -> Any:
        nonlocal items
        items += 1
        if depth > MAX_DEPTH or items > MAX_ITEMS:
            raise ConversionError("CBOR complexity limit exceeded")
        head = take(1)[0]
        major, ai = head >> 5, head & 31
        if head == 0xff:
            if allow_break:
                return _BREAK
            raise ConversionError("Unexpected CBOR break")
        if major == 7:
            if ai == 20: return False
            if ai == 21: return True
            if ai == 22: return None
            if ai in (25,26,27):
                n, fmt = {25:(2,'>e'),26:(4,'>f'),27:(8,'>d')}[ai]
                v = struct.unpack(fmt, take(n))[0]
                if not math.isfinite(v):
                    raise ConversionError("Non-finite CBOR float")
                return v
            return Simple(uint(ai))
        indefinite = ai == 31
        n = None if indefinite else uint(ai)
        if major in (0,1):
            if indefinite:
                raise ConversionError("Indefinite integer")
            return n if major == 0 else -1-n
        if major in (2,3):
            if indefinite:
                chunks = []
                while True:
                    if pos >= len(data): raise ConversionError("Truncated CBOR string")
                    if data[pos] == 0xff:
                        take(1); break
                    if data[pos] >> 5 != major or data[pos] & 31 == 31:
                        raise ConversionError("Invalid indefinite string chunk")
                    chunks.append(read(depth+1))
                return (b'' if major == 2 else '').join(chunks)
            b = take(n)
            return b if major == 2 else b.decode('utf-8')
        if major in (4,5):
            if n is not None and n > MAX_ITEMS:
                raise ConversionError("CBOR collection is too large")
            if major == 4 and n is not None and n >= 64 and data[pos:pos+1] == b'\xfa':
                values = _unpack_float32(data, pos, n)
                if values is not None:
                    items += n
                    if items > MAX_ITEMS:
                        raise ConversionError("CBOR complexity limit exceeded")
                    take(5 * n)
                    return Float32List(values)
            result = [] if major == 4 else {}
            count = 0
            while n is None or count < n:
                k = read(depth+1, allow_break=indefinite)
                if k is _BREAK: break
                if major == 4:
                    result.append(k)
                else:
                    if not isinstance(k, (str,bytes,int,float,bool,type(None))):
                        raise ConversionError("Unsupported CBOR map key")
                    if k in result:
                        raise ConversionError("Duplicate CBOR map key")
                    result[k] = read(depth+1)
                count += 1
            return result
        if major == 6 and not indefinite:
            return Tag(n, read(depth+1))
        raise ConversionError("Unsupported CBOR item")
    try:
        result = read(0)
    except (UnicodeError, struct.error, RecursionError) as exc:
        raise ConversionError(f"Invalid CBOR: {exc}") from exc
    if pos != len(data):
        raise ConversionError("Trailing data after CBOR document")
    return result


class _LibZstd:
    def __init__(self) -> None:
        name = ctypes.util.find_library('zstd')
        if not name:
            raise ConversionError("Zstandard is missing. Run: python -m pip install zstandard")
        self.lib = ctypes.CDLL(name)
        for fname, args in {
            'ZSTD_compressBound':[ctypes.c_size_t],
            'ZSTD_compress':[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int],
            'ZSTD_decompress':[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t],
            'ZSTD_findFrameCompressedSize':[ctypes.c_void_p,ctypes.c_size_t],
            'ZSTD_isError':[ctypes.c_size_t],
        }.items():
            fun = getattr(self.lib, fname)
            fun.argtypes = args
            fun.restype = ctypes.c_size_t
        self.lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
        self.lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        self.lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_getErrorName.restype = ctypes.c_char_p
    def checked(self, n: int) -> int:
        if self.lib.ZSTD_isError(n):
            raise ConversionError(self.lib.ZSTD_getErrorName(n).decode('utf-8'))
        return n
    def compress(self, data: bytes) -> bytes:
        src = ctypes.create_string_buffer(data)
        cap = self.lib.ZSTD_compressBound(len(data))
        dst = ctypes.create_string_buffer(cap)
        n = self.checked(self.lib.ZSTD_compress(dst,cap,src,len(data),3))
        return dst.raw[:n]
    def decompress(self, data: bytes, expected: int) -> bytes:
        src = ctypes.create_string_buffer(data)
        declared = self.lib.ZSTD_getFrameContentSize(src,len(data))
        if declared != 0xffffffffffffffff and declared != expected:
            raise ConversionError('Zstd frame content size differs from Serum header')
        size = self.checked(self.lib.ZSTD_findFrameCompressedSize(src,len(data)))
        if size != len(data):
            raise ConversionError("Multiple Zstd frames or trailing compressed data")
        dst = ctypes.create_string_buffer(max(1, expected))
        n = self.checked(self.lib.ZSTD_decompress(dst,expected,src,len(data)))
        if n != expected:
            raise ConversionError("Decompressed payload size differs from Serum header")
        return dst.raw[:n]


def compress(data: bytes) -> bytes:
    if len(data) > MAX_BYTES:
        raise ConversionError("Payload too large")
    try:
        import zstandard as zstd
    except ImportError:
        return _LibZstd().compress(data)
    return zstd.ZstdCompressor(level=3).compress(data)


def decompress(data: bytes, expected: int) -> bytes:
    if not (0 < expected <= MAX_BYTES) or len(data) > MAX_BYTES:
        raise ConversionError("Invalid/excessive Zstd payload size")
    try:
        import zstandard as zstd
    except ImportError:
        return _LibZstd().decompress(data, expected)
    try:
        declared = zstd.frame_content_size(data)
        if declared != -1 and declared != expected:
            raise ConversionError('Zstd frame content size differs from Serum header')
        # reject extra frames and cap unknown-length frames; window is in KiB
        out = zstd.ZstdDecompressor(max_window_size=MAX_BYTES//1024).decompress(
            data, max_output_size=expected, allow_extra_data=False)
    except zstd.ZstdError as exc:
        raise ConversionError(f"Invalid Zstd frame: {exc}") from exc
    if len(out) != expected:
        raise ConversionError("Decompressed payload size differs from Serum header")
    return out
