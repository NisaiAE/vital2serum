"""Modulation translation with evidence-based destination IDs.
Unknown numeric parameter IDs are never guessed. --learn-from reads IDs from
Serum-created routings; a conflicting ID table is rejected.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math
import re
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from .curves import line_curve, line_value, remap_curve, sampled_curve, vital_power_curve, vital_remap
from .errors import ConversionError
from .ranges import RANGES
from .serum import SerumPreset

# Observed decoded Serum 2 modulation records, not inferred enum positions.
# Serum 2.1.5 resolves a routing by its parameter name and falls back to the number only when the
# name is unknown (measured), so the numbers below matter less than the names.
# Entries marked [lib] were read from Serum 2 presets (productVersion 2.0.11-2.0.24); the count
# is the number of routings that agreed.
# A few 2.0.11 files still carry older numbers (kParamDetune 10, kParamDetuneWid 11, kParamTablePos 1);
# the newer numbers, also present in 2.0.12+ files, are used. Remaining entries come from the
# format notes referenced by btesser/serum2vital.
KNOWN_IDS = {
    ('Oscillator','kParamVolume'):1,      # [lib] 1884
    ('Oscillator','kParamPan'):2,         # [lib] 241
    ('Oscillator','kParamOctave'):3,      # [lib] 102
    ('Oscillator','kParamPitch'):4,       # [lib] 43
    ('Oscillator','kParamFine'):5,        # [lib] 571
    ('Oscillator','kParamCoarsePit'):6,   # [lib] 213
    ('Oscillator','kParamDetune'):26,     # [lib] 168 (2.0.11 also shows 10 four times)
    ('Oscillator','kParamDetuneWid'):27,  # [lib] 29 (2.0.11 also shows 11 once)
    ('WTOsc','kParamWarp'):0,             # [lib] 582
    ('WTOsc','kParamTablePos'):6,         # [lib] 519 (2.0.11 also shows 1)
    ('WTOsc','kParamInitialPhase'):8,     # [lib] 14
    ('WTOsc','kParamRandomPhase'):9,      # [lib] 8
    ('VoiceFilter','kParamWet'):1,        # [lib] 155
    ('VoiceFilter','kParamFreq'):3,       # [lib] 1551
    ('VoiceFilter','kParamReso'):4,       # [lib] 233
    ('VoiceFilter','kParamDrive'):5,      # [lib] 150
    ('VoiceFilter','kParamVar'):6,        # [lib] 83
    ('VoiceFilter','kParamLevelOut'):8,   # [lib] 153
    ('Oscillator','kParamUnisonStereo'):28,  # [lib] 25
    ('Env','kParamAttack'):0,             # [lib] 317
    ('Env','kParamHold'):1,               # [lib] 5
    ('Env','kParamDecay'):2,              # [lib] 361
    ('Env','kParamSustain'):3,            # [lib] 188
    ('Env','kParamRelease'):4,            # [lib] 94
    ('LFO','kParamRate'):0,               # [lib] 297
    ('LFO','kParamPhase'):4,              # [lib] 18
    ('LFO','kParamRise'):2,               # [lib] 8
    ('LFO','kParamDelay'):3,              # [lib] 4
    ('Macro','kParamValue'):0,            # [lib] 238
    ('Global','kParamPortamentoTime'):3,  # [lib] 12
    ('Global','kParamVoiceAmp'):2,        # [lib] 20, all from velocity
    ('NoiseOsc','kParamColor'):0,         # read from Serum presets routing to the Noise oscillator
    ('NoiseOsc','kParamFine'):1,
    ('NoiseOsc','kParamInitialPhase'):2,
    ('NoiseOsc','kParamRandomPhase'):3,
    ('FXEQ','kParamGain2'):6, ('FXEQ','kParamFreq1'):1, ('FXEQ','kParamLevelOut'):9,
    # FX rack entries are addressed by their position in the rack; the numbers match the voice filter's.
    ('FXFilter','kParamWet'):1,           # [lib] 132
    ('FXFilter','kParamFreq'):3,          # [lib] 354
    ('FXFilter','kParamReso'):4,          # [lib] 54
    ('FXFilter','kParamDrive'):5,         # [lib] 46
    ('FXFilter','kParamVar'):6,           # [lib] 39
    ('FXFilter','kParamLevelOut'):8,      # [lib] 33
    ('FXDistortion','kParamFreq'):5,
    ('FXDistortion','kParamDrive'):2,
    ('FXDistortion','kParamWet'):1,
    ('FXDistortion','kParamLevelOut'):8,
    ('FXDelay','kParamWet'):1, ('FXDelay','kParamLevelOut'):12,
    ('FXPhaser','kParamFreq'):6,
    ('FXPhaser','kParamWet'):1,
    ('FXPhaser','kParamFeedback'):7,
}
SOURCES = {
    'mod_wheel':1, **{f'env_{i}':i+1 for i in range(1,5)},
    **{f'lfo_{i}':i+5 for i in range(1,9)},
    'velocity':16,'note':17,'aftertouch':18,'random':21,
    **{f'macro_control_{i}':i+24 for i in range(1,5)},
    'pitch_wheel':33,'slide':35,'lift':37,
}
# Serum's first envelope is its amplitude envelope, converted to follow Vital's squared one, and
# it reads the same as a modulation source (measured). Vital's value is the square root of it.
SOURCE_LAWS = {'env_1':math.sqrt}
SOURCE_LAW_INVERSES = {'env_1':lambda x: x*x}
SOURCE_CURVE_POINTS = 33
RAMP_OFF = -10.0                  # modulation_N_ramp_up / ramp_down when unused (Vital 1.6)
STEP_WIDTH = 1e-4                 # of the source range, either side of a step in a sampled curve
# Serum has eight macros (sources 25..32); Vital uses the first four.
MACRO_SOURCE = 25

@dataclass
class Destination:
    module: str
    index: int
    parameter: str
    scale: float = 1.0
    nonlinear: bool = False
    # Vital value -> position of the Serum control on its 0..1 modulation scale, for parameters
    # whose two scales are not proportional (times, detune). None: proportional, use `scale`.
    law: Callable[[float], float] | None = None
    # Opt-in native-position tolerance and point budget; ordinary laws retain their 33 points.
    curve_policy: tuple[float, int] | None = None


def _version_key(text) -> tuple:
    out=[]
    for piece in str(text or '').split('.'):
        digits=''.join(ch for ch in piece if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def learn_ids(paths: list[Path], warn=None) -> dict[tuple[str,str],int]:
    """Read destination IDs from Serum-saved routings.

    Real libraries mix Serum versions whose numbering changed (2.0.11 -> 2.0.12); for each
    (module, parameter) the ID seen in the newest productVersion wins, ties by count. Conflicts
    and unreadable files are reported through `warn` (default: stderr) instead of aborting.
    """
    import sys
    warn=warn or (lambda text: print(f'vital2serum: {text}',file=sys.stderr))
    seen: dict[tuple[str,str],dict[int,tuple[tuple,int]]]={}
    for path in paths:
        try:
            patch=SerumPreset.read(path)
        except (ConversionError,OSError) as exc:
            warn(f'--learn-from skipped {path}: {exc}'); continue
        version=_version_key(patch.meta.get('productVersion'))
        for name,m in patch.state.items():
            if not name.startswith('ModSlot') or not isinstance(m,dict): continue
            typ=m.get('destModuleTypeString');param=m.get('destModuleParamName');number=m.get('destModuleParamID')
            if not isinstance(typ,str) or not isinstance(param,str) or number is None: continue
            if not isinstance(number,int) or isinstance(number,bool) or not 0<=number<=65535:
                warn(f'--learn-from skipped an invalid destination ID in {path}'); continue
            entry=seen.setdefault((typ,param),{})
            best,count=entry.get(number,((),0))
            entry[number]=(max(best,version),count+1)
    result=dict(KNOWN_IDS)
    for key,candidates in seen.items():
        number=max(candidates,key=lambda n:candidates[n])
        if len(candidates)>1:
            warn(f'--learn-from: {key} has several IDs {sorted(candidates)}; using {number} (newest Serum version)')
        if key in KNOWN_IDS and KNOWN_IDS[key]!=number:
            warn(f'--learn-from: {key} learned as {number}, overriding built-in {KNOWN_IDS[key]}')
        result[key]=number
    return result


MAX_SLOTS = 64
# Vital stores a modulation amount as a fraction of the destination's full range (measured in
# Vital 1.0.7: macro 1.0 with amount 0.25 on osc tune, range +-1 semitone, moved the pitch 50 cents).
# A bipolar routing swings half of that to either side. A modulation of modulation_N_amount
# moves that amount by its own amount * 2 (the amount's range is -1..1).
MOD_AMOUNT_RANGE = 2.0
MAX_DEPTH = 4
# Serum adds routings that share a destination, so an amount above 100 % is spread over slots.
MAX_SPLIT = 4
# Macros 5..8 carry the product of two sources when Vital scales an amount by more than one
# (measured: an aux source reads a modulated macro without delay).
BUS_MACROS = (4,5,6,7)
# Sources whose value is a stored control rather than a moving signal.
STATIC_SOURCES = {f'macro_control_{i}' for i in range(1,5)} | {'mod_wheel'}
_AMOUNT = re.compile(r'modulation_(\d+)_amount')


@dataclass
class Modulation:
    slot: int
    source: str
    target: str
    amount: float
    bipolar: bool
    power: float
    stereo: bool
    points: tuple = ()          # Vital's remap curve ("line mapping") as (x, y) with y = 0 at the top;
    powers: tuple = ()          # empty when it is the straight default

    def shaped(self, raw: float) -> float:
        """The source at `raw` (0..1) after this routing's remap curve and power, still 0..1.

        Vital looks the source up in the remap curve first and bends the result with the power
        (modulation_connection_processor.cpp; the order matters only where both are set).
        """
        if self.points: raw=line_value(self.points,self.powers,min(1.0,max(0.0,raw)))
        return vital_remap(raw,self.power,self.bipolar)

    def value(self, raw: float) -> float:
        """Signed value this routing reads from a source at `raw` (0..1), before its amount."""
        return self.shaped(raw)-(0.5 if self.bipolar else 0.0)

    @property
    def shape(self) -> tuple:
        """What distinguishes this routing's reading of its source from another's."""
        return (self.power,self.bipolar and abs(self.power)>=0.01,self.points,self.powers)


def _line_mapping(m: dict,feature: str) -> tuple:
    """(points, powers) of a routing's remap curve, or two empty tuples for the straight default."""
    line=m.get('line_mapping')
    if not isinstance(line,dict): return (),()
    flat=line.get('points');powers=line.get('powers') or []
    if not isinstance(flat,list) or len(flat)<4 or len(flat)%2 or not all(isinstance(x,(int,float)) for x in list(flat)+list(powers)):
        return (),()
    points=tuple((float(flat[i]),float(flat[i+1])) for i in range(0,len(flat),2))
    if any(b[0]<a[0]-1e-9 for a,b in zip(points,points[1:])):
        raise ConversionError(f'{feature}: remap curve points are not in order')
    powers=tuple(float(p) for p in powers[:len(points)-1])
    # A step at the very end is the line's way back to its start, for looping. Vital holds the
    # level before it at full scale (a source resting at 1 measured as the upper level).
    while len(points)>2 and points[-1][0]-points[-2][0]<1e-9:
        points=points[:-1];powers=powers[:len(points)-1]
    straight=all(abs(1.0-y-x)<1e-6 for x,y in points) and points[0][0]<1e-6 and points[-1][0]>1-1e-6 and all(abs(p)<0.01 for p in powers)
    return ((),()) if straight else (points,powers)


def active_modulations(v) -> dict[int,Modulation]:
    """Vital's routings by slot number, without empty and bypassed slots."""
    out: dict[int,Modulation]={}
    for i,m in enumerate(v.array('modulations'),1):
        if not m or not m.get('source') or not m.get('destination'): continue
        source,target=m['source'],m['destination']
        if not isinstance(source,str) or not isinstance(target,str):
            raise ConversionError(f'Invalid modulation source/destination at slot {i}')
        prefix=f'modulation_{i}_'
        if v.get(prefix+'bypass'):
            v.ignore_prefix(prefix);continue
        out[i]=Modulation(i,source,target,v.get(prefix+'amount'),bool(v.integer(prefix+'bipolar',0,1)),
                          v.get(prefix+'power'),bool(v.get(prefix+'stereo')),*_line_mapping(m,prefix+'line_mapping'))
    return out


def amount_modulators(mods: dict[int,Modulation]) -> dict[int,list[int]]:
    """Slots whose destination is another slot's amount, grouped by that slot."""
    out: dict[int,list[int]]={}
    for i,m in mods.items():
        hit=_AMOUNT.fullmatch(m.target)
        if hit: out.setdefault(int(hit.group(1)),[]).append(i)
    return out


def mix_route_supported(v,mods: dict[int,Modulation],into: dict[int,list[int]],i: int,report=None) -> bool:
    """A memoryless unit-source response must cover the primary and every amount source."""
    pending=[i];seen=set();supported=True
    while pending:
        j=pending.pop()
        if j in seen: continue
        seen.add(j);pending.extend(into.get(j,()))
        m=mods[j];prefix=f'modulation_{j}_'
        if j!=i and abs(m.amount)<=1e-12 and not into.get(j): continue
        issues=[]
        if any(v.settings.get(prefix+key,RAMP_OFF)>RAMP_OFF+1e-6 for key in ('ramp_up','ramp_down')):
            issues.append(('ramp','A mix response with a primary or auxiliary ramp is not converted; the stored mix is retained.'))
        lo,hi=RANGES.get(m.source,(0.0,1.0))
        if lo<0.0 or hi>1.0:
            issues.append(('source_range','A mix response requires source ranges within 0..1, including auxiliary sources; the stored mix is retained.'))
        supported=supported and not issues
        if report is not None:
            for feature,message in issues:
                feature=prefix+feature
                if not any(note['feature']==feature and note.get('destination')==mods[i].target for note in report.notes):
                    report.note('unsupported',feature,message,source=m.source,destination=mods[i].target)
    return supported


def static_amount(v,mods: dict[int,Modulation],into: dict[int,list[int]],i: int,moved: set[str],
                  seen: frozenset=frozenset(), *, ignore_zero_leaves: bool=False) -> tuple[float,list[int]] | None:
    """Amount of slot i when only stored controls scale it, with the slots that take part.

    None when a moving source (an LFO, an envelope, velocity or a control that is itself
    modulated) is involved. Vital limits the result to -1..1.
    """
    total=mods[i].amount;used=[i]
    for j in into.get(i,()):
        m=mods[j]
        if ignore_zero_leaves and abs(m.amount)<=1e-12 and not into.get(j): continue
        if j in seen or len(seen)>=MAX_DEPTH or m.source not in STATIC_SOURCES or m.source in moved or m.stereo:
            return None
        inner=static_amount(v,mods,into,j,moved,seen|{i},ignore_zero_leaves=ignore_zero_leaves)
        if inner is None: return None
        total+=MOD_AMOUNT_RANGE*m.value(v.get(m.source))*inner[0];used+=inner[1]
    return max(-1.0,min(1.0,total)),used


def _terms(mods,into,i: int,report,depth: int=0,seen: frozenset=frozenset()) -> list:
    """Slot i's amount as a sum of (coefficient, factors); a factor is a slot scaling the amount."""
    out=[]
    if abs(mods[i].amount)>=1e-12: out.append((mods[i].amount,()))
    for j in into.get(i,()):
        if j in seen or depth>=MAX_DEPTH:
            report.note('unsupported',f'modulation_{j}_','Modulation of a modulation amount is nested too deeply; omitted.')
            continue
        for coefficient,factors in _terms(mods,into,j,report,depth+1,seen|{i}):
            out.append((coefficient*MOD_AMOUNT_RANGE,factors+(j,)))
    return out


def _source_curve(m: Modulation) -> dict | None:
    """Curve on a slot's source giving Vital's value: its remap curve and power, after any SOURCE_LAWS."""
    law=SOURCE_LAWS.get(m.source)
    if law is None and not m.points: return vital_power_curve(m.power,m.bipolar)
    if law is None and abs(m.power)<0.01: return line_curve(m.points,m.powers)
    # Sampled where the combination has no closed form. The grid is even in Vital's source value,
    # so the bend of a source law near zero keeps its shape, and takes in the remap curve's own
    # points with a sample either side of a step.
    grid={i/(SOURCE_CURVE_POINTS-1) for i in range(SOURCE_CURVE_POINTS)}
    for x,_ in m.points: grid|={min(1.0,max(0.0,x+d)) for d in (-STEP_WIDTH,0.0,STEP_WIDTH)}
    back=SOURCE_LAW_INVERSES[m.source] if law else (lambda x: x)
    return remap_curve([(back(x),m.shaped(x)) for x in sorted(grid)])


def _trajectory_source_curve(m: Modulation, *, tolerance: float = 2e-5, limit: int = 480) -> dict | None:
    """Bound a raw square-root projection without changing ordinary source curves.

    The secant's maximum error lies at the square of the mean endpoint roots.
    This bounds the serialized curve; native rounding and table quantization remain.
    """
    law=SOURCE_LAWS.get(m.source)
    if law is None: return _source_curve(m)
    if law is not math.sqrt or m.points or m.bipolar or m.power!=0:
        raise ValueError('Only a raw square-root trajectory projection is supported')
    if not math.isfinite(tolerance) or tolerance<=0 or not isinstance(limit,int) or not SOURCE_CURVE_POINTS<=limit<=480:
        raise ValueError('Invalid trajectory projection policy')
    curve=_source_curve(m)
    values={float(x):float(1-y) for x,y in zip(curve['xVals'],curve['yVals'])}
    def interval(a,b):
        ya,yb=math.sqrt(a),math.sqrt(b)
        x=((ya+yb)/2)**2
        error=(yb-ya)**2/(4*(ya+yb)) if ya+yb else 0.0
        return (-error,a,b,x)
    keys=sorted(values)
    pending=[interval(a,b) for a,b in zip(keys,keys[1:])]
    heapq.heapify(pending)
    while pending:
        negative,a,b,x=heapq.heappop(pending)
        if -negative<=tolerance: break
        if len(values)>=limit or not a<x<b:
            raise ValueError('Trajectory source response exceeds point budget')
        values[x]=math.sqrt(x)
        heapq.heappush(pending,interval(a,x))
        heapq.heappush(pending,interval(x,b))
    return remap_curve(sorted(values.items()))


class _Writer:
    def __init__(self,ctx,ids):
        self.ctx,self.ids=ctx,ids
        self.next=0
        self.buses: dict[tuple,int]={}
        self.free=list(BUS_MACROS)
        self.full=False

    def slot(self,source: int,aux: int,dest: Destination,number: int,amount: float,bipolar: bool,
             main: dict | None=None,aux_curve: dict | None=None) -> int | None:
        if self.next>=MAX_SLOTS:
            if not self.full:
                self.ctx.r.note('unsupported','modulations',f'Serum has {MAX_SLOTS} modulation slots; further routings are omitted.')
            self.full=True
            return None
        params={'kParamAmount':float(amount),'kParamBipolar':float(bipolar)}
        module={'source':[source,aux],'destModuleTypeString':dest.module,'destModuleID':dest.index,
                'destModuleParamName':dest.parameter,'destModuleParamID':number,'plainParams':params}
        if main or aux_curve:
            # Same layout as slots Serum saves with a drawn curve: a flag per curve and the pair in flex.
            if main: params['kParamMainCurveData']=1.0
            if aux_curve: params['kParamAuxCurveData']=1.0
            module['flex']=[main or {},aux_curve or {}]
        self.ctx.s.state[f'ModSlot{self.next}']=module
        self.next+=1
        return self.next-1

    def bus_keys(self,mods,groups) -> set[tuple]:
        """Missing product prefixes, shared across all requested factor groups."""
        keys=set()
        for factors in groups:
            key=tuple((mods[j].source,)+mods[j].shape for j in factors)
            if key in self.buses: continue
            for length in range(2,len(key)+1):
                if key[:length] not in self.buses: keys.add(key[:length])
        return keys

    def can_bus(self,mods,groups) -> bool:
        needed=len(self.bus_keys(mods,groups))
        # A product without room for its final destination cannot carry a routing.
        return (not needed or self.ids.get(('Macro','kParamValue')) is not None) and needed<=len(self.free) and self.next+needed<MAX_SLOTS

    def bus(self,mods,factors: tuple) -> int | None:
        """Source ID of a spare macro carrying the product of the factor slots' sources."""
        key=tuple((mods[j].source,)+mods[j].shape for j in factors)
        if key in self.buses: return self.buses[key]
        number=self.ids.get(('Macro','kParamValue'))
        if number is None or not self.can_bus(mods,[factors]): return None
        first,second=mods[factors[0]],mods[factors[1]]
        source=SOURCES[first.source]
        if len(factors)>2:
            source=self.bus(mods,factors[:-1]);second=mods[factors[-1]]
            if source is None or not self.free: return None
            first=None
        macro=self.free.pop(0)
        target=Destination('Macro',macro,'kParamValue')
        written=self.slot(source,SOURCES[second.source],target,number,100.0,False,
                          _source_curve(first) if first else None,_source_curve(second))
        if written is None:
            self.free.insert(0,macro);return None
        module=self.ctx.s.set_params(f'Macro{macro}',{'kParamValue':0.0})
        module['name']='x '+' x '.join(_label(self.ctx.v,mods[j].source) for j in factors)
        self.buses[key]=MACRO_SOURCE+macro
        return self.buses[key]


def _label(v,source: str) -> str:
    hit=re.fullmatch(r'macro_control_(\d)',source)
    if hit: return v.doc.get(f'macro{hit.group(1)}') or f'MACRO {hit.group(1)}'
    return source.replace('_',' ').upper()


def _plan(v,mods,terms: list,moved: set[str],writer: _Writer,report,prefix: str,
          *, provenance: list | None=None, omitted: set | None=None,
          destination: str | None=None) -> list:
    """Turn a slot's amount terms into (coefficient, aux source ID, aux curve) triples.

    A bipolar scaling source s contributes (s - 0.5), which Serum's unipolar aux cannot do, so it
    is split into an aux part and a constant part. Terms scaled by two or more sources go through
    a spare macro; when none is left, stored controls are folded in at their current position.
    """
    flat=[]
    for coefficient,factors in terms:
        parts=[(coefficient,())]
        for j in factors:
            split=[]
            for c,f in parts:
                split.append((c,f+(j,)))
                if mods[j].bipolar: split.append((-0.5*c,f))
            parts=split
        flat+=parts
    merged: dict[tuple,float]={}
    for coefficient,factors in flat:
        key=tuple(sorted(factors,key=lambda j:(mods[j].source,mods[j].power,mods[j].points,j)))
        merged[key]=merged.get(key,0.0)+coefficient
    fixed_by_group={key:set() for key in merged}
    details={'destination':destination} if destination is not None else {}
    groups=[factors for factors,coefficient in merged.items() if abs(coefficient)>=1e-9 and len(factors)>1
            and all(mods[j].source in SOURCES and not mods[j].stereo for j in factors)]
    if groups and not writer.can_bus(mods,groups):
        reduced={};fixed_reduced={};folded=False;fixed_sources=set()
        for factors,coefficient in merged.items():
            moving=[];fixed=set()
            for j in factors:
                m=mods[j]
                if m.source in STATIC_SOURCES and m.source not in moved and not m.stereo:
                    # Bipolar constants were split above, so read the unipolar shaped value.
                    coefficient*=m.shaped(v.get(m.source));folded=True
                    fixed.add(j);fixed_sources.add(m.source)
                else: moving.append(j)
            key=tuple(moving)
            reduced[key]=reduced.get(key,0.0)+coefficient
            fixed_reduced.setdefault(key,set()).update(fixed)
        merged=reduced
        fixed_by_group=fixed_reduced
        if folded:
            report.note('approximation',prefix,'Stored controls scaling this modulation amount were folded in to preserve the routing within the available macro and slot limits; changing those controls is not represented.',
                        user_impact='control_fixed',sources=sorted(fixed_sources),**details)
    out=[]
    def emit(coefficient,source,curve,moving,fixed=()):
        out.append((coefficient,source,curve))
        if provenance is not None:provenance.append((set(moving),set(fixed)))
    for factors,coefficient in merged.items():
        if abs(coefficient)<1e-9: continue
        unknown=[mods[j].source for j in factors if mods[j].source not in SOURCES or mods[j].stereo]
        if unknown:
            if omitted is not None:omitted.update(factors)
            report.note('unsupported',prefix,'A source scaling this modulation amount has no verified target; that part is omitted.',
                        sources=unknown,**details);continue
        fixed=fixed_by_group[factors]
        if not factors:
            emit(coefficient,0,None,(),fixed);continue
        if len(factors)==1:
            m=mods[factors[0]]
            emit(coefficient,SOURCES[m.source],_source_curve(m),factors,fixed);continue
        bus=writer.bus(mods,factors)
        if bus is not None:
            emit(coefficient,bus,None,factors,fixed);continue
        moving=[j for j in factors if mods[j].source not in STATIC_SOURCES or mods[j].source in moved]
        fixed=fixed|(set(factors)-set(moving))
        for j in factors:
            if j not in moving: coefficient*=mods[j].shaped(v.get(mods[j].source))
        if abs(coefficient)<1e-9: continue
        bus=writer.bus(mods,tuple(moving)) if len(moving)>1 else None
        if bus is not None:
            report.note('approximation',prefix,'A stored control scaling this modulation amount was folded in at its current position; changing that control is not represented.',
                        user_impact='control_fixed',sources=sorted({mods[j].source for j in fixed}),**details)
            emit(coefficient,bus,None,moving,fixed);continue
        if len(moving)>1:
            if omitted is not None:omitted.update(factors)
            report.note('unsupported',prefix,'This modulation amount is scaled by several moving sources and no spare macro is left; that part is omitted.',**details)
            continue
        report.note('approximation',prefix,'A macro scaling this modulation amount was folded in at its stored position (no spare macro left).',
                    user_impact='control_fixed',sources=sorted({mods[j].source for j in fixed}),**details)
        m=mods[moving[0]] if moving else None
        emit(coefficient,SOURCES[m.source] if m else 0,_source_curve(m) if m else None,moving,fixed)
    return out


def _order_phase_buses(writer: _Writer) -> None:
    """Update LFO phase before products reading that LFO, retaining stable slot order."""
    state=writer.ctx.s.state
    slots=[state[f'ModSlot{i}'] for i in range(writer.next)]
    phase={};macros={};products=[]
    for i,slot in enumerate(slots):
        module,parameter=slot.get('destModuleTypeString'),slot.get('destModuleParamName')
        index=slot.get('destModuleID')
        if module=='LFO' and parameter=='kParamPhase': phase.setdefault(index+6,[]).append(i)
        if module=='Macro' and parameter=='kParamValue' and index in BUS_MACROS:
            macros.setdefault(index+MACRO_SOURCE,[]).append(i);products.append(i)
    edges=[set() for _ in slots];indegree=[0]*len(slots);phase_edges=False
    def before(a,b):
        if b not in edges[a]: edges[a].add(b);indegree[b]+=1
    for i in products:
        for source in slots[i]['source']:
            for p in phase.get(source,()): before(p,i);phase_edges=True
    if not phase_edges: return
    # Moving a product must also keep its consumers after it. Macro/phase feedback cycles
    # have no verified evaluation order and retain their original layout.
    for i,slot in enumerate(slots):
        for source in slot['source']:
            for producer in macros.get(source,()): before(producer,i)
    ready=[i for i,n in enumerate(indegree) if not n];heapq.heapify(ready);order=[]
    while ready:
        i=heapq.heappop(ready);order.append(i)
        for j in edges[i]:
            indegree[j]-=1
            if not indegree[j]: heapq.heappush(ready,j)
    if len(order)!=len(slots):
        writer.ctx.r.note('approximation','modulations_phase_order','A feedback cycle between an LFO phase and a helper macro retains its original routing order; repeated-note behavior may differ.')
        return
    if order==list(range(len(slots))): return
    renumber={old:new for new,old in enumerate(order)}
    for new,old in enumerate(order): state[f'ModSlot{new}']=slots[old]
    updated=set()
    for record in writer.ctx.r.modulations:
        if 'serum_slot' in record: record['serum_slot']=renumber[record['serum_slot']]
        if 'serum_slots' in record: record['serum_slots']=[renumber[i] for i in record['serum_slots']]
        extra=record.get('supplementary_destinations')
        if extra is not None and id(extra) not in updated:
            for item in extra: item['slot']=renumber[item['slot']]
            updated.add(id(extra))
        atomic=record.get('atomic_law_destinations')
        if atomic is not None and id(atomic) not in updated:
            for item in atomic: item['slot']=renumber[item['slot']]
            updated.add(id(atomic))
        bus=record.get('complete_control_bus')
        if bus is not None and id(bus) not in updated:
            for key in ('input_slot','output_slot'): bus[key]=renumber[bus[key]]
            updated.add(id(bus))


def _adaptive_law_points(function, m: Modulation, policy: tuple[float,int]) -> list:
    """Refine a law's sampled straight-line error, including the source remap's own knots.

    Quarter and midpoint probes estimate the error; they do not bound every source value.
    The same points determine signed pieces and their extrema. If the budget cannot carry
    the response, the caller omits the complete group before allocating any slots.
    """
    tolerance,limit=policy
    if not math.isfinite(tolerance) or tolerance<=0 or limit<33:
        raise ValueError('Invalid response curve policy')
    seeds={i/32 for i in range(33)}
    # Resolve square-root endpoint bends without depending on the parameter's response law.
    seeds.update(x for j in range(6,21) for x in (2.0**-j,1.0-2.0**-j))
    inverse=SOURCE_LAW_INVERSES.get(m.source,lambda x:x)
    seeds.update(inverse(max(0.0,min(1.0,x))) for x,_ in m.points)
    if len(seeds)>limit: raise ValueError('Response curve exceeds its point budget')
    cache={}
    def value(x):
        if x not in cache:cache[x]=function(x)
        return cache[x]
    points={x:value(x) for x in seeds};pending=[]
    def interval(a,b):
        ya,yb=points[a],points[b]
        probes=[(a+(b-a)*t,t) for t in (.25,.5,.75)]
        error,x,y=max((abs(value(x)-(ya+(yb-ya)*t)),x,value(x)) for x,t in probes)
        heapq.heappush(pending,(-error,a,b,x,y))
    ordered=sorted(points)
    for a,b in zip(ordered,ordered[1:]):interval(a,b)
    while pending:
        negative_error,a,b,x,y=heapq.heappop(pending)
        if -negative_error<=tolerance:break
        if len(points)>=limit or x<=a or x>=b:
            raise ValueError('Response curve exceeds its point budget')
        points[x]=y;interval(a,x);interval(x,b)
    return sorted(points.items())


def _law_slots(v,m: Modulation,dest: Destination,coefficient: float) -> list:
    """Unipolar (amount %, curve) pairs moving a control with its own scale the way Vital does.

    Vital adds coefficient * range * source to its parameter; Serum adds amount * curve(source)
    to the position of its control. The curve carries the difference between the two scales, and
    Vital's clamping at the ends of the range.
    """
    lo,hi=RANGES.get(m.target,(0.0,1.0));base=v.get(m.target);start=dest.law(base)      # controls without an entry run 0..1
    source=SOURCE_LAWS.get(m.source,lambda s: s)
    def moved(s: float) -> float:
        return dest.law(max(lo,min(hi,base+coefficient*(hi-lo)*m.value(source(s)))))-start
    points=_adaptive_law_points(moved,m,dest.curve_policy) if dest.curve_policy is not None else None
    values=[y for _,y in points] if points is not None else [moved(i/32) for i in range(33)]
    out=[]
    for extreme in (min(0.0,min(values)),max(0.0,max(values))):
        if abs(extreme)<1e-6: continue
        sign=1.0 if extreme>0 else -1.0
        curve=(remap_curve([(x,max(0.0,sign*y)/abs(extreme)) for x,y in points]) if points is not None else
               sampled_curve(lambda s,e=extreme,k=sign: max(0.0,k*moved(s))/abs(e)))
        out.append((100.0*extreme,curve))
    return out


def _reporting_update(report, record: dict, update: dict) -> None:
    # Conversion guards still inspect the established records. Apply notice-only
    # corrections after every state-changing feature has finished.
    if not hasattr(report,'_modulation_reporting_updates'):
        report._modulation_reporting_updates=[]
    report._modulation_reporting_updates.append((record['vital_slot'],record.get('aux_for_slot'),update))


def finalize_modulation_reporting(report) -> None:
    """Describe retained controls without changing intermediate conversion decisions.

    Native references identify the converted consumer group, including its helper
    pieces; they do not assert that each source factor appears in every slot.
    """
    for slot,parent,update in getattr(report,'_modulation_reporting_updates',()):
        for record in report.modulations:
            if record.get('vital_slot')!=slot or record.get('aux_for_slot')!=parent:continue
            if update.get('status') in ('fixed','omitted'):
                for key in ('serum_slot','serum_slots','serum_amount','serum_parameter_id',
                            'supplementary_destinations','atomic_law_destinations'):
                    record.pop(key,None)
            record.update(update)
    if hasattr(report,'_modulation_reporting_updates'):
        del report._modulation_reporting_updates


def convert_modulations(ctx,ids: dict[tuple[str,str],int]) -> None:
    v,s,r=ctx.v,ctx.s,ctx.r
    mods=active_modulations(v);into=amount_modulators(mods)
    moved={m.target for m in mods.values()}
    writer=_Writer(ctx,ids)
    supplementary=[]
    complete=[]
    atomic=[]
    trajectories={m.slot:(plan,dest) for plan,dest in getattr(ctx,'table_trajectories',()) for m in plan.routes}
    assigned=set()
    for source,dest,amount,bipolar,*curve in getattr(ctx,'fixed_routings',()):
        writer.slot(SOURCES[source],0,dest,ids.get((dest.module,dest.parameter),0),amount,bipolar,*curve)
    for i,m in mods.items():
        prefix=f'modulation_{i}_'
        if i in v.folded or _AMOUNT.fullmatch(m.target): continue
        joint=trajectories.get(i)
        if joint is not None:
            plan,dest=joint
            if plan.prefix in assigned: continue
            assigned.add(plan.prefix)
            number=ids.get((dest.module,dest.parameter))
            if number is None or writer.next>=MAX_SLOTS or getattr(ctx,'source_gain',{}).get(plan.source,1.0)!=1.0:
                ctx.failed_trajectories.add(plan.prefix)
                continue
            raw=Modulation(i,plan.source,m.target,1.0,False,0.0,False)
            depth=100.0*dest.scale
            try:
                source_curve=_trajectory_source_curve(raw)
            except ValueError:
                ctx.failed_trajectories.add(plan.prefix)
                continue
            written=writer.slot(SOURCES[plan.source],0,dest,number,depth,False,source_curve)
            if written is None:
                ctx.failed_trajectories.add(plan.prefix)
                continue
            for route in plan.routes:
                v.ignore_prefix(f'modulation_{route.slot}_')
                r.modulations.append(dict(vital_slot=route.slot,source=route.source,destination=route.target,
                                          amount=route.amount,status='mapped',serum_slot=written,serum_slots=[written],
                                          serum_module=dest.module,serum_module_id=dest.index,serum_parameter=dest.parameter,
                                          serum_parameter_id=number,serum_amount=depth,joint_trajectory=plan.prefix))
            continue
        terms=_terms(mods,into,i,r)
        chain=sorted({j for _,factors in terms for j in factors})
        atomic_plan=getattr(ctx,'atomic_law_destinations',{}).get(i)
        if not terms:
            if atomic_plan is not None:
                atomic_plan[2]()
                r.modulations.append(dict(vital_slot=i,source=m.source,destination=m.target,amount=m.amount,status='omitted'))
                r.note('approximation',prefix+'atomic_law','The complete response has no active amount; the stored control is retained.',
                       user_impact='control_partial',source=m.source,destination=m.target)
            v.ignore_prefix(prefix);continue
        planned=dict(vital_slot=i,source=m.source,destination=m.target,amount=m.amount,status='omitted')
        r.modulations.append(planned)
        records=[planned]
        for j in chain:
            records.append(dict(vital_slot=j,source=mods[j].source,destination=mods[j].target,amount=mods[j].amount,
                                status='omitted',aux_for_slot=i))
            r.modulations.append(records[-1]);v.ignore_prefix(f'modulation_{j}_')
        v.ignore_prefix(prefix)
        # Vital 1.6 can slew a routing: a full swing then takes about 2^(value - 1) seconds, up and
        # down separately (-10 is off). Serum's slots react at once.
        if (any(v.settings.get(prefix+key,RAMP_OFF)>RAMP_OFF+1e-6 for key in ('ramp_up','ramp_down'))
                and not any(note['feature']==prefix+'ramp' and note.get('destination')==m.target for note in r.notes)):
            r.note('unsupported',prefix+'ramp','Modulation ramp times are not converted; the routing follows its source at once.')
        if atomic_plan is not None:
            atomic.append((i,m,atomic_plan,records,chain))
            continue
        control_bus=getattr(ctx,'complete_control_buses',{}).get(i)
        if control_bus is not None:
            complete.append((i,m,control_bus,records))
            continue
        if m.source not in SOURCES:
            r.note('unsupported',prefix,'Modulation source has no verified target.',source=m.source,destination=m.target);continue
        dest=ctx.destinations.get(m.target)
        if dest is None:
            r.note('unsupported',prefix,'Modulation destination is not implemented or its module was omitted.',destination=m.target);continue
        for record in records:
            record.update(serum_module=dest.module,serum_module_id=dest.index,serum_parameter=dest.parameter)
        number=ids.get((dest.module,dest.parameter))
        if number is None:
            r.note('unsupported',prefix,'Numeric Serum destination ID is unknown; use --learn-from with a Serum-created routing to this parameter.',
                   module=dest.module,parameter=dest.parameter,destination=m.target);continue
        if dest.nonlinear and dest.law is None:
            r.note('unsupported',prefix,'This control responds on a different scale in Serum and no conversion is known; routing omitted.',destination=m.target);continue
        if m.stereo:
            r.note('unsupported',prefix,'Stereo modulation is not supported; routing omitted.',destination=m.target);continue
        slots=[];fixed_chain=set()
        gains=getattr(ctx,'source_gain',{})
        if dest.law is not None and m.target in getattr(ctx,'static_aux_law_targets',()):
            effective=static_amount(v,mods,into,i,moved)
            if effective is not None and into.get(i):
                terms=[(effective[0],())]
                fixed_chain=set(chain)
                r.note('approximation',prefix+'static_aux','A fixed auxiliary position was folded into this response curve; changing that auxiliary control is not represented.',
                       user_impact='control_fixed',sources=sorted({mods[j].source for j in chain}),destination=m.target)
        terms=[(coefficient*math.prod(gains.get(mods[j].source,1.0) for j in factors)*gains.get(m.source,1.0),factors)
               for coefficient,factors in terms]
        provenance=[];omitted=set();retained_controls=set();fixed_controls=set();failed_controls=set()
        pieces_plan=_plan(v,mods,terms,moved,writer,r,prefix,provenance=provenance,omitted=omitted,destination=m.target)
        for (coefficient,aux,aux_curve),(controls,fixed) in zip(pieces_plan,provenance):
            fixed=fixed|fixed_chain
            if dest.law is not None:
                if aux: r.note('approximation',prefix,'The amount of this modulation is scaled by another source; the response curve is exact only at full scale.',
                               user_impact='control_partial',destination=m.target)
                pieces=[(amount,False,curve) for amount,curve in _law_slots(v,m,dest,coefficient)]
            else:
                percent=coefficient*100*dest.scale
                # Serum limits the sum of routings to the control's range, so more than 100 %
                # only helps where an aux source scales the routing down again.
                count=min(MAX_SPLIT,max(1,math.ceil(abs(percent)/100-1e-9))) if aux else 1
                if abs(percent)>100*count+1e-6:
                    r.note('clipped',prefix+'target_amount','Value clamped to the supported target range.',original=percent,converted=math.copysign(100.0*count,percent))
                    percent=math.copysign(100.0*count,percent)
                pieces=[(percent/count,m.bipolar,_source_curve(m))]*count
            for amount,bipolar,curve in pieces:
                written=writer.slot(SOURCES[m.source],aux,dest,number,amount,bipolar,curve,aux_curve)
                if written is not None:
                    slots.append((written,amount));fixed_controls.update(fixed)
                    retained_controls.update(controls)
                else:failed_controls.update(controls|fixed)
        if slots and m.target=='portamento_time' and m.source=='velocity':
            r.note('approximation',prefix,'Serum slides the velocity source along with the pitch, so the glide time follows the new note alone only when the previous note had the same velocity.')
        if slots:
            for record in records:
                record.update(status='mapped',serum_slot=slots[0][0],serum_slots=[n for n,_ in slots],
                              serum_amount=sum(a for _,a in slots),serum_parameter_id=number)
            if omitted or failed_controls or writer.full:
                _reporting_update(r,planned,dict(control_status='partial'))
            for record in records[1:]:
                j=record['vital_slot']
                if j in retained_controls:
                    if j in omitted or j in failed_controls or j in fixed_controls:
                        _reporting_update(r,record,dict(control_status='partial'))
                elif j in fixed_controls:
                    # The stored contribution exists, but this control no longer operates it.
                    _reporting_update(r,record,dict(status='fixed',control_status='fixed'))
                else:_reporting_update(r,record,dict(status='omitted'))
            extra=getattr(ctx,'supplementary_destinations',{}).get(m.target)
            if extra and all(not factors for _,factors in terms):
                supplementary.append((i,m,extra,sum(coefficient for coefficient,_ in terms),records))
    for i,m,destinations,coefficient,records in supplementary:
        pieces=[];known=True
        for dest in destinations:
            number=ids.get((dest.module,dest.parameter))
            if number is None or dest.law is None:
                known=False;break
            pieces.extend((dest,number,amount,curve) for amount,curve in _law_slots(v,m,dest,coefficient))
        if not known or writer.next+len(pieces)>MAX_SLOTS:
            r.note('approximation',f'modulation_{i}_makeup','Additional correlated destinations could not be written together; the fixed output correction is retained.',
                   user_impact='control_partial',source=m.source,destination=m.target)
            continue
        written=[]
        for dest,number,amount,curve in pieces:
            slot=writer.slot(SOURCES[m.source],0,dest,number,amount,False,curve)
            written.append(dict(slot=slot,module=dest.module,module_id=dest.index,parameter=dest.parameter,parameter_id=number,amount=amount))
        for record in records:record['supplementary_destinations']=written
    for i,m,plan,records in complete:
        base,source,aux,main_curve,aux_curve,dest,amount,output_curve,on_written,on_omitted=plan
        source_id,aux_id=SOURCES.get(source),SOURCES.get(aux)
        number=ids.get((dest.module,dest.parameter));macro_number=ids.get(('Macro','kParamValue'))
        if (not writer.free or writer.next+2>MAX_SLOTS or source_id is None or aux_id is None
                or number is None or macro_number is None or not math.isfinite(base) or not 0<=base<=1
                or not math.isfinite(amount) or abs(amount)>100):
            on_omitted()
            r.note('approximation',f'modulation_{i}_control_bus','The complete control needs a free macro and two modulation slots; the stored control is retained.',
                   user_impact='control_partial',source=m.source,destination=m.target)
            continue
        macro=writer.free.pop(0)
        target=Destination('Macro',macro,'kParamValue')
        input_slot=writer.slot(source_id,aux_id,target,macro_number,100.0,False,main_curve,aux_curve)
        output_slot=writer.slot(MACRO_SOURCE+macro,0,dest,number,amount,False,output_curve)
        ctx.s.set_params(f'Macro{macro}',{'kParamValue':100.0*base})
        on_written()
        for record in records:
            record.update(status='mapped',serum_module=dest.module,serum_module_id=dest.index,
                          serum_parameter=dest.parameter,serum_parameter_id=number,
                          serum_slot=output_slot,serum_slots=[output_slot],serum_amount=amount,
                          complete_control_bus=dict(macro=macro,input_slot=input_slot,output_slot=output_slot))
    for i,m,plan,records,chain in atomic:
        destinations,on_written,on_omitted=plan
        source_id=SOURCES.get(m.source)
        coefficient=m.amount
        fixed_aux=False
        valid=(bool(destinations) and source_id is not None and not m.stereo
               and (not m.bipolar or m.target in ('distortion_mix','delay_dry_wet'))
               and getattr(ctx,'source_gain',{}).get(m.source,1.0)==1.0)
        if m.target in ('distortion_mix','delay_dry_wet'):
            valid=valid and mix_route_supported(v,mods,into,i,r)
        if into.get(i):
            effective=static_amount(v,mods,into,i,moved,ignore_zero_leaves=m.target in ('distortion_mix','delay_dry_wet'))
            valid=valid and m.target in getattr(ctx,'static_aux_law_targets',()) and effective is not None
            if effective is not None:
                coefficient=effective[0];fixed_aux=True
        valid=valid and math.isfinite(coefficient) and all(
            getattr(ctx,'source_gain',{}).get(mods[j].source,1.0)==1.0 for j in chain)
        pieces=[]
        if valid:
            try:
                for dest in destinations:
                    number=ids.get((dest.module,dest.parameter))
                    if number is None or dest.law is None:
                        valid=False;break
                    def checked(raw,law=dest.law):
                        value=law(raw)
                        if not math.isfinite(value) or not 0.0<=value<=1.0:
                            raise ValueError('Response position is outside its native range')
                        return value
                    bounded=Destination(dest.module,dest.index,dest.parameter,dest.scale,dest.nonlinear,checked,dest.curve_policy)
                    pieces.extend((dest,number,amount,curve) for amount,curve in _law_slots(v,m,bounded,coefficient))
                valid=valid and bool(pieces) and all(math.isfinite(amount) and abs(amount)<=100.0 for _,_,amount,_ in pieces)
            except (ArithmeticError,TypeError,ValueError):
                valid=False
        if not valid or writer.next+len(pieces)>MAX_SLOTS:
            on_omitted()
            r.note('approximation',f'modulation_{i}_atomic_law',
                   'The complete response needs verified unit sources, response ranges, destination IDs and enough modulation slots together; the stored control is retained.',
                   user_impact='control_partial',source=m.source,destination=m.target)
            continue
        written=[]
        for dest,number,amount,curve in pieces:
            slot=writer.slot(source_id,0,dest,number,amount,False,curve)
            written.append(dict(slot=slot,module=dest.module,module_id=dest.index,parameter=dest.parameter,parameter_id=number,amount=amount))
        on_written()
        if fixed_aux:
            r.note('approximation',f'modulation_{i}_static_aux',
                   'A fixed auxiliary position was folded into this response curve; changing that auxiliary control is not represented.',
                   user_impact='control_fixed',sources=sorted({mods[j].source for j in chain}),destination=m.target)
        first=written[0]
        for record in records:
            record.update(status='mapped',serum_module=first['module'],serum_module_id=first['module_id'],
                          serum_parameter=first['parameter'],serum_parameter_id=first['parameter_id'],
                          serum_slot=first['slot'],serum_slots=[item['slot'] for item in written],
                          serum_amount=sum(item['amount'] for item in written if
                                           (item['module'],item['module_id'],item['parameter'])==
                                           (first['module'],first['module_id'],first['parameter'])),
                          atomic_law_destinations=written)
            if fixed_aux and record.get('aux_for_slot')==i:
                _reporting_update(r,record,dict(status='fixed',control_status='fixed'))
    for source,dest,amount,bipolar,on_written,on_omitted in getattr(ctx,'optional_routings',()):
        source_id=SOURCES.get(source)
        number=ids.get((dest.module,dest.parameter))
        if writer.next<MAX_SLOTS and source_id is not None and number is not None:
            writer.slot(source_id,0,dest,number,amount,bipolar)
            on_written()
        else:
            on_omitted()
    _order_phase_buses(writer)
    if writer.next:
        r.mapped.append(f'modulations:{writer.next}')
        r.note('approximation','modulations','Ranges are converted in parameter space; DSP/source curves and clipping can differ between the synths.')
