from __future__ import annotations
from dataclasses import dataclass, field
import copy
import re
from . import __version__

def effective_kind(note: dict):
    """The kind a note counts as. An approximation that leaves a setting unconverted counts as
    unsupported; its stored kind is kept because conversion decisions read it."""
    kind=note.get('kind')
    return 'unsupported' if kind=='approximation' and note.get('user_impact')=='unsupported' else kind


@dataclass
class Report:
    source: str
    name: str = ''
    notes: list[dict] = field(default_factory=list)
    mapped: list[str] = field(default_factory=list)
    assets: list[dict] = field(default_factory=list)
    modulations: list[dict] = field(default_factory=list)

    def note(self, kind: str, feature: str, message: str, **details) -> None:
        item = dict(kind=kind,feature=feature,message=message,**details)
        if item not in self.notes: self.notes.append(item)

    @property
    def lossy(self) -> bool:
        return any(effective_kind(n) in ('unsupported','clipped') for n in self.notes)

    def user_notices(self) -> list[dict]:
        """Expose actionable constraints without changing conversion decisions."""
        categories={'unsupported','range_limited','control_fixed','control_partial','baked','alternative'}
        details=('source','sources','destination','controls','original','converted','target_type')
        notices=[]
        for note in self.notes:
            kind=note.get('kind')
            category={'unsupported':'unsupported','clipped':'range_limited'}.get(kind)
            if category is None and kind=='approximation':
                category=note.get('user_impact')
            if category not in categories:continue
            feature=note.get('feature','')
            item=dict(kind={'unsupported':'unsupported','range_limited':'clipped'}.get(category,'approximation'),
                      category=category,feature=feature,message=note.get('message',''))
            for key in details:
                if key in note and note[key] is not None:item[key]=copy.deepcopy(note[key])
            match=re.match(r'^modulation_(\d+)(?:_|$)',feature) if isinstance(feature,str) else None
            if match:
                slot=int(match[1])
                records=[record for record in self.modulations if isinstance(record,dict)
                         and type(record.get('vital_slot')) is int and record['vital_slot']==slot]
                if 'source' not in item and 'sources' not in item:
                    sources=sorted({record['source'] for record in records if isinstance(record.get('source'),str)})
                    if len(sources)==1:item['source']=sources[0]
                    elif sources:item['sources']=sources
                if 'destination' not in item:
                    destinations={record['destination'] for record in records if isinstance(record.get('destination'),str)}
                    if len(destinations)==1:item['destination']=next(iter(destinations))
            if item not in notices:notices.append(item)
        return notices

    def as_dict(self) -> dict:
        return dict(converter='vital2serum',version=__version__,target='Serum 2',
                    source=self.source,preset_name=self.name,
                    mapped=self.mapped,assets=self.assets,modulations=self.modulations,
                    notes=self.notes,user_notices=self.user_notices(),
                    counts={k:sum(effective_kind(n)==k for n in self.notes)
                            for k in ('approximation','unsupported','clipped','info')})
