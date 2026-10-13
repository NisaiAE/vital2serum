"""CLI with explicit loss reports, dry-run, batch paths and no default overwrite."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from . import __version__
from .errors import ConversionError
from .mapping import Result,convert
from .modulation import learn_ids
from .serum import SerumPreset
from .vital import VitalPreset


def json_text(obj) -> str:
    return json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def scan(path: Path,extension: str,recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower()!=extension.lower():
            raise ConversionError(f'Expected {extension}: {path}')
        return [path]
    if not path.is_dir(): raise ConversionError(f'Input not found: {path}')
    results=sorted(p for p in (path.rglob('*') if recursive else path.iterdir())
                   if p.is_file() and not p.is_symlink() and p.suffix.lower()==extension.lower())
    if not results: raise ConversionError(f'No {extension} files in {path}')
    return results


def safe_name(text: str) -> str:
    out=re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_',text).rstrip(' .')[:120] or 'Preset'
    if out.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'{a}{i}' for a in ('COM','LPT') for i in range(1,10)]}:
        out='_'+out
    return out


def output_relative(source: Path,input_path: Path) -> Path:
    relative=source.relative_to(input_path) if input_path.is_dir() else Path(source.name)
    parts=[safe_name(p) for p in relative.parent.parts]
    return Path(*parts,safe_name(relative.stem)+'.SerumPreset')


def secure_path(root: Path,relative: Path) -> Path:
    root=root.resolve()
    path=root/relative
    if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root):
        raise ConversionError(f'Output path escapes root: {relative}')
    cursor=path
    while cursor!=root:
        if cursor.is_symlink(): raise ConversionError(f'Refusing symlink output: {cursor}')
        cursor=cursor.parent
    return path


def atomic_write(path: Path,data: bytes,overwrite: bool) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.vital2serum-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        if overwrite:
            os.replace(tmp,path)
        else:
            # Hard-link installation is atomic and cannot replace an existing
            # file. Supported by ordinary NTFS/APFS/ext4 user data folders.
            os.link(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_result(root: Path,relative: Path,result: Result,overwrite: bool,strict: bool) -> bool:
    blocked=strict and result.report.lossy
    report_path=relative.with_suffix('.report.json')
    planned=[(report_path,json_text(result.report.as_dict()).encode('utf-8'))]
    if not blocked:
        payload=result.preset.to_bytes()
        # Independent reparse catches header/size/hash/schema serialization errors.
        reparsed=SerumPreset.from_bytes(payload)
        if reparsed.state!=result.preset.state:
            raise ConversionError('Serialized Serum state changed on read-back')
        planned.extend((Path(a.relative_path),a.data) for a in result.assets)
        planned.append((relative,payload))
    # Preflight everything before writing the preset/report. Assets with the
    # same content may be reused, but a hash-name collision is never replaced.
    todo=[]
    for rel,data in planned:
        path=secure_path(root,rel)
        is_asset=rel.parts[0] in ('Tables','Noises')
        if path.exists():
            if not path.is_file(): raise ConversionError(f'Output is not a file: {path}')
            if is_asset:
                if path.read_bytes()!=data: raise ConversionError(f'Asset name/content collision: {path}')
                continue
            if not overwrite: raise ConversionError(f'Output exists; use --overwrite explicitly: {path}')
        todo.append((path,data,is_asset))
    # Write assets first, then report, preset last; a preset never references
    # assets that have not yet been written. This is per-file, not batch rollback.
    todo.sort(key=lambda item:(not item[2],item[0].suffix=='.SerumPreset'))
    for path,data,is_asset in todo:
        atomic_write(path,data,overwrite and not is_asset)
    return not blocked


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog='vital2serum',description='Vital 1.x -> Serum 2 preset converter')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True)
    cv=sub.add_parser('convert',help='convert a .vital file or directory')
    cv.add_argument('input',type=Path)
    cv.add_argument('-o','--out','--output',dest='output',type=Path)
    cv.add_argument('--template',type=Path,help='Init .SerumPreset saved by your Serum version (default: the bundled one from Serum 2.0.24)')
    cv.add_argument('--recursive',action='store_true')
    cv.add_argument('--dry-run',action='store_true',help='print full reports without writing anything')
    cv.add_argument('--strict',action='store_true',help='write only a report if an unsupported/clipped feature is found')
    cv.add_argument('--overwrite',action='store_true')
    cv.add_argument('--allow-placeholder',action='store_true',help='explicitly substitute a sine for unsupported/missing wavetables')
    cv.add_argument('--wav-tables',action='store_true',help='write wavetables/samples as WAV files under Tables/ and Noises/ instead of embedding them in the preset')
    cv.add_argument('--learn-from',type=Path,action='append',default=[],help='Serum preset/directory with real modulation destination IDs; repeatable')
    info=sub.add_parser('inspect',help='read and validate an output/template container')
    info.add_argument('input',type=Path)
    info.add_argument('--modules',action='store_true',help='also list module names (not embedded binary content)')
    return p


def main(argv: list[str] | None=None) -> int:
    args=parser().parse_args(argv)
    try:
        if args.command=='inspect':
            patch=SerumPreset.read(args.input)
            info={'metadata':patch.meta,'module_count':len(patch.state),'container_valid':True}
            if args.modules: info['modules']=sorted(patch.state)
            print(json_text(info),end='');return 0
        if not args.dry_run and args.output is None:
            raise ConversionError('--output is required unless --dry-run is used')
        inputs=scan(args.input,'.vital',args.recursive)
        template=SerumPreset.read(args.template) if args.template else SerumPreset.from_bytes(
            files('vital2serum').joinpath('data/Init-2.0.24.SerumPreset').read_bytes())
        learned=[]
        for path in args.learn_from: learned.extend(scan(path,'.SerumPreset',True))
        ids=learn_ids(learned)
        paths={}
        for source in inputs:
            relative=output_relative(source,args.input)
            normalized=str(relative).casefold()
            if normalized in paths:
                raise ConversionError(f'Case-folded/sanitized output collision: {source} and {paths[normalized]}')
            paths[normalized]=source
        failures=0
        for source in inputs:
            try:
                v=VitalPreset.read(source)
                result=convert(v,template,allow_placeholder=args.allow_placeholder,destination_ids=ids,
                               embed=not args.wav_tables)
                if args.dry_run:
                    print(json_text(result.report.as_dict()),end='')
                    if args.strict and result.report.lossy: failures+=1
                else:
                    rel=output_relative(source,args.input)
                    saved=write_result(args.output,rel,result,args.overwrite,args.strict)
                    print(json_text({'source':str(source),'status':'written' if saved else 'strict-blocked',
                                     'output':str(args.output/rel) if saved else None,
                                     'report':str(args.output/rel.with_suffix('.report.json')),
                                     'counts':result.report.as_dict()['counts']}),end='')
                    if not saved: failures+=1
            except (ConversionError,OSError,OverflowError,TypeError) as exc:
                failures+=1
                print(json_text({'source':str(source),'status':'error','error':str(exc)}),file=sys.stderr,end='')
        return 2 if failures else 0
    except (ConversionError,OSError,OverflowError,TypeError) as exc:
        print(f'vital2serum: {exc}',file=sys.stderr);return 2

if __name__=='__main__':
    sys.exit(main())
