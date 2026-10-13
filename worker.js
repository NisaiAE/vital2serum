const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';
const MODULES = ['__init__', 'assets', 'codec', 'compressor', 'curves', 'defaults', 'distortion_model', 'effects', 'errors', 'filter_structure', 'fm',
  'mapping', 'modulation', 'modulation_fx', 'phase_modulation', 'post_master', 'ranges', 'report', 'ring_modulation', 'sampler', 'serum', 'trajectory', 'spectral', 'unison', 'vital', 'voice_gain', 'wavetable'];
const DATA = ['data/Init-2.0.24.SerumPreset'];

const GLUE = `
import json, sys, zlib
sys.path.insert(0, '/pkg')
import vital2serum
from vital2serum.errors import ConversionError
from vital2serum.mapping import convert
from vital2serum.serum import SerumPreset
from vital2serum.vital import VitalPreset

TEMPLATE = SerumPreset.from_bytes(open('/pkg/vital2serum/data/Init-2.0.24.SerumPreset', 'rb').read())
LAST = b''

def run(path):
    global LAST
    LAST = b''
    try:
        result = convert(VitalPreset.read(path), TEMPLATE)
        data = result.preset.to_bytes()
        if SerumPreset.from_bytes(data).state != result.preset.state:
            raise ConversionError('Serialized Serum state changed on read-back')
        notes = [[n['kind'], n['feature'] + ': ' + n['message']] for n in result.report.notes
                 if n['kind'] in ('unsupported', 'clipped')]
        user_notices = result.report.user_notices()
        LAST = data
        return json.dumps({'ok': True, 'notes': notes, 'user_notices': user_notices, 'crc': zlib.crc32(data)})
    except (ConversionError, OSError, OverflowError, TypeError, ValueError, KeyError) as exc:
        return json.dumps({'ok': False, 'error': str(exc) or type(exc).__name__})

def take():
    global LAST
    data, LAST = LAST, b''
    return data
`;

importScripts(PYODIDE + 'pyodide.js');

const ready = (async () => {
  const py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(['numpy', 'zstandard']);
  py.FS.mkdirTree('/pkg/vital2serum/data');
  py.FS.mkdirTree('/in');
  await Promise.all([...MODULES.map((m) => m + '.py'), ...DATA].map(async (file) => {
    const response = await fetch('vital2serum/' + file, { cache: 'no-cache' });
    if (!response.ok) throw new Error('vital2serum/' + file + ': ' + response.status);
    py.FS.writeFile('/pkg/vital2serum/' + file, new Uint8Array(await response.arrayBuffer()));
  }));
  py.runPython(GLUE);
  postMessage({ type: 'ready', version: py.runPython('vital2serum.__version__') });
  return py;
})();
ready.catch((error) => postMessage({ type: 'failed', error: String(error && error.message || error) }));

function bytes(proxy) {
  const data = proxy.toJs();
  proxy.destroy();
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
}

onmessage = async (event) => {
  const message = event.data;
  let py;
  try {
    py = await ready;
  } catch (error) {
    return;
  }
  if (message.type === 'convert') {
    const dir = '/in/' + message.key;
    const path = dir + '/' + message.name.replace(/[\/\\\0]/g, '_');
    py.FS.mkdirTree(dir);
    py.FS.writeFile(path, new Uint8Array(message.buffer));
    postMessage({ type: 'start', key: message.key });
    let result;
    try {
      result = JSON.parse(py.globals.get('run')(path));
    } catch (error) {
      result = { ok: false, error: String(error && error.message || error).split('\n').slice(-2)[0] };
    }
    py.FS.unlink(path);
    py.FS.rmdir(dir);
    if (result.ok) {
      const buffer = bytes(py.globals.get('take')());
      postMessage({ type: 'done', key: message.key, ...result, buffer }, [buffer]);
    } else postMessage({ type: 'done', key: message.key, ...result });
  }
};
