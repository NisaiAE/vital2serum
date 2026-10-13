(() => {
  const MAX_FILE = 64 * 1024 * 1024;
  const MAX_FILES = 2000;
  const MAX_TOTAL = 2 * 1024 * 1024 * 1024;
  const TIMEOUT = 120000;

  const TEXT = {
    ja: {
      drop: '.vital ファイルをドロップ、またはクリックして選択',
      formats: 'Vital (.vital) → Serum 2 (.SerumPreset)　複数ファイル・フォルダー可',
      loading: '準備中…',
      failed: '読み込みに失敗しました',
      retry: 'クリックすると、もう一度読み込みます',
      clear: 'クリア',
      download: 'ダウンロード',
      downloadAll: 'まとめてダウンロード (.zip)',
      wait: '待機中',
      run: '変換中…',
      ok: '完了',
      bad: '失敗',
      save: '保存',
      back: '戻る',
      big: 'ファイルが大きすぎます（上限 64 MB）',
      timeout: '時間内に終わりませんでした',
      full: '合計サイズの上限です。ダウンロードしてからクリアしてください',
      unsupported: (n) => `未対応 ${n}`,
      clipped: (n) => `値を制限 ${n}`,
      review: (n) => `要確認 ${n}`,
      noticeLabels: {
        unsupported: '未対応', range_limited: '値を制限', control_fixed: '値を固定',
        control_partial: '一部のみ再現', baked: '波形を直接加工', alternative: '別の構成へ変換',
      },
      noticeHelp: {
        unsupported: '引き継がれていない設定や動きです。',
        range_limited: '変換先の範囲に合わせて値を制限しています。',
        control_fixed: 'マクロやホイールの保存時の位置を、値に取り込んでいます。Serum 側でそれを動かしても、ここは変化しません。',
        control_partial: '動きや連動の一部だけを引き継いでいます。',
        baked: 'Vital のワープやモーフの効果を、波形そのものに加えています。Serum 側に同じノブはありません。',
        alternative: '別の構成へ置き換えています。操作や音の違いを確認してください。',
      },
      note: '注意',
      summary: (done, total, bad) => `${done} / ${total}` + (bad ? `（失敗 ${bad}）` : ''),
      skipped: (n) => `一度に扱えるのは ${n} 件までです`,
      local: 'ファイルはブラウザ内で変換され、外部には送信されません。',
      unofficial: 'Xfer Records・Vital Audio とは関係のない非公式ツールです。変換結果は元の音色と完全には一致しません。',
      rights: '変換後のプリセットにも、元のプリセットの利用規約が適用されます。',
      other: 'EN',
    },
    en: {
      drop: 'Drop .vital files here, or click to choose',
      formats: 'Vital (.vital) → Serum 2 (.SerumPreset)   multiple files and folders',
      loading: 'Loading…',
      failed: 'Failed to load',
      retry: 'Click to try again',
      clear: 'Clear',
      download: 'Download',
      downloadAll: 'Download all (.zip)',
      wait: 'Waiting',
      run: 'Converting…',
      ok: 'Done',
      bad: 'Failed',
      save: 'Save',
      back: 'Back',
      big: 'File too large (64 MB at most)',
      timeout: 'Did not finish in time',
      full: 'Total size limit reached. Download, then clear',
      unsupported: (n) => `${n} unsupported`,
      clipped: (n) => `${n} limited`,
      review: (n) => `${n} to review`,
      noticeLabels: {
        unsupported: 'Unsupported', range_limited: 'Value limited', control_fixed: 'Value fixed',
        control_partial: 'Partly retained', baked: 'Waveform edited', alternative: 'Alternative structure',
      },
      noticeHelp: {
        unsupported: 'Some settings or movement are not carried over.',
        range_limited: 'The value is limited to the target range.',
        control_fixed: 'The saved position of the macro or wheel is folded into the value. Moving it in Serum no longer changes this.',
        control_partial: 'Only part of the control or interaction is retained. Check the movement that is not carried over.',
        baked: 'The effect of Vital’s warp or morph is applied to the waveform itself. Serum has no matching knob for it.',
        alternative: 'A different structure is used. Check its controls and sound.',
      },
      note: 'Note',
      summary: (done, total, bad) => `${done} / ${total}` + (bad ? ` (${bad} failed)` : ''),
      skipped: (n) => `Up to ${n} files at a time`,
      local: 'Files are converted in your browser and are never uploaded.',
      unofficial: 'An unofficial tool, not affiliated with Xfer Records or Vital Audio. Converted presets will not sound identical to the originals.',
      rights: 'The license terms of the original preset still apply to the converted preset.',
      other: '日本語',
    },
  };

  const $ = (id) => document.getElementById(id);
  const drop = $('drop'), input = $('files'), list = $('list'), bar = $('bar');
  const items = new Map();
  let lang = 'ja', worker = null, ready = false, failed = null, busy = null, timer = 0;
  let next = 0, total = 0, skipped = false, pending = 0, notesLoaded = false, epoch = 0;
  const queue = [];

  const LABELS = {
    ja: {
      mod_wheel: 'モジュレーションホイール', pitch_wheel: 'ピッチホイール', velocity: 'ベロシティ',
      aftertouch: 'アフタータッチ', note: 'ノート', random: 'ランダム', tuning: 'チューニング',
      sample: 'サンプル', filter_fx: 'FX フィルター', distortion: 'Distortion', chorus: 'Chorus', compressor: 'Compressor',
      eq: 'EQ', flanger: 'Flanger', phaser: 'Phaser', reverb: 'Reverb',
      volume: 'マスター音量', master_volume: 'マスター音量', levels: '音量', envelopes: 'エンベロープ',
      modulations: 'モジュレーション', filter_routing: 'フィルター経路', stereo_mode: 'ステレオモード',
      level: '音量', gain: 'ゲイン', pan: 'パン', pitch: 'ピッチ', cutoff: 'カットオフ', resonance: 'レゾナンス',
      drive: 'Drive', mix: 'Mix', dry_wet: 'Mix', blend: 'Blend', transpose: 'トランスポーズ',
      attack: 'アタック', decay: 'ディケイ', sustain: 'サステイン', release: 'リリース', hold: 'ホールド',
      delay: 'ディレイ', delay_time: '開始遅延', frequency: '周波数', tempo: 'テンポ', sync: '同期', trigger: 'トリガー',
      wave_frame: '波形位置', warp: 'ワープ', spectral_morph: 'スペクトルモーフ', trajectory: '音色の動き',
      distortion_amount: 'ワープ量', distortion_type: 'ワープ方式', distortion_phase: 'ワープ位相',
      distortion_spread: 'ワープのボイス間差', spectral_morph_amount: 'スペクトルモーフ量',
      spectral_morph_phase: 'スペクトルモーフ位相', spectral_morph_spread: 'モーフのボイス間差',
      detune_range: 'デチューン幅', detune_power: 'デチューン配分', stack_style: 'ユニゾン構成',
      unison_voices: 'ユニゾン数', voices: 'ボイス数', feedback: 'Feedback', size: 'サイズ', tone: '音色',
      filter: 'フィルター', destination: '出力先', tune: 'チューン', keytrack: 'キートラック',
      keytrack_transpose: 'キートラックのトランスポーズ', transpose_quantize: 'トランスポーズのスナップ',
      stereo: 'ステレオ', stereo_spread: 'ステレオ幅', frame_spread: '波形位置のボイス間差',
      spectral_unison: 'スペクトルユニゾン', smooth_interpolation: '波形の補間', smooth_time: 'スムーズ',
      phase_offset: '位相差', center: '中心周波数', freeze: 'Freeze', delay_1: 'ディレイ 1', delay_2: 'ディレイ 2',
      makeup: '音量補正', blend_modulation: 'Blend の変調', drive_modulation: 'Drive の変調',
      rt60: '残響時間', decay_time: '残響時間', band_mode: '中域の種類', band_gain: '中域のゲイン',
      band_cutoff: '中域の周波数', low_cutoff: '低域の周波数', high_cutoff: '高域の周波数',
      low_lower_threshold: '低域の下側しきい値', low_upper_threshold: '低域の上側しきい値',
      band_lower_threshold: '中域の下側しきい値', band_upper_threshold: '中域の上側しきい値',
      high_lower_threshold: '高域の下側しきい値', high_upper_threshold: '高域の上側しきい値',
      pitch_bend_range: 'ピッチベンド幅', pitch_bend_up: 'ピッチベンド幅（上）', pitch_bend_down: 'ピッチベンド幅（下）',
      portamento_time: 'ポルタメント時間', portamento_slope: 'ポルタメントのカーブ', portamento_scale: 'ポルタメントのスケール',
      voice_transpose: '全体のトランスポーズ', voice_tune: '全体のチューン', voice_priority: 'ボイスの優先順位',
      stereo_routing: 'ステレオルーティング', polyphony: 'ポリフォニー', legato: 'レガート',
      formant_x: 'フォルマント X', formant_y: 'フォルマント Y', formant_transpose: 'フォルマントのトランスポーズ',
      formant_resonance: 'フォルマントのレゾナンス', formant_spread: 'フォルマントの広がり', blend_transpose: 'Blend のトランスポーズ',
    },
    en: {
      mod_wheel: 'Mod wheel', pitch_wheel: 'Pitch wheel', velocity: 'Velocity', aftertouch: 'Aftertouch',
      note: 'Note', random: 'Random', tuning: 'Tuning', volume: 'Master volume', master_volume: 'Master volume',
      sample: 'Sample', filter_fx: 'FX filter', distortion: 'Distortion', chorus: 'Chorus', compressor: 'Compressor',
      eq: 'EQ', flanger: 'Flanger', phaser: 'Phaser', reverb: 'Reverb',
      levels: 'Levels', envelopes: 'Envelopes', modulations: 'Modulation', filter_routing: 'Filter routing',
      stereo_mode: 'Stereo mode', level: 'Level', gain: 'Gain', pan: 'Pan', pitch: 'Pitch', cutoff: 'Cutoff',
      resonance: 'Resonance', drive: 'Drive', mix: 'Mix', dry_wet: 'Mix', blend: 'Blend', transpose: 'Transpose',
      attack: 'Attack', decay: 'Decay', sustain: 'Sustain', release: 'Release', hold: 'Hold', delay: 'Delay',
      delay_time: 'Start delay', frequency: 'Frequency', tempo: 'Tempo', sync: 'Sync', trigger: 'Trigger',
      wave_frame: 'Table position', warp: 'Warp', spectral_morph: 'Spectral morph', trajectory: 'Tone movement',
      distortion_amount: 'Warp amount', distortion_type: 'Warp type', distortion_phase: 'Warp phase',
      distortion_spread: 'Warp spread', spectral_morph_amount: 'Spectral morph amount',
      spectral_morph_phase: 'Spectral morph phase', spectral_morph_spread: 'Spectral morph spread',
      detune_range: 'Detune range', detune_power: 'Detune distribution', stack_style: 'Unison stack',
      unison_voices: 'Unison voices', voices: 'Voices', feedback: 'Feedback', size: 'Size', tone: 'Tone',
      filter: 'Filter', destination: 'Routing', tune: 'Tune', keytrack: 'Keytrack',
      keytrack_transpose: 'Keytrack transpose', transpose_quantize: 'Transpose snap',
      stereo: 'Stereo', stereo_spread: 'Stereo spread', frame_spread: 'Frame spread',
      spectral_unison: 'Spectral unison', smooth_interpolation: 'Interpolation', smooth_time: 'Smooth',
      phase_offset: 'Phase offset', center: 'Center', freeze: 'Freeze', delay_1: 'Delay 1', delay_2: 'Delay 2',
      makeup: 'Level compensation', blend_modulation: 'Blend modulation', drive_modulation: 'Drive modulation',
      rt60: 'Decay time', decay_time: 'Decay time', band_mode: 'Mid band mode', band_gain: 'Mid band gain',
      band_cutoff: 'Mid band frequency', low_cutoff: 'Low band frequency', high_cutoff: 'High band frequency',
      low_lower_threshold: 'Low band lower threshold', low_upper_threshold: 'Low band upper threshold',
      band_lower_threshold: 'Mid band lower threshold', band_upper_threshold: 'Mid band upper threshold',
      high_lower_threshold: 'High band lower threshold', high_upper_threshold: 'High band upper threshold',
      pitch_bend_range: 'Pitch bend range', pitch_bend_up: 'Pitch bend range (up)', pitch_bend_down: 'Pitch bend range (down)',
      portamento_time: 'Portamento time', portamento_slope: 'Portamento slope', portamento_scale: 'Portamento scale',
      voice_transpose: 'Global transpose', voice_tune: 'Global tune', voice_priority: 'Voice priority',
      stereo_routing: 'Stereo routing', polyphony: 'Polyphony', legato: 'Legato',
      formant_x: 'Formant X', formant_y: 'Formant Y', formant_transpose: 'Formant transpose',
      formant_resonance: 'Formant resonance', formant_spread: 'Formant spread', blend_transpose: 'Blend transpose',
    },
  };
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

  function controlLabel(value) {
    const id = String(value).replace(/_$/, '');
    const labels = LABELS[lang];
    const label = (key) => own(labels, key) ? labels[key] : key;
    if (own(labels, id)) return labels[id];
    let match = /^(macro_control|env|lfo|random)_(\d+)$/.exec(id);
    if (match) return `${{ macro_control: 'Macro', env: 'ENV', lfo: 'LFO', random: labels.random }[match[1]]} ${match[2]}`;
    match = /^(osc|filter|env|lfo)_(\d+)_?$/.exec(id);
    if (match) return `${{ osc: 'OSC', filter: lang === 'ja' ? 'フィルター' : 'Filter', env: 'ENV', lfo: 'LFO' }[match[1]]} ${match[2]}`;
    match = /^modulation_(\d+)(?:_(.*))?$/.exec(id);
    if (match) {
      const names = lang === 'ja'
        ? { amount: '変調量', power: 'カーブ', ramp: '変化の時間', ramp_up: '上昇の時間', ramp_down: '下降の時間', static_aux: '補助制御', atomic_law: '連動制御', control_bus: '連動制御', makeup: '連動補正', target_amount: '変調量' }
        : { amount: 'Amount', power: 'Curve', ramp: 'Ramp time', ramp_up: 'Rise time', ramp_down: 'Fall time', static_aux: 'Auxiliary control', atomic_law: 'Linked control', control_bus: 'Linked control', makeup: 'Linked correction', target_amount: 'Amount' };
      const parameter = own(names, match[2]) ? names[match[2]] : match[2];
      return `${lang === 'ja' ? 'モジュレーション' : 'Modulation'} ${match[1]}` + (parameter ? ' · ' + parameter : '');
    }
    match = /^(osc|filter|env|lfo)_(\d+)_(.+)$/.exec(id);
    if (match) {
      const family = { osc: 'OSC', filter: lang === 'ja' ? 'フィルター' : 'Filter', env: 'ENV', lfo: 'LFO' }[match[1]];
      const parameter = match[1] === 'lfo' && match[3] === 'frequency' ? (lang === 'ja' ? '速度' : 'Rate')
        : match[1] === 'env' && match[3] === 'delay' ? labels.delay_time : label(match[3]);
      return `${family} ${match[2]} · ${parameter}`;
    }
    match = /^(filter_fx|sample|distortion|delay|chorus|compressor|eq|flanger|phaser|reverb)_(.+)$/.exec(id);
    if (match) {
      const families = lang === 'ja'
        ? { filter_fx: 'FX フィルター', sample: 'サンプル', distortion: 'Distortion', delay: 'Delay', chorus: 'Chorus', compressor: 'Compressor', eq: 'EQ', flanger: 'Flanger', phaser: 'Phaser', reverb: 'Reverb' }
        : { filter_fx: 'FX filter', sample: 'Sample', distortion: 'Distortion', delay: 'Delay', chorus: 'Chorus', compressor: 'Compressor', eq: 'EQ', flanger: 'Flanger', phaser: 'Phaser', reverb: 'Reverb' };
      const parameter = ['chorus', 'flanger', 'phaser'].includes(match[1]) && match[2] === 'frequency'
        ? (lang === 'ja' ? '速度' : 'Rate') : label(match[2]);
      return `${families[match[1]]} · ${parameter}`;
    }
    return id;
  }

  function normalizeNotice(note) {
    if (Array.isArray(note)) note = { kind: note[0], message: note[1] };
    note = note && typeof note === 'object' ? note : {};
    return { ...note, category: note.category || (note.kind === 'clipped' ? 'range_limited' : note.kind === 'unsupported' ? 'unsupported' : '') };
  }

  function subject(note) {
    const sources = (Array.isArray(note.sources) && note.sources.length ? note.sources : note.source ? [note.source] : []).map(controlLabel);
    let text = sources.length && note.destination ? sources.join(' + ') + ' → ' + controlLabel(note.destination)
      : note.feature ? controlLabel(note.feature) : note.destination ? controlLabel(note.destination) : '';
    const extra = [];
    if (sources.length && !note.destination) extra.push(sources.join(' + '));
    if (own(note, 'original') && own(note, 'converted')) {
      const value = (v) => v !== null && typeof v === 'object' ? JSON.stringify(v) : String(v);
      extra.push(value(note.original) + ' → ' + value(note.converted));
    }
    if (note.target_type !== undefined && note.target_type !== null) extra.push(String(note.target_type));
    if (extra.length) text += lang === 'ja' ? `（${extra.join('、')}）` : ` (${extra.join(', ')})`;
    return text;
  }

  // Notices that say the same thing are shown once, with every control they apply to.
  function noticeGroups(notes) {
    const groups = new Map();
    for (const note of notes) {
      const key = note.category + '\n' + (note.message || '');
      if (!groups.has(key)) groups.set(key, { category: note.category, message: note.message || '', subjects: [] });
      const text = subject(note);
      if (text && !groups.get(key).subjects.includes(text)) groups.get(key).subjects.push(text);
    }
    const order = ['unsupported', 'range_limited', 'control_fixed', 'control_partial', 'baked', 'alternative'];
    const rank = (group) => (order.indexOf(group.category) + order.length + 1) % (order.length + 1);
    return [...groups.values()].sort((x, y) => rank(x) - rank(y));
  }

  function noticeDetail(group, t) {
    const line = document.createElement('li');
    const title = own(t.noticeLabels, group.category) ? t.noticeLabels[group.category] : t.note;
    const head = document.createElement('div');
    head.textContent = (lang === 'ja' ? `［${title}］` : `[${title}] `) + group.subjects.join(lang === 'ja' ? '、' : ', ');
    line.appendChild(head);
    if (own(t.noticeHelp, group.category)) line.title = t.noticeHelp[group.category];
    if (group.message) {
      const body = document.createElement('div');
      body.textContent = group.message;
      line.appendChild(body);
    }
    return line;
  }

  function start() {
    ready = false;
    worker = new Worker('worker.js');
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === 'ready') {
        ready = true;
        $('version').textContent = message.version;
        render();
        pump();
      } else if (message.type === 'failed') {
        failed = message.error;
        render();
      } else if (message.type === 'start') {
        update(message.key, { state: 'run' });
      } else if (message.type === 'done') {
        clearTimeout(timer);
        busy = null;
        finish(message);
        pump();
      }
    };
    worker.onerror = (event) => {
      const reason = event.message || 'worker';
      if (busy) { update(busy, { state: 'bad', error: reason }); restart(); }
      else { failed = reason; render(); }
    };
  }

  // Stops whatever is running: the converter cannot be interrupted, so its worker is replaced.
  function restart() {
    clearTimeout(timer);
    worker.terminate();
    busy = null;
    start();
    render();
  }

  function finish(message) {
    if (!message.ok) return update(message.key, { state: 'bad', error: message.error });
    if (!items.has(message.key)) return;
    const size = message.buffer.byteLength;
    if (total + size > MAX_TOTAL) return update(message.key, { state: 'bad', code: 'full' });
    total += size;
    const notes = Array.isArray(message.user_notices) ? message.user_notices : Array.isArray(message.notes) ? message.notes : [];
    update(message.key, { state: 'ok', notes: notes.map(normalizeNotice), blob: new Blob([message.buffer]), crc: message.crc, size });
  }

  function save(blob, name) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  }

  // A stored (uncompressed) zip: the presets are compressed already, and the parts stay as blobs.
  function zip(entries) {
    const encoder = new TextEncoder();
    const now = new Date();
    const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
    const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    const files = [], directory = [];
    let offset = 0, length = 0;
    for (const item of entries) {
      const name = encoder.encode(item.output);
      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true);
      local.setUint16(4, 20, true);
      local.setUint16(6, 0x0800, true);                // names are UTF-8
      local.setUint16(10, time, true);
      local.setUint16(12, date, true);
      local.setUint32(14, item.crc, true);
      local.setUint32(18, item.size, true);
      local.setUint32(22, item.size, true);
      local.setUint16(26, name.length, true);
      const entry = new DataView(new ArrayBuffer(46));
      entry.setUint32(0, 0x02014b50, true);
      entry.setUint16(4, 0x0314, true);                // made on Unix, or unzip misreads the names
      entry.setUint16(6, 20, true);
      entry.setUint16(8, 0x0800, true);
      entry.setUint16(12, time, true);
      entry.setUint16(14, date, true);
      entry.setUint32(16, item.crc, true);
      entry.setUint32(20, item.size, true);
      entry.setUint32(24, item.size, true);
      entry.setUint16(28, name.length, true);
      entry.setUint32(38, 0x81A40000, true);           // a regular file, rw-r--r--
      entry.setUint32(42, offset, true);
      files.push(local, name, item.blob);
      directory.push(entry, name);
      offset += 30 + name.length + item.size;
      length += 46 + name.length;
    }
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, entries.length, true);
    end.setUint16(10, entries.length, true);
    end.setUint32(12, length, true);
    end.setUint32(16, offset, true);
    return new Blob([...files, ...directory, end], { type: 'application/zip' });
  }

  // One path segment that every system can hold: no reserved characters or device names, no trailing dot or space.
  const safe = (text) => {
    const name = text.replace(/[<>:"\\|?*\x00-\x1f]/g, '_').replace(/[ .]+$/, '').slice(0, 120) || 'Preset';
    return /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(name.split('.')[0]) ? '_' + name : name;
  };

  function add(file, folder) {
    if (!/\.vital$/i.test(file.name)) return;
    if (items.size >= MAX_FILES) { skipped = true; return; }
    const key = 'f' + next++;
    const used = new Set([...items.values()].map((item) => item.output.toLowerCase()));
    const stem = (folder ? folder.split('/').map(safe).join('/') + '/' : '') + safe(file.name.replace(/\.vital$/i, ''));
    let output = stem + '.SerumPreset';
    for (let n = 2; used.has(output.toLowerCase()); n++) output = `${stem} (${n}).SerumPreset`;
    items.set(key, { key, file, name: (folder ? folder + '/' : '') + file.name, output, state: 'wait', notes: [], open: false });
    queue.push(key);
  }

  async function pump() {
    while (ready && !busy && queue.length) {
      const key = queue.shift();
      const item = items.get(key);
      if (!item) continue;
      if (item.file.size > MAX_FILE) { update(key, { state: 'bad', code: 'big' }); continue; }
      busy = key;
      try {
        const buffer = await item.file.arrayBuffer();
        if (busy !== key) return;
        worker.postMessage({ type: 'convert', key, name: item.file.name, buffer }, [buffer]);
        timer = setTimeout(() => { update(key, { state: 'bad', code: 'timeout' }); restart(); }, TIMEOUT);
        return;
      } catch (error) {
        if (busy !== key) return;
        busy = null;
        update(key, { state: 'bad', error: String(error && error.message || error) });
      }
    }
  }

  // Progress arrives faster than a long list can be redrawn, so redraws are gathered.
  function update(key, change) {
    const item = items.get(key);
    if (!item) return;
    Object.assign(item, change);
    if (!pending) pending = setTimeout(() => { pending = 0; render(); }, 50);
  }

  // Reads a dropped folder. It stops once Clear has been pressed (the epoch moved on) or the
  // file limit is passed; one file past the limit is kept so that the limit is reported.
  function entries(entry, folder, out, started) {
    return new Promise((resolve) => {
      const stale = () => started !== epoch || out.length > MAX_FILES;
      if (stale()) return resolve();
      if (entry.isFile) {
        if (!/\.vital$/i.test(entry.name)) return resolve();
        entry.file((file) => { out.push([file, folder]); resolve(); }, () => resolve());
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const path = folder ? folder + '/' + entry.name : entry.name;
        const batch = () => reader.readEntries(async (found) => {
          if (!found.length || stale()) return resolve();
          for (const child of found) await entries(child, path, out, started);
          batch();
        }, () => resolve());
        batch();
      } else resolve();
    });
  }

  // The release notes replace the converter while #release-notes is in the address.
  async function route() {
    const open = location.hash === '#release-notes';
    $('notes').hidden = !open;
    $('convert').hidden = open;
    if (!open || notesLoaded) return;
    notesLoaded = true;
    const box = $('notes-list');
    try {
      const response = await fetch('RELEASES.txt', { cache: 'no-cache' });
      if (!response.ok) throw new Error(String(response.status));
      const blocks = (await response.text()).trim().split(/\n\s*\n/);
      box.textContent = '';
      for (const block of blocks) {
        const lines = block.split('\n').map((line) => line.trim()).filter(Boolean);
        if (!lines.length) continue;
        const title = document.createElement('h3');
        title.textContent = lines[0];
        const changes = document.createElement('ul');
        for (const text of lines.slice(1)) {
          const line = document.createElement('li');
          line.textContent = text;
          changes.appendChild(line);
        }
        box.append(title, changes);
      }
    } catch (error) {
      notesLoaded = false;
      box.textContent = TEXT[lang].failed;
    }
  }

  function leaveNotes() {
    if (location.hash !== '#release-notes') return;
    history.pushState(null, '', location.pathname + location.search);
    route();
  }

  async function take(transfer) {
    leaveNotes();
    const found = [], started = epoch;
    const roots = [...(transfer.items || [])].map((item) => item.webkitGetAsEntry && item.webkitGetAsEntry()).filter(Boolean);
    if (roots.length) {
      for (const root of roots) await entries(root, '', found, started);
      if (started !== epoch) return;
    } else {
      for (const file of transfer.files || []) found.push([file, '']);
    }
    found.sort((a, b) => (a[1] + '/' + a[0].name).localeCompare(b[1] + '/' + b[0].name));
    for (const [file, folder] of found) add(file, folder);
    render();
    pump();
  }

  function render() {
    const t = TEXT[lang];
    document.documentElement.lang = lang;
    $('lang').textContent = t.other;
    for (const node of document.querySelectorAll('[data-i18n]')) node.textContent = t[node.dataset.i18n];
    if (failed) { $('drop-title').textContent = t.failed + ': ' + failed; $('drop-sub').textContent = t.retry; }
    else if (!ready) $('drop-sub').textContent = t.loading;

    const all = [...items.values()];
    const done = all.filter((item) => item.state === 'ok' || item.state === 'bad').length;
    const good = all.filter((item) => item.state === 'ok');
    bar.hidden = !all.length;
    $('summary').textContent = t.summary(done, all.length, all.filter((item) => item.state === 'bad').length)
      + (skipped ? '　' + t.skipped(MAX_FILES) : '');
    $('download').textContent = good.length > 1 ? t.downloadAll : t.download;
    $('download').disabled = !good.length || done < all.length;

    list.textContent = '';
    for (const item of all) {
      const li = document.createElement('li');
      const row = document.createElement('div');
      row.className = 'row';
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = item.name;
      name.title = item.name;
      row.appendChild(name);
      if (item.state === 'ok' && item.notes.length) {
        // The summary counts the entries listed below it, so merged notices count once.
        if (item.groupsLang !== lang) { item.groups = noticeGroups(item.notes); item.groupsLang = lang; }
        const clipped = item.groups.filter((group) => group.category === 'range_limited').length;
        const unsupported = item.groups.filter((group) => group.category === 'unsupported').length;
        const review = item.groups.length - clipped - unsupported;
        const more = document.createElement('button');
        more.type = 'button';
        more.className = 'more';
        more.textContent = [unsupported && t.unsupported(unsupported), clipped && t.clipped(clipped), review && t.review(review)]
          .filter(Boolean).join(' · ');
        more.onclick = () => { item.open = !item.open; render(); };
        row.appendChild(more);
      }
      const state = document.createElement('span');
      state.className = 'state' + (item.state === 'ok' ? ' ok' : item.state === 'bad' ? ' bad' : '');
      const reason = item.code ? t[item.code] : item.error;
      state.textContent = item.state === 'bad' && reason ? t.bad + ': ' + reason : t[item.state];
      state.title = state.textContent;
      row.appendChild(state);
      if (item.state === 'ok') {
        const one = document.createElement('button');
        one.type = 'button';
        one.textContent = t.save;
        one.onclick = () => save(item.blob, item.output.split('/').pop());
        row.appendChild(one);
      }
      li.appendChild(row);
      if (item.state === 'ok' && item.notes.length) {
        const detail = document.createElement('ul');
        detail.className = 'detail';
        detail.hidden = !item.open;
        for (const group of item.groups) detail.appendChild(noticeDetail(group, t));
        li.appendChild(detail);
      }
      list.appendChild(li);
    }
  }

  const choose = () => { if (failed) { failed = null; restart(); } else input.click(); };
  drop.onclick = choose;
  drop.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); } };
  input.onchange = () => { for (const file of input.files) add(file, ''); input.value = ''; render(); pump(); };
  for (const type of ['dragenter', 'dragover']) {
    document.addEventListener(type, (event) => { event.preventDefault(); drop.classList.add('over'); });
  }
  for (const type of ['dragleave', 'drop']) {
    document.addEventListener(type, (event) => { event.preventDefault(); drop.classList.remove('over'); });
  }
  document.addEventListener('drop', (event) => take(event.dataTransfer));
  $('download').onclick = () => {
    const good = [...items.values()].filter((item) => item.state === 'ok');
    if (good.length === 1) save(good[0].blob, good[0].output.split('/').pop());
    else if (good.length) save(zip(good), 'vital2serum.zip');
  };
  $('clear').onclick = () => {
    epoch++;
    items.clear();
    queue.length = 0;
    total = 0;
    skipped = false;
    if (busy) restart();
    render();
  };
  $('lang').onclick = () => {
    lang = lang === 'ja' ? 'en' : 'ja';
    try { localStorage.setItem('lang', lang); } catch (error) { /* storage unavailable */ }
    render();
  };
  $('back').onclick = leaveNotes;
  window.addEventListener('hashchange', route);
  window.addEventListener('popstate', route);
  window.__vital2serum = { items, add, pump, render, zip };
  start();
  render();
  route();
})();
