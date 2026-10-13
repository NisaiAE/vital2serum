# vital2serum

Converts Vital presets (`.vital`) to Serum 2 presets (`.SerumPreset`).

Web version: https://nisaiae.github.io/vital2serum/ (runs in the browser; presets are not uploaded)

## Usage

```sh
pip install .
vital2serum convert preset.vital -o out/
vital2serum convert presets/ -o out/ --recursive
```

## Limitations

- Use the result as a starting point for editing in Serum 2. Parameter values and signal paths use the available Serum controls and may differ from Vital.
- Open each file's notices to check omitted settings, limited values, controls fixed at their saved positions, and waveforms baked into tables. The web notices and CLI report identify the affected controls where known.
- Complex filter routing, high-feedback resonators, several moving warp/morph axes, and nested modulation can be partial or replaced by a simpler configuration. Some Macro and wheel operations cannot be retained.
- Not converted: flanger effects, formant filters, random modulation sources, envelope delays, modulation ramp times and custom tunings. Spectral/warp modes, comb and phaser filters, phaser effects and distortion modes have conditional support; check each result's notices.
- Presets saved by Vital before 1.0 must be saved again in Vital 1.0 or later first.
- Checked against Serum 2.1.5 and Vital 1.0.7 and 1.6.4.
- Converting a preset does not change the terms it was licensed under. Check them before sharing a converted preset.

GPL-3.0-or-later; see `NOTICE.txt` for the works this program is derived from. Not affiliated with or endorsed by Xfer Records or Vital Audio.
