# BlindTesting

Browser accessibility test automation using screen reader speech capture + Whisper transcription.

**Experimental status:** This repository is an experimental example of accessibility test automation (screen reader + audio transcription). It is not production-ready and APIs/workflows may change.

## What this project does
- Drives keyboard interactions (PyAutoGUI).
- Captures loopback audio from system output (`SoundCard`).
- Transcribes spoken screen reader output with Whisper.
- Asserts expected phrases in automated pytest flows.

Current example test:
- `examples/test_browser_nvada.py`
  - launches Firefox with a temporary clean profile
  - navigates to DuckDuckGo and searches for `Python`
  - listens for NVDA-announced result text and asserts `Wikipedia` was heard

## How the example works (step by step)
1. Builds a `BlindTestInterface` with sound + whisper configuration from `.env`.
2. Creates a clean temporary Firefox profile (`%TEMP%\\ff-temp`) with onboarding/welcome prefs disabled.
3. Uses keyboard automation (`Win+R`, typing, hotkeys) to perform navigation and search.
4. Captures loopback audio and transcribes NVDA speech to text.
5. Waits/retries until expected phrases are heard (`fuzzy`/`contains` matching).
6. Asserts final expected phrase and performs cleanup (close browser, stop loopback, remove temp profile).

## Project structure
- `src/blindtesting/blind_test_interface.py` - high-level keyboard + speech-match helpers.
- `src/blindtesting/sounddevice_adapter.py` - loopback recording adapter.
- `src/blindtesting/whisper_adapter.py` - Whisper transcription adapter.
- `examples/test_browser_nvada.py` - end-to-end Firefox/NVDA example.

## Core helper methods
- `listen_once(duration_sec=None)` - one transcription; default is record-until-silence.
- `listen_for(duration_sec)` - fixed-length capture.
- `wait_until_heard(...)` - poll/listen until phrase appears or timeout.
- `repeat_press_until_heard(...)` - press key repeatedly and listen each attempt.
- `repeat_hotkey_until_heard(...)` - same as above for hotkeys.

Matching modes:
- `contains` - substring match
- `regex` - regular expression
- `fuzzy` - similarity-based match (`fuzzy_threshold`)

## Requirements
- Windows (desktop automation target)
- Python 3.10+
- NVDA running
- Audio loopback device available
- NVIDIA GPU + CUDA-compatible PyTorch build (optional, for faster Whisper transcription)

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CUDA note:
- `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cu130` and pins `torch==2.9.1`.
- If you want CPU-only transcription, keep `WHISPER_DEVICE=cpu` in `.env` (and you may replace CUDA torch with a CPU build if desired).

## Configuration
Create `.env` (or copy from `.env.example`) and adjust if needed.

Recommended minimal `.env`:

```env
LOG_LEVEL=INFO
WHISPER_DEVICE=cuda
SOUND_DEVICE=1
```

Variable guide:
- `LOG_LEVEL` - `DEBUG` for troubleshooting, `INFO` for normal runs.
- `WHISPER_DEVICE` - `cuda` (faster) or `cpu` (portable).
- `SOUND_DEVICE` - output device selector (index like `1` or name fragment). If not provided, the default output device is used.
- `SOUND_SAMPLE_RATE` - fixed sample rate (empty = auto-probe when enabled).
- `SOUND_AUTO_PROBE_SR` - `1` to auto-detect stable sample rate, `0` to use fixed/default.
- `SOUND_BLOCKSIZE` - audio chunk size (higher = more stable, slower reaction).
- `SOUND_START_THRESHOLD` - level to start recording speech.
- `SOUND_SILENCE_THRESHOLD` - level treated as silence.
- `SOUND_PRE_ROLL_SEC` - audio kept before speech start (helps catch first syllables).
- `SOUND_APPLY_POSTPROCESS` - `1` enables normalization/gain postprocess.
- `SOUND_DEBUG_DIR` - writes captured `.wav` files for diagnosis.

Practical tuning:
- Misses beginning of words -> increase `SOUND_PRE_ROLL_SEC` (e.g. `1.0`).
- Triggers too easily on noise -> increase `SOUND_START_THRESHOLD`.
- Cuts speech too early -> lower `SOUND_SILENCE_THRESHOLD` or increase silence settings in code.
- No audio captured -> set `SOUND_DEVICE` explicitly to the correct speaker/loopback source.

## Run tests
Run the example directly:

```powershell
pytest examples/test_browser_nvada.py -s
```

Pytest is configured in `pytest.ini` with:
- `pythonpath = src` (so imports use `blindtesting.*`)
- CLI logging enabled

## Notes
- Tests are interactive and control the active desktop session.
- Keep keyboard/mouse idle during execution.
- Speech matching uses `contains`, `regex`, or `fuzzy` matching modes.
- You can cap recording window per listen call using `listen_duration_sec` in wait/repeat helpers.
- Results depend on NVDA voice, language, system audio routing, and background sounds.
