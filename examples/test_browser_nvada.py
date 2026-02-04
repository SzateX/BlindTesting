import logging
import os
import shutil
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from blindtesting.blind_test_interface import BlindTestConfig, BlindTestInterface
from blindtesting.sounddevice_adapter import SounddeviceConfig
from blindtesting.whisper_adapter import WhisperAdapterConfig

load_dotenv()
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CLEAN_PROFILE_NAME = "ff-temp"

USER_JS_PREFS = [
    'user_pref("browser.startup.homepage", "about:blank");',
    'user_pref("browser.startup.page", 0);',
    'user_pref("browser.aboutwelcome.enabled", false);',
    'user_pref("trailhead.firstrun.didSeeAboutWelcome", true);',
    'user_pref("browser.startup.homepage_override.mstone", "ignore");',
    'user_pref("browser.startup.homepage_override.buildID", "20200101000000");',
    'user_pref("browser.startup.firstrunSkipsHomepage", false);',
    'user_pref("startup.homepage_welcome_url", "");',
    'user_pref("startup.homepage_welcome_url.additional", "");',
    'user_pref("browser.newtabpage.enabled", false);',
    'user_pref("browser.newtab.preload", false);',
    'user_pref("browser.toolbars.bookmarks.visibility", "never");',
    'user_pref("browser.shell.checkDefaultBrowser", false);',
    "",
]


def _build_interface() -> BlindTestInterface:
    sound_device = os.getenv("SOUNDDEVICE_DEVICE") or os.getenv("SOUND_DEVICE")
    whisper_device = os.getenv("WHISPER_DEVICE", "cuda")
    sound_debug_dir = os.getenv("SOUND_DEBUG_DIR")
    sample_rate_env = os.getenv("SOUND_SAMPLE_RATE")
    sample_rate = int(sample_rate_env) if sample_rate_env else None
    start_threshold = float(os.getenv("SOUND_START_THRESHOLD", "0.0015"))
    silence_threshold = float(os.getenv("SOUND_SILENCE_THRESHOLD", "0.001"))
    pre_roll_sec = float(os.getenv("SOUND_PRE_ROLL_SEC", "0.8"))
    blocksize = int(os.getenv("SOUND_BLOCKSIZE", "4096"))
    auto_probe_sr = os.getenv("SOUND_AUTO_PROBE_SR", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    apply_postprocess = os.getenv("SOUND_APPLY_POSTPROCESS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    interface = BlindTestInterface(
        whisper_config=WhisperAdapterConfig(device=whisper_device),
        sound_config=SounddeviceConfig(
            device=sound_device if sound_device else None,
            sample_rate=sample_rate,
            channels=1,
            blocksize=blocksize,
            start_threshold=start_threshold,
            silence_threshold=silence_threshold,
            pre_roll_sec=pre_roll_sec,
            apply_postprocess=apply_postprocess,
            debug_dump_dir=sound_debug_dir if sound_debug_dir else None,
            auto_probe_sample_rate=auto_probe_sr,
        ),
        test_config=BlindTestConfig(
            match_mode="contains",
            match_case=False,
            max_clicks=25,
            click_delay_sec=0.3,
        ),
    )
    return interface


def _prepare_clean_profile(clean_profile_dir: str) -> None:
    if os.path.isdir(clean_profile_dir):
        shutil.rmtree(clean_profile_dir, ignore_errors=True)
    os.makedirs(clean_profile_dir, exist_ok=True)

    user_js_path = Path(clean_profile_dir) / "user.js"
    user_js_path.write_text("\n".join(USER_JS_PREFS), encoding="utf-8")


def _close_firefox(interface: BlindTestInterface) -> None:
    interface.hotkey("alt", "f4")
    time.sleep(0.8)


@pytest.fixture(scope="module", autouse=True)
def suite_setup_and_cleanup():
    logger.info("Starting NVDA browser suite")
    yield
    logger.info("Finished NVDA browser suite")


@pytest.fixture
def browser_test_context():
    interface = _build_interface()
    clean_profile_dir = os.path.join(os.getenv("TEMP", os.getcwd()), CLEAN_PROFILE_NAME)
    _prepare_clean_profile(clean_profile_dir)

    logger.info("Audio devices:")
    for d in interface.sound.list_devices():
        logger.info(
            "idx=%s hostapi=%s in=%s out=%s default_in=%s default_out=%s name=%s",
            d["index"],
            d["hostapi"],
            d["max_input_channels"],
            d["max_output_channels"],
            d["is_default_input"],
            d["is_default_output"],
            d["name"],
        )
    logger.info("Selected loopback stream: %s", interface.sound.get_selected_stream_info())

    yield interface, clean_profile_dir

    try:
        _close_firefox(interface)
    finally:
        interface.sound.stop()
        shutil.rmtree(clean_profile_dir, ignore_errors=True)


def test_firefox_duckduckgo_wikipedia_nvada(browser_test_context):
    interface, clean_profile_dir = browser_test_context

    # Open Firefox in private window with a clean, isolated profile.
    interface.hotkey("win", "r")
    interface.wait_until_heard(
        expected="Uruchamianie dialog",
        timeout_sec=15.0,
        listen_duration_sec=2.0,
        poll_delay_sec=0.3,
        match_mode="fuzzy",
        fuzzy_threshold=0.62,
    )
    interface.type_text(
        f'firefox -private-window -no-remote -profile "{clean_profile_dir}" about:blank'
    )
    interface.press("enter")
    interface.wait_until_heard(
        expected="Witamy w przegladarce",
        timeout_sec=12.0,
        listen_duration_sec=3.0,
        poll_delay_sec=0.3,
        match_mode="fuzzy",
        fuzzy_threshold=0.58,
    )
    interface.press("enter")
    interface.wait_until_heard(
        expected="Mozilla Firefox",
        timeout_sec=12.0,
        listen_duration_sec=2.0,
        poll_delay_sec=0.3,
        match_mode="fuzzy",
        fuzzy_threshold=0.62,
    )

    # Focus address bar and open DuckDuckGo homepage.
    interface.hotkey("ctrl", "l")
    time.sleep(0.2)
    interface.type_text("duckduckgo.pl")
    interface.press("enter")
    time.sleep(3.0)

    # Enter the search phrase on DuckDuckGo.
    interface.type_text("Python")
    interface.press("enter")
    time.sleep(3.0)

    # Press NVDA+J until we hear the Wikipedia result announcement.
    heard_text = interface.repeat_hotkey_until_heard(
        ("insert", "j"),
        expected="python minus wikipedia, wolna encyklopedia",
        match_mode="fuzzy",
        fuzzy_threshold=0.65,
        max_clicks=40,
    )
    logger.info("Matched NVDA text: %s", heard_text)

    interface.press("enter")
    interface.hotkey("insert", "t")

    final_heard = interface.wait_until_heard(
        expected="Wikipedia",
        timeout_sec=12.0,
        poll_delay_sec=0.3,
        match_mode="contains",
    )
    assert interface._match_text(
        final_heard,
        "Wikipedia",
        match_mode="contains",
    ), f"Expected Wikipedia heading not heard. Last heard: {final_heard}"
