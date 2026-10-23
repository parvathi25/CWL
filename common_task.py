"""
common_task.py
===============
Shared components for the Multisensory Perceptual Load Task.
Haptics start at t=0ms using single-track combined stimuli.
Visual and Auditory switch to CS after BS jitter duration.

Auditory stimuli are synthesized (pure tone, amplitude change) and played
through sounddevice, so no .wav files are needed.

Includes ResponsePadInput: a drop-in wrapper so participant answers can come
from the Cedrus Riponda response pad (or the keyboard when disabled).
"""

from psychopy import visual, core, event
import numpy as np
import random
import socket
import sounddevice as sd


def send_event_marker(
    code,
    eeg_trigger=None,
    lsl_markers=None,
):
    """Send a marker to whichever outputs are initialized."""
    if eeg_trigger is not None:
        eeg_trigger.send(int(code))

    if lsl_markers is not None:
        lsl_markers.send(int(code))

# =========================================================================
# RESPONSE PAD KEY MAPPING & TIMEOUTS
# =========================================================================
SCREEN1_KEY_MAP = {'0': 0, '1': 1, '2': 2, '3': 3}
SCREEN2_KEY_MAP = {'1': 'auditory', '2': 'touch', '3': 'visual'}
CONFIDENCE_KEY_MAP = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
UNSURE_KEY = 'u'          # 'Unsure' on screen 1: yellow button on the Riponda (XID key 6), 'u' on keyboard
UNSURE_LABEL = 'unsure'   # value stored as n_reported when the participant answers Unsure
RATING_TIMEOUT = 10.000
FOCUS_KEY_MAP = {'1': 'auditory', '2': 'touch', '3': 'visual', '4': 'none'}
DETECTION_KEY_MAP = {'0': 0, '1': 1}   # matches the "0 = No   1 = Yes" on screen
QUIT_KEY = 'escape'

# =========================================================================
# RESPONSE PAD (Cedrus Riponda via pyxid2)
# =========================================================================
RIPONDA_PRODUCT_ID = b'5'
STIMTRACKER_PRODUCT_ID = b'S'

# XID key number (0-based) -> key name the task expects.
#   key 0    -> answer '0' (No in the detection task)
#   keys 1-5 -> answers / confidence ratings '1'-'5'
#   key 6    -> yellow 'Unsure' button (UNSURE_KEY)
# Confirm with test_response_pad() before real testing.
RIPONDA_BUTTON_MAP = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: UNSURE_KEY}

_XID_DEVICES = None


def get_xid_devices_cached():
    """Enumerate XID devices once (avoids opening the same serial port twice)."""
    global _XID_DEVICES
    if _XID_DEVICES is None:
        import pyxid2
        _XID_DEVICES = pyxid2.get_xid_devices()
    return _XID_DEVICES


def find_xid_device(product_id):
    """Return the first XID device whose product_id matches, or None."""
    for dev in get_xid_devices_cached():
        if getattr(dev, 'product_id', None) == product_id:
            return dev
    return None


class _PadKey:
    """Mimics a PsychoPy keyboard key event (.name, .rt)."""
    def __init__(self, name, rt=None):
        self.name = name
        self.rt = rt


class ResponsePadInput:
    """
    Drop-in replacement for psychopy.hardware.keyboard.Keyboard for participant answers.

    use_pad=True  : answers come from the Riponda; keyboard only listens for QUIT_KEY.
                    Raises RuntimeError if no Riponda is found.
    use_pad=False : plain keyboard passthrough (for testing without the pad).
    """
    def __init__(self, kb, use_pad=True, button_map=RIPONDA_BUTTON_MAP):
        self.kb = kb
        self.button_map = button_map
        self.dev = None

        if use_pad:
            self.dev = find_xid_device(RIPONDA_PRODUCT_ID)
            if self.dev is None:
                raise RuntimeError(
                    "USE_RESPONSE_PAD is True, but no Riponda was found "
                    f"(product_id={RIPONDA_PRODUCT_ID!r}). Check the USB connection."
                )
            print(f"Response pad connected: {self.dev.device_name}")
        else:
            print("Response pad disabled: using keyboard for participant responses.")

        self.clearEvents()

    @property
    def keyboard_mode(self):
        return self.dev is None

    def clearEvents(self):
        self.kb.clearEvents()
        if self.dev is not None:
            self.dev.poll_for_response()
            self.dev.clear_response_queue()

    def getKeys(self, keyList=None, waitRelease=False, clear=True):
        if self.dev is None:
            return self.kb.getKeys(keyList=keyList, waitRelease=waitRelease, clear=clear)

        out = []

        # Keyboard: escape only
        for k in self.kb.getKeys(keyList=[QUIT_KEY], waitRelease=False, clear=True):
            out.append(_PadKey(QUIT_KEY, getattr(k, 'rt', None)))

        # Response pad: presses only (ignore releases)
        self.dev.poll_for_response()
        while self.dev.has_response():
            r = self.dev.get_next_response()
            self.dev.poll_for_response()
            if not r.get('pressed', False):
                continue
            name = self.button_map.get(r['key'], f"pad{r['key']}")
            if keyList is not None and name not in keyList:
                continue
            rt = r['time'] / 1000.0 if r.get('time') is not None else None  # ms -> s
            out.append(_PadKey(name, rt))
        return out


def test_response_pad(seconds=30):
    """Run once to see which XID key number each physical button sends."""
    dev = find_xid_device(RIPONDA_PRODUCT_ID)
    if dev is None:
        print("No Riponda found.")
        return
    print(f"Press buttons for {seconds}s...")
    dev.poll_for_response()
    dev.clear_response_queue()
    t0 = core.getTime()
    while core.getTime() - t0 < seconds:
        dev.poll_for_response()
        while dev.has_response():
            r = dev.get_next_response()
            dev.poll_for_response()
            if r.get('pressed'):
                print("key number:", r['key'])

# =========================================================================
# TIMING (seconds)
# =========================================================================
BASELINE_JITTER_VALUES = [0.950, 0.975, 1.000, 1.025, 1.050] # +/- 50ms jitter around 1s baseline
CS_STAGGER_VALUES = [0.250, 0.275, 0.300]

PRESENTATION_TARGET = 1.950      # 1950ms total presentation window
MIN_FINAL_SEGMENT = 0.300
RESPONSE_DURATION = 8.000
FEEDBACK_MEAN = 0.300
FEEDBACK_JITTER = 0.050
ITI_DURATION = 1.000

# Threshold block: no jitter, fixed 1000 ms baseline + 950 ms change
THRESH_BASELINE_DURATION = 1.000
THRESH_CHANGE_DURATION = 0.950

# =========================================================================
# THRESHOLD-BLOCK LEVELS (% change) PER MODALITY
# =========================================================================
AUDIO_LEVELS = [2, 3, 5, 7, 10, 15, 22]     # % amplitude increase
VISUAL_LEVELS = [3, 6, 10, 15, 20, 30, 40]    # % coherent dots (100 dots -> whole-dot steps)
TOUCH_LEVELS = [2, 3, 5, 7, 10, 15, 20]  # % amplitude increase (tracks Change_5, Change_10, ...)      # % amplitude increase (tracks Change_5, Change_10, ...)

LEVELS_BY_MODALITY = {
    'visual': VISUAL_LEVELS,
    'auditory': AUDIO_LEVELS,
    'touch': TOUCH_LEVELS,
}

# Jitter to Haptic Mapping for L1, L3, L5 & Position 1
JITTER_HAPTIC_MAP = {
    1.000: 'BC1_L1',
    1.025: 'BC2_L1',
    0.975: 'BC3_L1',
    0.950: 'BC4_L1',
    1.050: 'BC5_L1'
}

# Haptic Mapping for Threshold-Finding task: Change_5, Change_10, ...
HAPTIC_THRESHOLD_TRACKS = {lv: f'Change_{lv}' for lv in TOUCH_LEVELS}

L2_TACTILE_TRACKS = ['BC1_L2', 'BC2_L2', 'BC3_L2', 'BC4_L2', 'BC5_L2', 'BC6_L2', 'BC7_L2']
L4_TACTILE_TRACKS = ['BC1_L4', 'BC2_L4', 'BC3_L4', 'BC4_L4', 'BC5_L4', 'BC6_L4', 'BC7_L4', 'BC8_L4', 'BC9_L4']

# =========================================================================
# VISUAL & AUDITORY CONFIG
# =========================================================================
N_DOTS = 100
DOT_SIZE_PIX = 4
APERTURE_DIAMETER_PX = 500
DOT_SPEED_PX_PER_SEC = 100
DOT_LIFE_FRAMES = 50
DOT_COLOR = 'white'

HAPTICS_HOST = 'localhost'
HAPTICS_PORT = 64209
HAPTICS_BASELINE_TRACK = 'Baseline'

BACKGROUND_IMAGE = 'apriltag.png'
MODALITIES = ['visual', 'auditory', 'touch']
DATA_DIR = 'data_Perceptual'


def measure_frame_rate(win, fallback=60.0):
    rate = win.getActualFrameRate(nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1)
    return rate if rate is not None else fallback


class HapticsController:
    def __init__(self, host=HAPTICS_HOST, port=HAPTICS_PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2)
            self.sock.connect((self.host, self.port))
        except Exception:
            self.sock = None

    def start_track(self, track_name):
        if self.sock is None:
            return
        try:
            self.sock.sendall((f'startTrack("{track_name}");\n').encode('utf-8'))
        except Exception:
            pass

    def stop(self):
        if self.sock is None:
            return
        try:
            self.sock.sendall(b'stop();\n')
        except Exception:
            pass

    def close(self):
        if self.sock is not None:
            try:
                self.stop()
                self.sock.close()
            except Exception:
                pass


class RDKField:
    def __init__(self, win, n_dots=N_DOTS, dot_size=DOT_SIZE_PIX, color=DOT_COLOR,
                 aperture_diameter_px=APERTURE_DIAMETER_PX,
                 speed_px_per_sec=DOT_SPEED_PX_PER_SEC,
                 dot_life=DOT_LIFE_FRAMES, frame_rate=None):
        if frame_rate is None:
            frame_rate = measure_frame_rate(win)

        speed_px_per_frame = speed_px_per_sec / frame_rate
        self.stim = visual.DotStim(
            win, units='pix', nDots=n_dots, fieldShape='circle',
            fieldSize=aperture_diameter_px, dotSize=dot_size, dotLife=dot_life,
            dir=0.0, speed=speed_px_per_frame, coherence=0.0, color=color,
            signalDots='same', noiseDots='direction')

    def set_baseline(self):
        self.stim.coherence = 0.0

    def set_change(self, percent, direction_deg=None):
        if direction_deg is None:
            direction_deg = random.uniform(0, 360)
        self.stim.dir = direction_deg
        self.stim.coherence = percent / 100.0

    def draw(self):
        self.stim.draw()


# =========================================================================
# AUDIO (synthesized pure tone, played through sounddevice)
# =========================================================================
AUDIO_SAFETY_PAD = 0.15
AUDIO_DEVICE = None        # None = Windows default output
AUDIO_SAMPLE_RATE = 48000  # your headphones run at 48000 Hz
AUDIO_GAIN = 1.0           # raise if too quiet; samples are clipped to +/-1
AUDIO_TONE_FREQ = 440.0    # Hz
AUDIO_BASE_AMP = 0.3       # baseline amplitude (leaves headroom for amplitude increases)


def synth_tone(duration, change_pct=None, change_at=None,
               freq=AUDIO_TONE_FREQ, amp=AUDIO_BASE_AMP, sr=AUDIO_SAMPLE_RATE,
               edge=0.010, amp_ramp=0.005):
    """
    Pure tone at constant frequency. If change_pct is given, the amplitude rises
    by change_pct % at `change_at` seconds (short linear ramp so there is no click).
    Returns a stereo float32 array.
    """
    n = int(np.ceil((duration + AUDIO_SAFETY_PAD) * sr))
    t = np.arange(n) / sr

    a = np.full(n, amp)
    if change_pct is not None and change_at is not None:
        a = amp * (1 + (change_pct / 100.0) * np.clip((t - change_at) / amp_ramp, 0, 1))

    x = a * np.sin(2 * np.pi * freq * t)

    # tone lasts exactly 'duration'; the safety padding after it is silence
    n_on = int(round(duration * sr))
    ne = int(edge * sr)
    env = np.zeros(n)
    env[:n_on] = 1.0
    env[:ne] = np.linspace(0, 1, ne)
    env[n_on - ne:n_on] = np.linspace(1, 0, ne)

    x = np.clip(x * env * AUDIO_GAIN, -1.0, 1.0).astype('float32')
    return np.ascontiguousarray(np.column_stack([x, x]))


class AuditoryStimulus:
    """Synthesizes and plays tones through sounddevice (PTB can't open a stream on this machine)."""

    def __init__(self):
        sd.default.latency = 'low'
        if AUDIO_DEVICE is not None:
            sd.default.device = AUDIO_DEVICE

    def _play(self, clip):
        sd.stop()
        try:
            sd.play(clip, AUDIO_SAMPLE_RATE)   # non-blocking
        except Exception as e:
            print("[audio] playback error:", e)

    def start_trial(self, duration, change_pct=None, change_at=None):
        """
        Play ONE continuous tone for the whole presentation.
        If change_pct is given, amplitude increases by that % at `change_at` seconds.
        """
        self._play(synth_tone(duration, change_pct=change_pct, change_at=change_at))

    def start_baseline(self, duration=PRESENTATION_TARGET):
        self._play(synth_tone(duration))

    def stop_all(self):
        try:
            sd.stop()
        except Exception:
            pass


def test_audio():
    """Run once from a separate script to check that sound plays."""
    a = AuditoryStimulus()
    a.start_trial(2.0, change_pct=30, change_at=1.0)
    core.wait(2.2)
    a.stop_all()


def get_haptic_track(level, modalities_changed, touch_position, bs_jitter, percent=0, is_threshold_block=False):
    """Determines exact haptic track to play according to Level/Task type."""
    if 'touch' not in modalities_changed:
        return HAPTICS_BASELINE_TRACK

    # Threshold Block: Change_5, Change_10, ...
    if is_threshold_block:
        return HAPTIC_THRESHOLD_TRACKS.get(percent, HAPTICS_BASELINE_TRACK)

    # Main MSPL Task Logic (Level 1-5 tracks)
    if level in (1, 3, 5):
        return JITTER_HAPTIC_MAP.get(bs_jitter, 'BC1_L1')
    elif level == 2:
        if touch_position == 1:
            return JITTER_HAPTIC_MAP.get(bs_jitter, 'BC1_L1')
        elif touch_position == 2:
            return random.choice(L2_TACTILE_TRACKS)
    elif level == 4:
        if touch_position == 1:
            return JITTER_HAPTIC_MAP.get(bs_jitter, 'BC1_L1')
        elif touch_position == 2:
            return random.choice(L2_TACTILE_TRACKS)
        elif touch_position == 3:
            return random.choice(L4_TACTILE_TRACKS)

    return HAPTICS_BASELINE_TRACK


def stop_all_stimuli(dots, audio, haptics):
    dots.set_baseline()
    audio.stop_all()
    haptics.stop()


def make_background(win, image_path=BACKGROUND_IMAGE):
    return visual.ImageStim(win, image=image_path, units='pix', size=win.size)


def run_instructions(win, kb, background, text):
    """
    Show instructions until a button is pressed.
    `kb` can be a PsychoPy Keyboard or a ResponsePadInput; any press advances.
    """
    text_stim = visual.TextStim(win, text=text, color='white', height=0.06, wrapWidth=1.6)
    kb.clearEvents()
    event.clearEvents()
    while True:
        background.draw()
        text_stim.draw()
        win.flip()
        keys = kb.getKeys()
        if keys:
            if any(k.name == QUIT_KEY for k in keys):
                raise KeyboardInterrupt
            break


def run_presentation(
    win, dots, audio, haptics, trial, background,
    is_threshold_block=False, eeg_trigger=None, lsl_markers=None
    ):
    """
    Presentation Window: Total time 1950ms.
    Audio, Haptics, and Visual dots start baseline at t = 0 ms synchronously.
    Audio is ONE continuous clip with the change spliced in at the right time.
    Visual changes (and the markers) are timed in the frame loop.
    """
    bs_jitter = random.choice(BASELINE_JITTER_VALUES)
    modalities = trial['modalities_changed']
    timing_mode = trial['timing_mode']
    percents = trial.get('percents', {})
    level = trial.get('level', 1)

    touch_position = (modalities.index('touch') + 1) if 'touch' in modalities else 0
    touch_percent = percents.get('touch', 0)
    haptic_track = get_haptic_track(
        level, modalities, touch_position, bs_jitter,
        percent=touch_percent, is_threshold_block=is_threshold_block
    )
    dots.set_baseline()

    remaining = max(PRESENTATION_TARGET - bs_jitter, MIN_FINAL_SEGMENT)

    # ---- Pre-compute the change schedule so audio can be one continuous clip ----
    # segs[i] = how long to stay on modality i's segment (sequential mode only)
    segs = []
    audio_change_at = None          # seconds after trial start
    if timing_mode == 'sequential':
        n = len(modalities)
        gaps = [random.choice(CS_STAGGER_VALUES) for _ in range(n - 1)]
        elapsed = 0.0
        for i, m in enumerate(modalities):
            if m == 'auditory':
                audio_change_at = bs_jitter + elapsed
            if i < n - 1:
                seg = min(gaps[i], max(remaining - elapsed, 0))
            else:
                seg = max(remaining - elapsed, MIN_FINAL_SEGMENT)
            segs.append(seg)
            elapsed += seg
    elif 'auditory' in modalities:
        audio_change_at = bs_jitter

    audio_pct = percents.get('auditory') if 'auditory' in modalities else None

    # Pre-render initial frame buffer so visual display is ready
    background.draw()
    dots.draw()

    # SYNCHRONOUS ONSET AT t = 0 ms
    clock = core.Clock()
    audio.start_trial(
        PRESENTATION_TARGET,
        change_pct=audio_pct,
        change_at=audio_change_at,
    )
    haptics.start_track(haptic_track)

    # Marker 102: baseline stimulus onset, aligned with the visual flip.
    win.callOnFlip(send_event_marker, 102, eeg_trigger, lsl_markers)
    win.flip()

    # RUN BASELINE PHASE (t = 0 to t = bs_jitter)
    while clock.getTime() < bs_jitter:
        background.draw()
        dots.draw()
        win.flip()
        if event.getKeys([QUIT_KEY]):
            raise KeyboardInterrupt

    # RUN CHANGING STIMULUS PHASE (Visual & Auditory)
    if timing_mode in ('none', 'single', 'simultaneous'):
        visual_change_pending = False

        for m in modalities:
            if m == 'visual':
                dots.set_change(percents.get('visual', 0))
                visual_change_pending = True
            elif m == 'auditory':
                # Marker 104: auditory change (audio is already in the clip).
                send_event_marker(104, eeg_trigger, lsl_markers)

        t_end = clock.getTime() + remaining

        while clock.getTime() < t_end:
            background.draw()
            dots.draw()

            if visual_change_pending:
                # Marker 103: visual change, on the next visual flip.
                win.callOnFlip(send_event_marker, 103, eeg_trigger, lsl_markers)
                visual_change_pending = False

            win.flip()

            if event.getKeys([QUIT_KEY]):
                raise KeyboardInterrupt

    elif timing_mode == 'sequential':
        for i, m in enumerate(modalities):
            visual_change_pending = False

            if m == 'visual':
                dots.set_change(percents.get('visual', 0))
                visual_change_pending = True
            elif m == 'auditory':
                # Marker 104: auditory change (audio is already in the clip).
                send_event_marker(104, eeg_trigger, lsl_markers)

            t_seg_end = clock.getTime() + segs[i]
            while clock.getTime() < t_seg_end:
                background.draw()
                dots.draw()

                if visual_change_pending:
                    win.callOnFlip(send_event_marker, 103, eeg_trigger, lsl_markers)
                    visual_change_pending = False

                win.flip()

                if event.getKeys([QUIT_KEY]):
                    raise KeyboardInterrupt


def make_screen1_stim(win, unsure_label='Unsure'):
    """Returns a list of stims: question + numbers (white) and the Unsure option (yellow)."""
    main = visual.TextStim(
        win, text='How many stimuli changed?\n\n0        1        2        3',
        height=0.1, wrapWidth=1.6, color='white', pos=(0, 0.1))
    unsure = visual.TextStim(
        win, text=unsure_label, height=0.1, wrapWidth=1.6, color='yellow', pos=(0, -0.25))
    return [main, unsure]


def make_screen2_stim(win, selected):
    labels = []
    for key, name in SCREEN2_KEY_MAP.items():
        mark = '[X]' if name in selected else '[ ]'
        labels.append(f'{mark} {name} ({key})')
    text = 'Which stimuli changed? (select all that apply)\n\n' + '   '.join(labels)
    return visual.TextStim(win, text=text, height=0.1, wrapWidth=1.6, color='white')


def run_response_two_screen(
    win, kb, background,
    eeg_trigger=None, lsl_markers=None,
    ):
    """`kb` can be a PsychoPy Keyboard or a ResponsePadInput.
    Screen 1 and screen 2 share ONE RESPONSE_DURATION (8 s) budget."""
    budget_clock = core.Clock()
    result = {'n_reported': None, 'modalities_reported': set(),
              'rt_screen1': None, 'rt_screen2': None}

    # Plain keyboard object (or pad disabled) -> tell the tester which key is Unsure.
    unsure_label = 'Unsure' if not getattr(kb, 'keyboard_mode', True) else 'Unsure (press U)'
    screen1 = make_screen1_stim(win, unsure_label)
    kb.clearEvents()
    locked = False
    response_onset_pending = True

    # ---- Screen 1: how many changed? ----
    while budget_clock.getTime() < RESPONSE_DURATION and not locked:
        background.draw()
        for s in screen1:
            s.draw()

        if response_onset_pending:
            # Marker 106: response window onset.
            win.callOnFlip(send_event_marker, 106, eeg_trigger, lsl_markers)
            response_onset_pending = False

        win.flip()
        keys = kb.getKeys(list(SCREEN1_KEY_MAP.keys()) + [UNSURE_KEY, QUIT_KEY], waitRelease=False)
        for k in keys:
            if k.name == QUIT_KEY:
                raise KeyboardInterrupt
            if k.name in SCREEN1_KEY_MAP:
                send_event_marker(107, eeg_trigger, lsl_markers)
                result['n_reported'] = SCREEN1_KEY_MAP[k.name]
                result['rt_screen1'] = budget_clock.getTime()
                locked = True
                break
            if k.name == UNSURE_KEY:
                send_event_marker(107, eeg_trigger, lsl_markers)
                result['n_reported'] = UNSURE_LABEL
                result['rt_screen1'] = budget_clock.getTime()
                locked = True
                break

    # ---- Screen 2: which ones changed? (only if 1 or 2 reported) ----
    if result['n_reported'] in (1, 2):
        selected = set()
        kb.clearEvents()
        screen2_start = budget_clock.getTime()
        while budget_clock.getTime() < RESPONSE_DURATION:
            screen2 = make_screen2_stim(win, selected)
            background.draw()
            screen2.draw()
            win.flip()
            keys = kb.getKeys(list(SCREEN2_KEY_MAP.keys()) + [QUIT_KEY], waitRelease=False)
            for k in keys:
                if k.name == QUIT_KEY:
                    raise KeyboardInterrupt
                if k.name in SCREEN2_KEY_MAP:
                    selected.add(SCREEN2_KEY_MAP[k.name])
            if len(selected) >= result['n_reported']:
                # Show the final tick briefly so the last press is visible.
                background.draw()
                make_screen2_stim(win, selected).draw()
                win.flip()
                core.wait(0.2)
                break
        result['modalities_reported'] = selected
        result['rt_screen2'] = budget_clock.getTime() - screen2_start

    return result


def make_detection_screen_stim(win):
    return visual.TextStim(
        win, text='Did the stimulus change?\n\n0 = No        1 = Yes',
        height=0.1, wrapWidth=1.6, color='white')


def run_response_detection(win, kb, background, eeg_trigger=None, lsl_markers=None):
    """Single-screen yes/no change detection. `kb` can be a Keyboard or ResponsePadInput."""
    budget_clock = core.Clock()
    result = {'reported': None, 'rt': None}

    screen1 = make_detection_screen_stim(win)
    kb.clearEvents()
    response_onset_pending = True

    while budget_clock.getTime() < RESPONSE_DURATION and result['reported'] is None:
        background.draw()
        screen1.draw()

        if response_onset_pending:
            win.callOnFlip(send_event_marker, 106, eeg_trigger, lsl_markers)
            response_onset_pending = False

        win.flip()
        keys = kb.getKeys(list(DETECTION_KEY_MAP.keys()) + [QUIT_KEY], waitRelease=False)
        for k in keys:
            if k.name == QUIT_KEY:
                raise KeyboardInterrupt
            if k.name in DETECTION_KEY_MAP:
                send_event_marker(107, eeg_trigger, lsl_markers)
                result['reported'] = DETECTION_KEY_MAP[k.name]
                result['rt'] = budget_clock.getTime()
                break

    return result


def run_confidence_screen(
    win, kb, background,
    eeg_trigger=None, lsl_markers=None,
):
    """
    Collect a 1-5 confidence rating using five colored squares.

    1 = Low confidence
    5 = High confidence

    `kb` can be a PsychoPy Keyboard or a ResponsePadInput.

    Returns:
        confidence (int or None)
        confidence_rt (float or None)
    """

    # Explicit pixel units because MSPL.py creates a norm-units window.
    question = visual.TextStim(
        win,
        text="How confident are you about your answer?",
        color='white',
        height=40,
        pos=(0, 220),
        units='pix'
    )

    square_colors = ['#FFD6D6', '#FF9999', '#FF5555', '#CC0000', '#800000']
    number_colors = ['black', 'black', 'black', 'white', 'white']
    x_positions = [-360, -180, 0, 180, 360]
    square_size = 120
    square_y = 20

    squares = []
    numbers = []

    for i, x in enumerate(x_positions):
        squares.append(
            visual.Rect(
                win,
                width=square_size,
                height=square_size,
                pos=(x, square_y),
                fillColor=square_colors[i],
                lineColor='white',
                lineWidth=2,
                units='pix'
            )
        )

        numbers.append(
            visual.TextStim(
                win,
                text=str(i + 1),
                color=number_colors[i],
                height=50,
                bold=True,
                pos=(x, square_y),
                units='pix'
            )
        )

    label_y = square_y - square_size / 2 - 40

    low_label = visual.TextStim(
        win,
        text="Low",
        color='white',
        height=32,
        pos=(x_positions[0], label_y),
        units='pix'
    )

    high_label = visual.TextStim(
        win,
        text="High",
        color='white',
        height=32,
        pos=(x_positions[-1], label_y),
        units='pix'
    )

    instruction = visual.TextStim(
        win,
        text="Press 1-5 on the response pad / keyboard",
        color='white',
        height=28,
        pos=(0, -220),
        units='pix'
    )

    # Clear any keypresses left over from the previous response.
    kb.clearEvents()
    event.clearEvents()

    confidence = None
    confidence_rt = None
    clock = core.Clock()

    confidence_onset_pending = True

    while clock.getTime() < RATING_TIMEOUT:

        background.draw()
        question.draw()

        for sq, num in zip(squares, numbers):
            sq.draw()
            num.draw()

        low_label.draw()
        high_label.draw()
        instruction.draw()

        if confidence_onset_pending:
            # Marker 200: confidence screen onset.
            win.callOnFlip(
                send_event_marker,
                200,
                eeg_trigger,
                lsl_markers,
            )
            confidence_onset_pending = False

        win.flip()

        keys = kb.getKeys(
            keyList=list(CONFIDENCE_KEY_MAP.keys()) + [QUIT_KEY],
            waitRelease=False
        )

        for key in keys:
            if key.name == QUIT_KEY:
                raise KeyboardInterrupt

            if key.name in CONFIDENCE_KEY_MAP:
                send_event_marker(201, eeg_trigger, lsl_markers)
                confidence = CONFIDENCE_KEY_MAP[key.name]
                confidence_rt = clock.getTime()
                return confidence, confidence_rt

    # No valid response within the timeout.
    return None, None


def run_focus_screen(win, kb, background):
    """`kb` can be a PsychoPy Keyboard or a ResponsePadInput."""
    text_stim = visual.TextStim(
        win,
        text='Did you focus more on any specific sense?\n\n'
             '1 = Auditory    2 = Touch    3 = Visual    4 = None (Equal)',
        height=0.08, wrapWidth=1.6, color='white'
    )

    budget_clock = core.Clock()
    kb.clearEvents()

    while budget_clock.getTime() < RATING_TIMEOUT:
        background.draw()
        text_stim.draw()
        win.flip()

        keys = kb.getKeys(list(FOCUS_KEY_MAP.keys()) + [QUIT_KEY], waitRelease=False)
        for k in keys:
            if k.name == QUIT_KEY:
                raise KeyboardInterrupt
            if k.name in FOCUS_KEY_MAP:
                return FOCUS_KEY_MAP[k.name], budget_clock.getTime()

    return None, None


def run_feedback(
    win, correct, background,
    eeg_trigger=None, lsl_markers=None,
    ):
    fb_dur = random.uniform(FEEDBACK_MEAN - FEEDBACK_JITTER, FEEDBACK_MEAN + FEEDBACK_JITTER)

    face_color = 'yellow'
    head = visual.Circle(win, radius=50, fillColor=face_color, lineColor='black',
                          lineWidth=3, units='pix')
    left_eye = visual.Circle(win, radius=6, fillColor='black', pos=(-18, 15), units='pix')
    right_eye = visual.Circle(win, radius=6, fillColor='black', pos=(18, 15), units='pix')

    n_pts = 20
    width, depth = 50, 10
    pts = []
    for i in range(n_pts):
        x = -width / 2 + width * i / (n_pts - 1)
        t = (x / (width / 2)) ** 2
        y = (depth * t - depth) if correct else (-depth * t + depth)
        pts.append((x, y - 10))
    mouth = visual.ShapeStim(win, vertices=pts, closeShape=False,
                              lineColor='black', lineWidth=4, units='pix')

    clock = core.Clock()
    feedback_onset_pending = True

    while clock.getTime() < fb_dur:
        background.draw()
        head.draw()
        left_eye.draw()
        right_eye.draw()
        mouth.draw()

        if feedback_onset_pending:
            # Marker 108 = correct feedback; 109 = incorrect feedback.
            feedback_code = 108 if correct else 109

            win.callOnFlip(
                send_event_marker,
                feedback_code,
                eeg_trigger,
                lsl_markers,
            )
            feedback_onset_pending = False

        win.flip()

    return fb_dur


def run_iti(
    win, background,
    eeg_trigger=None, lsl_markers=None,
    ):
    fixation = visual.TextStim(
        win, text='+', height=0.08, color='white'
    )

    t0 = core.getTime()
    fixation_onset_pending = True

    while core.getTime() - t0 < ITI_DURATION:
        background.draw()
        fixation.draw()

        if fixation_onset_pending:
            # Marker 101: fixation / ITI onset.
            win.callOnFlip(
                send_event_marker,
                101,
                eeg_trigger,
                lsl_markers,
            )
            fixation_onset_pending = False

        win.flip()


def cleanup_and_quit(
    win, audio, haptics,
    eeg_trigger=None, lsl_markers=None,
    ):
    try:
        audio.stop_all()
    finally:
        try:
            haptics.close()
        finally:
            try:
                if eeg_trigger is not None:
                    eeg_trigger.close()
            finally:
                try:
                    if lsl_markers is not None:
                        lsl_markers.close()
                finally:
                    win.close()

    core.quit()