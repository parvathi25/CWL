"""
Multiple Object Tracking (MOT) Task
====================================================================
5 difficulty levels, PsychoPy implementation, with Cedrus response-pad
support via pyxid2.

    Level 1: 2 slow targets   + 6 distractors
    Level 2: 3 slow targets   + 6 distractors
    Level 3: 3 fast targets   + 6 distractors
    Level 4: 4 slow targets   + 6 distractors
    Level 5: 4 fast targets   + 6 distractors

Trial structure
----------------
1. Fixation cross, jittered 800-1000 ms (mean 900 ms)
2. Cueing (1000 ms): all spheres shown static, targets in a different
   colour than distractors
3. Tracking (5000 ms): right at the start, targets change colour to match
   the distractors, then everything moves for 5 s inside a constrained
   central region of the screen
4. Response (max 5000 ms): motion stops, every sphere is numbered, and
   4 answer options (A/B/C/D, staggered up/down) are shown, each listing
   the FULL set of target numbers for that option. One option is the
   correct target set, the other three are foil sets. Participant
   responds via the Cedrus response pad (or keyboard a/b/c/d as fallback).
5. Feedback (300 +/- 50 ms): a drawn smiley face for correct, sad face
   for wrong (not emoji text -- see notes at bottom).

Trials are fully shuffled across all levels (not blocked), so an equal
number of trials from every level appears in a random order.
====================================================================
"""

import os
import csv
import math
import random
from psychopy import visual, core, event, gui, data
from psychopy.iohub import launchHubServer
import pyxid2
from eeg_triggers import EEGTrigger
from lsl_markers import LSLMarkers

# ---------------------------------------------------------------------------
# 1. EXPERIMENT CONFIGURATION -- edit these values to tune the task
# ---------------------------------------------------------------------------
FULLSCREEN = True                # set True for real data collection
WIN_SIZE = (1280, 1200)           # ignored when FULLSCREEN = True
BG_COLOR = 'black'

N_TRIALS_PER_LEVEL = 10           # trials per level

N_DISTRACTORS = 6                # constant across all levels

LEVELS = [
    {'label': 'Level 1', 'n_targets': 2, 'speed': 'slow'},
    {'label': 'Level 2', 'n_targets': 2, 'speed': 'fast'},
    {'label': 'Level 3', 'n_targets': 3, 'speed': 'slow'},
    {'label': 'Level 4', 'n_targets': 3, 'speed': 'fast'},
    {'label': 'Level 5', 'n_targets': 4, 'speed': 'fast'},
]

SPEED_PX_S = {'slow': 250, 'fast': 500}   # pixels/second -- tune to your monitor/viewing distance

SPHERE_RADIUS = 30                # pixels
TARGET_CUE_COLOR = 'yellow'      # colour of TARGETS only, during cueing
OBJECT_COLOR = 'fuchsia'            # distractor colour AND target colour once tracking starts

FIX_DUR_RANGE = (0.8, 1.0)        # seconds, uniform jitter -> mean 900 ms
CUE_DUR = 1.0                     # seconds
TRACK_DUR = 5.0                   # seconds
RESPONSE_TIMEOUT = 5.0            # seconds
FEEDBACK_DUR_RANGE = (0.25, 0.35)   # 300 +/- 50 ms

# ---- Response input ----
USE_RESPONSE_PAD = False                             # False = use keyboard a/b/c/d instead
RESPONSE_KEY_MAP = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}   # pad key number -> option letter
                                                       # !! confirm these numbers with the
                                                       # calibration script before real testing !!
RESPONSE_KEYS = ['a', 'b', 'c', 'd']                  # the letters used everywhere below
                                                       # (also the keyboard fallback keys)
# ---- Confidence rating ----
CONFIDENCE_KEY_MAP = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}

CONFIDENCE_TIMEOUT = 10.0

# ---- XID device identification ----
# Confirmed from diagnose_xid.py:
# Riponda Model L       -> product_id b'5'
# Cedrus StimTracker   -> product_id b'S'

RIPONDA_PRODUCT_ID = b'5'
STIMTRACKER_PRODUCT_ID = b'S'

# ---- Marker outputs ----
USE_EEG_TRIGGERS = False    # False = skip StimTracker entirely (e.g. testing without the amp)
USE_LSL_MARKERS = True     # False = skip LSL stream entirely (e.g. testing without eye tracker)
USE_IOHUB_LOG = True       # local backup log of every marker + timestamp, independent of hardware

DATA_DIR = 'data_MOT'   # folder to save the .csv files produced by this script

BACKGROUND_IMAGE_PATH = 'apriltag.png'   # <-- set this to your image's actual filename

# ---- Event markers (sent via ioHub's sendMessageEvent) ----
MARKERS = {
    'task_start':         100,
    'fixation_onset':     101,
    'cue_onset':          102,
    'tracking_onset':     103,
    'response_onset':     104,
    'key_press':          105,
    'feedback_correct':   106,
    'feedback_incorrect': 107,
    'confidence_onset':   200,
    'confidence_response':201,
    'task_end':           110,
}

# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

class MarkerBroadcaster:
    """Fans a single event marker out to whichever of these are enabled:
    ioHub's local log, the Cedrus StimTracker (EEG trigger), and an LSL
    stream (eye-tracker sync). Each hardware backend lives in its own
    module (eeg_triggers.py / lsl_markers.py) -- this class just calls
    out to them, so the rest of mot_task.py only ever does markers.send(name)."""

    def __init__(self, io=None, eeg=None, lsl=None):
        self.io = io
        self.eeg = eeg
        self.lsl = lsl

    def send(self, marker_name):
        code = MARKERS[marker_name]
        if self.io is not None:
            self.io.sendMessageEvent(text=str(code), category=marker_name)
        if self.eeg is not None:
            self.eeg.send(code)
        if self.lsl is not None:
            self.lsl.send(code)

    def close(self):
        if self.eeg is not None:
            self.eeg.close()
        if self.lsl is not None:
            self.lsl.close()


def get_region(win):
    """Return (xmin, xmax, ymin, ymax) of the allowed movement area:
    middle 6/8 of the width, middle 2/4 of the height, centred on screen."""
    w, h = win.size
    xmin = -w / 2 + w * (1 / 6)
    xmax = w / 2 - w * (1 / 6)
    ymin = -h / 2 + h * (1 / 4)
    ymax = h / 2 - h * (1 / 4)
    return xmin, xmax, ymin, ymax


def random_positions(n, radius, region, min_sep_factor=2.3, max_tries=4000):
    """Rejection-sample n non-overlapping (x, y) positions inside region."""
    xmin, xmax, ymin, ymax = region
    positions = []
    tries = 0
    while len(positions) < n and tries < max_tries:
        tries += 1
        x = random.uniform(xmin + radius, xmax - radius)
        y = random.uniform(ymin + radius, ymax - radius)
        if all(math.hypot(x - px, y - py) >= radius * min_sep_factor for px, py in positions):
            positions.append((x, y))
    if len(positions) < n:
        raise RuntimeError(
            "Could not place all objects without overlap -- reduce SPHERE_RADIUS "
            "or the number of objects, the tracking region may be too small.")
    return positions


def random_velocity(speed):
    angle = random.uniform(0, 2 * math.pi)
    return speed * math.cos(angle), speed * math.sin(angle)


def bounce_and_move(obj, dt, region, radius):
    """Advance obj['pos'] by obj['vel']*dt, bouncing off the region walls."""
    xmin, xmax, ymin, ymax = region
    x, y = obj['pos']
    vx, vy = obj['vel']
    x += vx * dt
    y += vy * dt
    if x - radius < xmin:
        x = xmin + radius
        vx = -vx
    elif x + radius > xmax:
        x = xmax - radius
        vx = -vx
    if y - radius < ymin:
        y = ymin + radius
        vy = -vy
    elif y + radius > ymax:
        y = ymax - radius
        vy = -vy
    obj['pos'] = (x, y)
    obj['vel'] = (vx, vy)


def resolve_collisions(objs, radius):
    """Simple equal-mass elastic collision + separation for overlapping spheres."""
    n = len(objs)
    min_dist = radius * 2
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = objs[i]['pos']
            x2, y2 = objs[j]['pos']
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            if 0 < dist < min_dist:
                nx, ny = dx / dist, dy / dist
                overlap = min_dist - dist
                x1 -= nx * overlap / 2
                y1 -= ny * overlap / 2
                x2 += nx * overlap / 2
                y2 += ny * overlap / 2
                objs[i]['pos'] = (x1, y1)
                objs[j]['pos'] = (x2, y2)
                v1, v2 = objs[i]['vel'], objs[j]['vel']
                v1n = v1[0] * nx + v1[1] * ny
                v2n = v2[0] * nx + v2[1] * ny
                objs[i]['vel'] = (v1[0] - v1n * nx + v2n * nx, v1[1] - v1n * ny + v2n * ny)
                objs[j]['vel'] = (v2[0] - v2n * nx + v1n * nx, v2[1] - v2n * ny + v1n * ny)

def normalize_speeds(objs):
    """Force every sphere back to its level's constant speed (direction kept),
    so collisions can't make some spheres slow down while others speed up."""
    for o in objs:
        vx, vy = o['vel']
        mag = math.hypot(vx, vy)
        if mag < 1e-6:                       # stalled: pick a fresh direction
            vx, vy = random_velocity(o['speed'])
        else:
            vx, vy = vx * o['speed'] / mag, vy * o['speed'] / mag
        o['vel'] = (vx, vy)

def run_fixation(win, bg, markers):
    fix_dur = random.uniform(*FIX_DUR_RANGE)
    fixation = visual.TextStim(win, text='+', color='white', height=60)
    markers.send('fixation_onset')
    clock = core.Clock()
    while clock.getTime() < fix_dur:
        bg.draw()
        fixation.draw()
        win.flip()
    return fix_dur


def build_objects(level):
    n_targets = level['n_targets']
    n_total = n_targets + N_DISTRACTORS
    speed = SPEED_PX_S[level['speed']]
    region_positions = random_positions(n_total, SPHERE_RADIUS, CURRENT_REGION)
    target_idx = set(random.sample(range(n_total), n_targets))
    objs = []
    for i in range(n_total):
        vx, vy = random_velocity(speed)
        objs.append({
            'pos': region_positions[i],
            'vel': (vx, vy),
            'speed': speed,
            'is_target': i in target_idx,
        })
    return objs


def run_cueing(win, circles, objs, bg, markers):
    for c, o in zip(circles, objs):
        c.pos = o['pos']
        colour = TARGET_CUE_COLOR if o['is_target'] else OBJECT_COLOR
        c.fillColor = colour
        c.lineColor = colour
    markers.send('cue_onset')
    clock = core.Clock()
    while clock.getTime() < CUE_DUR:
        bg.draw()
        for c in circles:
            c.draw()
        win.flip()


def run_tracking(win, circles, objs, bg, markers):
    # right before tracking starts: targets become indistinguishable from distractors
    for c in circles:
        c.fillColor = OBJECT_COLOR
        c.lineColor = OBJECT_COLOR

    markers.send('tracking_onset')
    frame_clock = core.Clock()
    track_clock = core.Clock()
    frame_clock.reset()
    while track_clock.getTime() < TRACK_DUR:
        dt = frame_clock.getTime()
        frame_clock.reset()
        dt = min(dt, 0.05)  # guard against occasional long frames (e.g. first frame)
        for o in objs:
            bounce_and_move(o, dt, CURRENT_REGION, SPHERE_RADIUS)
        resolve_collisions(objs, SPHERE_RADIUS)
        normalize_speeds(objs)
        bg.draw()
        for c, o in zip(circles, objs):
            c.pos = o['pos']
            c.draw()
        win.flip()
        if 'escape' in event.getKeys(['escape']):
            win.close()
            core.quit()

def build_option_sets(all_numbers, correct_set, n_targets, max_tries=3000):
    """
    Build the 4 answer options (1 correct + 3 foils), shuffled.

    Rules:
      - Any two options share at most (n_targets - 2) numbers (min 1),
        i.e. they differ in 2+ numbers -> no near-duplicate options.
      - Every true target appears in at least one wrong option, so a
        participant can't answer correctly by tracking only a subset of
        the targets (keeps the load manipulation meaningful).
    """
    correct = frozenset(correct_set)
    base_overlap = max(n_targets - 2, 1)

    for max_overlap in range(base_overlap, n_targets):   # relax only if needed
        for _ in range(max_tries):
            foils = []
            for _ in range(3):
                cand = frozenset(random.sample(all_numbers, n_targets))
                if any(len(cand & other) > max_overlap for other in [correct] + foils):
                    break
                foils.append(cand)
            else:
                if correct <= frozenset().union(*foils):
                    options = [correct_set] + [sorted(f) for f in foils]
                    random.shuffle(options)
                    return options

    # fallback (should essentially never be reached): any 3 distinct foils
    options = [correct_set]
    while len(options) < 4:
        cand = sorted(random.sample(all_numbers, n_targets))
        if cand not in options:
            options.append(cand)
    random.shuffle(options)
    return options

def run_response(win, circles, objs, dev=None, bg=None, markers=None):
    # stop motion, assign a unique number to every sphere
    numbers = list(range(1, len(objs) + 1))
    random.shuffle(numbers)
    for o, num in zip(objs, numbers):
        o['number'] = num

    number_stims = []
    for c, o in zip(circles, objs):
        c.fillColor = OBJECT_COLOR
        c.lineColor = OBJECT_COLOR
        ns = visual.TextStim(win, text=str(o['number']), pos=o['pos'],
                              color='black', height=SPHERE_RADIUS * 1.5, bold=True)
        number_stims.append(ns)

    n_targets = sum(o['is_target'] for o in objs)
    all_numbers = [o['number'] for o in objs]
    correct_set = sorted(o['number'] for o in objs if o['is_target'])

    option_sets = build_option_sets(all_numbers, correct_set, n_targets)
    correct_letter = RESPONSE_KEYS[option_sets.index(correct_set)]

    def make_foil_set(existing_sets):
        # keep sampling until we get a set of n_targets numbers that is
        # not identical to the correct set or to a foil already chosen
        while True:
            candidate = sorted(random.sample(all_numbers, n_targets))
            if candidate != correct_set and candidate not in existing_sets:
                return candidate

    option_sets = [correct_set]
    for _ in range(3):
        option_sets.append(make_foil_set(option_sets))
    random.shuffle(option_sets)
    correct_letter = RESPONSE_KEYS[option_sets.index(correct_set)]

    # staggered layout: A up, B down, C up, D down
    xpos = [-320, -100, 100, 320]
    y_stagger = [30, -30, 30, -30]
    base_y = -370
    option_stims = []
    for letter, num_set, x, ys in zip(['A', 'B', 'C', 'D'], option_sets, xpos, y_stagger):
        label = ', '.join(str(n) for n in num_set)
        ts = visual.TextStim(win, text=f'{letter}: {label}', pos=(x, base_y + ys),
                              color='white', height=32, wrapWidth=220)
        option_stims.append(ts)

    event.clearEvents()
    if dev is not None:
        dev.poll_for_response()
        dev.clear_response_queue()
        dev.reset_timer()

    resp_clock = core.Clock()
    response, rt = None, None
    markers.send('response_onset')
    while resp_clock.getTime() < RESPONSE_TIMEOUT:
        bg.draw()
        for c in circles:
            c.draw()
        for ns in number_stims:
            ns.draw()
        for ts in option_stims:
            ts.draw()
        win.flip()

        if 'escape' in event.getKeys(['escape']):  # keyboard escape always works to abort
            win.close()
            core.quit()

        if dev is not None:
            dev.poll_for_response()
            if dev.has_response():
                r = dev.get_next_response()
                if r['pressed'] and r['key'] in RESPONSE_KEY_MAP:
                    response = RESPONSE_KEY_MAP[r['key']]
                    rt = r['time'] / 1000.0   # pyxid2 reports ms -> convert to seconds
                    markers.send('key_press')
                    dev.clear_response_queue()
                    break
                dev.clear_response_queue()
        else:
            keys = event.getKeys(keyList=RESPONSE_KEYS, timeStamped=resp_clock)
            if keys:
                response, rt = keys[0]
                markers.send('key_press')
                break

    if dev is not None:
        dev.flush_serial_buffer()
        dev.clear_response_queue()

    correct = (response == correct_letter)
    return response, rt, correct, correct_letter, option_sets

def run_confidence(win, bg, dev=None, markers=None):
    """
    Collect a 1-5 confidence rating shown as five red squares, from light
    red (1 = Low) to dark red (5 = High), using the Riponda response pad
    or keyboard number keys ('1'-'5') as fallback.

    Returns:
        confidence (int or None)
        confidence_rt (float or None)
    """

    question = visual.TextStim(
        win,
        text="How confident are you about your answer?",
        color='white',
        height=40,
        pos=(0, 220)
    )

    # light red -> dark red
    square_colors = ['#FFD6D6', '#FF9999', '#FF5555', '#CC0000', '#800000']
    # dark text on the light squares, white text on the dark squares
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
                lineWidth=2
            )
        )
        numbers.append(
            visual.TextStim(
                win,
                text=str(i + 1),
                color=number_colors[i],
                height=50,
                bold=True,
                pos=(x, square_y)
            )
        )

    label_y = square_y - square_size / 2 - 40
    low_label = visual.TextStim(
        win, text="Low", color='white', height=32,
        pos=(x_positions[0], label_y)
    )
    high_label = visual.TextStim(
        win, text="High", color='white', height=32,
        pos=(x_positions[-1], label_y)
    )

    prompt_text = (
        "Press 1-5 on the response pad" if dev is not None
        else "Press 1-5 on the keyboard"
    )
    instruction = visual.TextStim(
        win, text=prompt_text, color='white', height=28, pos=(0, -220)
    )

    event.clearEvents()

    # Clear anything left over from the previous A-D response
    if dev is not None:
        dev.poll_for_response()
        dev.clear_response_queue()
        dev.reset_timer()

    markers.send('confidence_onset')

    confidence = None
    confidence_rt = None
    clock = core.Clock()

    while clock.getTime() < CONFIDENCE_TIMEOUT:

        bg.draw()
        question.draw()
        for sq, num in zip(squares, numbers):
            sq.draw()
            num.draw()
        low_label.draw()
        high_label.draw()
        instruction.draw()
        win.flip()

        # Keyboard escape always remains available
        if 'escape' in event.getKeys(['escape']):
            win.close()
            core.quit()

        if dev is not None:
            dev.poll_for_response()
            if dev.has_response():
                r = dev.get_next_response()
                if r['pressed'] and r['key'] in CONFIDENCE_KEY_MAP:
                    confidence = CONFIDENCE_KEY_MAP[r['key']]
                    confidence_rt = r['time'] / 1000.0
                    markers.send('confidence_response')
                    dev.clear_response_queue()
                    break
                dev.clear_response_queue()
        else:
            keys = event.getKeys(keyList=['1', '2', '3', '4', '5'], timeStamped=clock)
            if keys:
                key_pressed, confidence_rt = keys[0]
                confidence = int(key_pressed)
                markers.send('confidence_response')
                break

    # Clean up any leftover response-pad events
    if dev is not None:
        dev.flush_serial_buffer()
        dev.clear_response_queue()

    return confidence, confidence_rt

def run_feedback(win, correct, bg, markers):
    fb_dur = random.uniform(*FEEDBACK_DUR_RANGE)

    face_color = 'yellow'
    head = visual.Circle(win, radius=50, fillColor=face_color, lineColor='black', lineWidth=3)
    left_eye = visual.Circle(win, radius=6, fillColor='black', pos=(-18, 15))
    right_eye = visual.Circle(win, radius=6, fillColor='black', pos=(18, 15))

    # mouth: a curved line, shaped differently for correct vs incorrect
    n_pts = 20
    width, depth = 50, 10
    pts = []
    for i in range(n_pts):
        x = -width / 2 + width * i / (n_pts - 1)
        t = (x / (width / 2)) ** 2
        y = (depth * t - depth) if correct else (-depth * t + depth)
        pts.append((x, y - 10))
    mouth = visual.ShapeStim(win, vertices=pts, closeShape=False,
                              lineColor='black', lineWidth=4)

    markers.send('feedback_correct' if correct else 'feedback_incorrect')
    clock = core.Clock()
    while clock.getTime() < fb_dur:
        bg.draw()
        head.draw()
        left_eye.draw()
        right_eye.draw()
        mouth.draw()
        win.flip()


# ---------------------------------------------------------------------------
# 3. TRIAL RUNNER
# ---------------------------------------------------------------------------

def run_trial(win, level, trial_num, writer, dev=None, bg=None, markers=None):
    objs = build_objects(level)

    circles = [
        visual.Circle(
            win,
            radius=SPHERE_RADIUS,
            pos=o['pos'],
            edges=64,
            fillColor=OBJECT_COLOR,
            lineColor=OBJECT_COLOR
        )
        for o in objs
    ]

    # 1. Fixation
    fix_dur = run_fixation(win, bg, markers)

    # 2. Cue
    run_cueing(win, circles, objs, bg, markers)

    # 3. Tracking
    run_tracking(win, circles, objs, bg, markers)

    response, rt, correct, correct_letter, option_sets = run_response(
    win, circles, objs, dev, bg, markers
    )

    confidence, confidence_rt = run_confidence(
    win, bg, dev, markers
    )

    fb_dur = run_feedback(win, correct, bg, markers)

    writer.writerow([
        level['label'], level['n_targets'], level['speed'], trial_num,
        round(fix_dur, 4), CUE_DUR, TRACK_DUR,
        response if response else 'NA',
        round(rt, 4) if rt is not None else 'NA',
        correct_letter,
        int(correct),
        option_sets,
        confidence if confidence is not None else 'NA',
        round(confidence_rt, 4) if confidence_rt is not None else 'NA'
    ])


# ---------------------------------------------------------------------------
# 4. MAIN EXPERIMENT
# ---------------------------------------------------------------------------

def main():
    global CURRENT_REGION

    exp_info = {'participant': '', 'session': '001'}
    dlg = gui.DlgFromDict(exp_info, title='MOT Task')
    if not dlg.OK:
        core.quit()

    os.makedirs(DATA_DIR, exist_ok=True)

    fname = os.path.join(
        DATA_DIR,
        f"{exp_info['participant']}_{exp_info['session']}_{data.getDateStr()}.csv"
    )

    f = open(fname, 'w', newline='')
    writer = csv.writer(f)

    writer.writerow([
        'level', 'n_targets', 'speed', 'trial',
        'fixation_dur', 'cue_dur', 'track_dur',
        'response', 'rt', 'correct_letter', 'correct', 'option_sets',
        'confidence', 'confidence_rt'
    ])

    # ---------------------------------------------------------------
    # Create PsychoPy window
    # ---------------------------------------------------------------

    if FULLSCREEN:
        win = visual.Window(
            color=BG_COLOR,
            units='pix',
            fullscr=True,
            screen=0,
            allowGUI=False
        )
    else:
        win = visual.Window(
            size=WIN_SIZE,
            color=BG_COLOR,
            units='pix',
            fullscr=False,
            allowGUI=False
        )

    CURRENT_REGION = get_region(win)

    # ---------------------------------------------------------------
    # ioHub
    # ---------------------------------------------------------------
    # ioHub is only a local backup log of marker events.
    # It does NOT replace the EEG trigger or LSL marker stream.

    io = None

    if USE_IOHUB_LOG:
        io_fname = os.path.join(
            DATA_DIR,
            f"{exp_info['participant']}_{exp_info['session']}_{data.getDateStr()}"
        )

        io = launchHubServer(
            window=win,
            datastore_name=io_fname
        )

    # ---------------------------------------------------------------
    # Background image
    # ---------------------------------------------------------------

    bg = visual.ImageStim(
        win,
        image=BACKGROUND_IMAGE_PATH,
        size=win.size,
        pos=(0, 0)
    )

    # ---------------------------------------------------------------
    # Discover XID devices ONCE
    # ---------------------------------------------------------------
    #
    # Riponda Model L:
    #     product_id = b'5'
    #
    # StimTracker Quad:
    #     product_id = b'S'
    #
    # We use product_id to identify them rather than doing another
    # pyxid2.get_xid_devices() scan inside EEGTrigger.
    # ---------------------------------------------------------------

    dev = None          # Riponda response pad
    stim_dev = None     # Cedrus StimTracker Quad

    if USE_RESPONSE_PAD or USE_EEG_TRIGGERS:

        xid_devices = pyxid2.get_xid_devices()

        if not xid_devices:
            raise RuntimeError(
                "No Cedrus XID devices detected -- check USB connections."
            )

        # -----------------------------------------------------------
        # Find Riponda by product_id
        # -----------------------------------------------------------

        riponda_devices = [
            d for d in xid_devices
            if getattr(d, 'product_id', None) == RIPONDA_PRODUCT_ID
        ]

        # -----------------------------------------------------------
        # Find StimTracker by product_id
        # -----------------------------------------------------------

        stimtracker_devices = [
            d for d in xid_devices
            if getattr(d, 'product_id', None) == STIMTRACKER_PRODUCT_ID
        ]

        # -----------------------------------------------------------
        # Select Riponda response pad
        # -----------------------------------------------------------

        if USE_RESPONSE_PAD:

            if not riponda_devices:
                raise RuntimeError(
                    "USE_RESPONSE_PAD is True, but no Riponda Model L "
                    f"was found (product_id={RIPONDA_PRODUCT_ID!r})."
                )

            if len(riponda_devices) > 1:
                raise RuntimeError(
                    f"Expected one Riponda Model L, but found "
                    f"{len(riponda_devices)} devices."
                )

            dev = riponda_devices[0]

        # -----------------------------------------------------------
        # Select StimTracker
        # -----------------------------------------------------------

        if USE_EEG_TRIGGERS:

            if not stimtracker_devices:
                raise RuntimeError(
                    "USE_EEG_TRIGGERS is True, but no Cedrus "
                    "StimTracker Quad was found "
                    f"(product_id={STIMTRACKER_PRODUCT_ID!r})."
                )

            if len(stimtracker_devices) > 1:
                raise RuntimeError(
                    f"Expected one StimTracker Quad, but found "
                    f"{len(stimtracker_devices)} devices."
                )

            stim_dev = stimtracker_devices[0]

        # -----------------------------------------------------------
        # Print selected devices
        # -----------------------------------------------------------

        print("\nXID devices selected:")

        if dev is not None:
            print(
                "  Response pad:",
                dev.device_name,
                "| product_id:",
                dev.product_id
            )

        if stim_dev is not None:
            print(
                "  StimTracker:",
                stim_dev.device_name,
                "| product_id:",
                stim_dev.product_id
            )

        print()

    # ---------------------------------------------------------------
    # Create marker systems
    # ---------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # EEGTrigger receives the already-discovered StimTracker device.
    # Therefore EEGTrigger does NOT perform another
    # pyxid2.get_xid_devices() scan.
    #
    # LSL creates its own separate LSL marker stream.
    #
    # ioHub creates its own local marker log.
    #
    # MarkerBroadcaster sends the SAME marker code to all enabled
    # systems.
    # ---------------------------------------------------------------

    eeg = (
        EEGTrigger(dev=stim_dev)
        if USE_EEG_TRIGGERS
        else None
    )

    lsl = (
        LSLMarkers()
        if USE_LSL_MARKERS
        else None
    )

    markers = MarkerBroadcaster(
        io=io,
        eeg=eeg,
        lsl=lsl
    )

    # ---------------------------------------------------------------
    # Title screen (experimenter-controlled: KEYBOARD ONLY)
    # ---------------------------------------------------------------

    title_text = visual.TextStim(
        win,
        text="Multiple Object Tracking Task",
        color='white',
        height=50,
        bold=True,
        pos=(0, 60)
    )

    wait_text = visual.TextStim(
        win,
        text="Please wait for the instructions",
        color='white',
        height=36,
        pos=(0, -85)
    )

    bg.draw()
    title_text.draw()
    wait_text.draw()
    win.flip()

    # waitKeys() only listens to the keyboard, so the Riponda is ignored here
    event.clearEvents()
    keys = event.waitKeys()
    if 'escape' in keys:
        win.close()
        core.quit()

    # ---------------------------------------------------------------
    # Instructions (participant starts the task with the Riponda)
    # ---------------------------------------------------------------

    begin_prompt = (
        "Press the green button on the response pad to begin."
        if dev is not None
        else "Press any key to begin."
    )

    instructions = visual.TextStim(
        win,
        text=(
            "You will see several spheres.\n\n"
            "Some will briefly turn a different colour -- these are "
            "your TARGETS.\n"
            "Track them with your eyes as all spheres move and change "
            "back to the same colour.\n\n"
            "When they stop, numbers will appear and 4 options (A-D) "
            "will be shown.\n"
            "Press the button for the option listing your targets.\n\n"
            + begin_prompt
        ),
        color='white',
        height=34,
        wrapWidth=900
    )

    bg.draw()
    instructions.draw()
    win.flip()

    # ---------------------------------------------------------------
    # Wait for the participant's start press
    #   - response pad connected: ONLY the Riponda starts the task
    #   - no response pad (keyboard fallback): any key starts it
    #   - Escape on the keyboard always aborts
    # ---------------------------------------------------------------

    event.clearEvents()
    if dev is not None:
        dev.poll_for_response()
        dev.clear_response_queue()

    waiting = True
    while waiting:
        keys = event.getKeys()

        if 'escape' in keys:
            win.close()
            core.quit()

        if dev is None:
            if keys:
                waiting = False
        else:
            dev.poll_for_response()
            if dev.has_response():
                r = dev.get_next_response()
                if r['pressed']:
                    waiting = False
                dev.clear_response_queue()

    if dev is not None:
        dev.flush_serial_buffer()
        dev.clear_response_queue()

    # ---------------------------------------------------------------
    # Build and shuffle trial list
    # ---------------------------------------------------------------

    trial_list = []

    for level in LEVELS:
        for trial_num in range(
            1,
            N_TRIALS_PER_LEVEL + 1
        ):
            trial_list.append(
                (level, trial_num)
            )

    random.shuffle(trial_list)

    # ---------------------------------------------------------------
    # Start experiment
    # ---------------------------------------------------------------

    markers.send('task_start')

    try:

        for level, trial_num in trial_list:

            run_trial(
                win,
                level,
                trial_num,
                writer,
                dev,
                bg,
                markers
            )

    finally:

        # -----------------------------------------------------------
        # End experiment
        # -----------------------------------------------------------

        markers.send('task_end')

        # Close EEG + LSL marker systems
        markers.close()

        # Close ioHub
        if io is not None:
            io.quit()

        # Close CSV
        f.close()

        # Close PsychoPy window
        win.close()

    core.quit()


if __name__ == '__main__':
    main()