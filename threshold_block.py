"""
threshold_block.py
===================
Threshold-finding ("screening") block, yes/no change detection.

- One modality per trial (visual / auditory / touch); the others stay inactive.
- Each trial: 1000 ms baseline, then 950 ms change (no jitter).
- Each modality has its own list of change levels (see common_task.LEVELS_BY_MODALITY).
- 20% of all trials are catch trials (no change), split equally across modalities;
  change trials are split equally across modalities and levels.
- Participant answers "Did the stimulus change?" (0 = No, 1 = Yes).
- Threshold = level where the hit rate is 50% (fit the curve afterwards).
Supports optional Cedrus Riponda response pad input.
"""

from psychopy import visual, core, event, data, gui
from psychopy.hardware import keyboard
import random
import os
import csv
import traceback

from common_task import (
    LEVELS_BY_MODALITY, DATA_DIR, QUIT_KEY,
    THRESH_BASELINE_DURATION, THRESH_CHANGE_DURATION,
    HAPTIC_THRESHOLD_TRACKS, HAPTICS_BASELINE_TRACK,
    STIMTRACKER_PRODUCT_ID,
    RDKField, AuditoryStimulus, HapticsController,
    make_background, run_instructions,
    stop_all_stimuli,
    run_response_detection, run_feedback, run_iti,
    cleanup_and_quit, send_event_marker,
    ResponsePadInput, find_xid_device,
)

# ---- Response input ----
USE_RESPONSE_PAD = True    # False = participant answers on the keyboard instead

# ---- Marker outputs ----
USE_EEG_TRIGGERS = False
USE_LSL_MARKERS = False

# ---- Trial structure ----
N_REPS_PER_LEVEL = 8      # change trials per level per modality (multiple of 4 -> catch is exactly 20%)
CATCH_FRACTION = 0.20      # share of the whole block that is catch trials

INSTRUCTIONS_TEXT = (
    "Threshold-finding task\n\n"
    "In each trial, you will test ONE sense at a time (Visual, Auditory, or Tactile).\n"
    "Sometimes the stimulus will change slightly (or stay the same).\n\n"
    "After the stimulus presentation, you will be asked:\n"
    "  Did the stimulus change?   0 = No     1 = Yes\n\n"
    "{BEGIN}"
)


def show_title_screen(win, kb):
    """Display the task title until a keyboard key is pressed (experimenter-controlled)."""
    title = visual.TextStim(
        win,
        text="Threshold-Finding Task",
        color="white",
        height=0.09,
        pos=(0, 0.10),
        alignText="center",
        anchorHoriz="center",
        anchorVert="center",
        wrapWidth=1.5,
    )

    subtitle = visual.TextStim(
        win,
        text="Please wait for the instructions",
        color="white",
        height=0.045,
        pos=(0, -0.06),
        alignText="center",
        anchorHoriz="center",
        anchorVert="center",
        wrapWidth=1.5,
    )

    kb.clearEvents()

    while True:
        title.draw()
        subtitle.draw()
        win.flip()

        keys = kb.getKeys(waitRelease=False, clear=True)

        if any(key.name == QUIT_KEY for key in keys):
            raise KeyboardInterrupt

        if keys:
            break

    # Prevent the title-screen key from carrying over.
    kb.clearEvents()


def build_trial_list(n_reps=N_REPS_PER_LEVEL, catch_fraction=CATCH_FRACTION):
    """
    For each modality: n_reps trials at every change level, plus enough catch
    trials (change_pct = 0) that catch trials make up `catch_fraction` of the block.
    """
    trials = []
    for modality, levels in LEVELS_BY_MODALITY.items():
        for pct in levels:
            for _ in range(n_reps):
                trials.append({'modality': modality, 'change_pct': pct})

        n_change = len(levels) * n_reps
        n_catch = int(round(n_change * catch_fraction / (1 - catch_fraction)))
        for _ in range(n_catch):
            trials.append({'modality': modality, 'change_pct': 0})

    random.shuffle(trials)
    return trials


def _draw_frame(win, background, dots, show_dots):
    background.draw()
    if show_dots:
        dots.draw()
    win.flip()
    if event.getKeys([QUIT_KEY]):
        raise KeyboardInterrupt


def run_unimodal_presentation(
    win, dots, audio, haptics, trial, background,
    eeg_trigger=None, lsl_markers=None
):
    """
    One active modality only; the others stay silent/inactive.
    t = 0 to 1.000 s : baseline
    t = 1.000 to 1.950 s : change (or continued baseline on catch trials)
    Markers: 102 onset, 103 visual change, 104 auditory change, 105 touch change.
    """
    modality = trial['modality']
    pct = trial['change_pct']
    is_change = pct > 0

    t_change = THRESH_BASELINE_DURATION
    t_total = THRESH_BASELINE_DURATION + THRESH_CHANGE_DURATION
    show_dots = (modality == 'visual')

    dots.set_baseline()
    background.draw()
    if show_dots:
        dots.draw()

    clock = core.Clock()

    # Onset at t = 0 (only the active modality)
    if modality == 'auditory':
        audio.start_trial(
            t_total,
            change_pct=pct if is_change else None,
            change_at=t_change,
        )
    elif modality == 'touch':
        # Single combined track: baseline for 1000 ms, then change for 950 ms.
        track = HAPTIC_THRESHOLD_TRACKS[pct] if is_change else HAPTICS_BASELINE_TRACK
        haptics.start_track(track)

    win.callOnFlip(clock.reset)
    win.callOnFlip(send_event_marker, 102, eeg_trigger, lsl_markers)
    win.flip()

    # Baseline phase
    while clock.getTime() < t_change:
        _draw_frame(win, background, dots, show_dots)

    # Change phase
    if is_change:
        if modality == 'visual':
            dots.set_change(pct)
            win.callOnFlip(send_event_marker, 103, eeg_trigger, lsl_markers)
        elif modality == 'auditory':
            send_event_marker(104, eeg_trigger, lsl_markers)
        else:
            send_event_marker(105, eeg_trigger, lsl_markers)

    while clock.getTime() < t_total:
        _draw_frame(win, background, dots, show_dots)


def score_detection(trial, response):
    """Correct only if the yes/no answer matches whether a change occurred."""
    if response['reported'] is None:      # timed out
        return False
    return (response['reported'] == 1) == (trial['change_pct'] > 0)


def initialize_marker_outputs():
    """Initialize only the enabled marker outputs."""

    eeg_trigger = None
    lsl_markers = None

    # ---- StimTracker TTL ----
    if USE_EEG_TRIGGERS:
        from eeg_triggers import EEGTrigger

        # Uses the shared, cached XID device list so the serial ports
        # are never opened twice (the Riponda shares the same scan).
        stimtracker = find_xid_device(STIMTRACKER_PRODUCT_ID)

        if stimtracker is None:
            raise RuntimeError(
                "StimTracker is enabled, but no device matched "
                f"product ID {STIMTRACKER_PRODUCT_ID!r}. Check the USB connection."
            )

        eeg_trigger = EEGTrigger(stimtracker)

    # ---- LSL ----
    if USE_LSL_MARKERS:
        from lsl_markers import LSLMarkers
        lsl_markers = LSLMarkers(
            stream_name="Threshold_Task_Markers"
        )

    return eeg_trigger, lsl_markers


def main():
    exp_info = {'participant': '', 'session': '001'}
    dlg = gui.DlgFromDict(exp_info, title='Threshold-Finding Block')
    if not dlg.OK:
        core.quit()

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(
        DATA_DIR,
        f"sub-{exp_info['participant']}_ses-{exp_info['session']}_threshold_{data.getDateStr()}.csv"
    )

    eeg_trigger, lsl_markers = initialize_marker_outputs()

    win = visual.Window(fullscr=True, color='black', units='norm', screen=1)
    kb = keyboard.Keyboard()

    # Participant input: Riponda if USE_RESPONSE_PAD, otherwise keyboard.
    # Raises RuntimeError if the pad is requested but not found.
    pad = ResponsePadInput(kb, use_pad=USE_RESPONSE_PAD)

    background = make_background(win)

    dots = RDKField(win)
    audio = AuditoryStimulus()
    haptics = HapticsController()

    begin_prompt = (
        "Press the green button on the response pad to begin."
        if USE_RESPONSE_PAD else "Press any key to begin."
    )
    instructions_text = INSTRUCTIONS_TEXT.format(BEGIN=begin_prompt)

    try:
        # Title screen: keyboard only (experimenter).
        show_title_screen(win, kb)
        # Instructions: participant proceeds using the response pad.
        run_instructions(win, pad, background, instructions_text)
    except KeyboardInterrupt:
        print('Quit during title/instructions screen (Escape pressed).')
        cleanup_and_quit(
            win, audio, haptics,
            eeg_trigger, lsl_markers,
        )

    trials = build_trial_list()
    print(f"Threshold block: {len(trials)} trials, "
          f"{sum(t['change_pct'] == 0 for t in trials)} catch.")
    send_event_marker(100, eeg_trigger, lsl_markers)

    fieldnames = [
        'trial_num', 'target_modality', 'level_percent',
        'change_occurred', 'reported_detection',
        'rt', 'detection_correct'
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            for i, trial in enumerate(trials):
                send_event_marker(121, eeg_trigger, lsl_markers)

                run_unimodal_presentation(
                    win, dots, audio, haptics, trial, background,
                    eeg_trigger=eeg_trigger,
                    lsl_markers=lsl_markers,
                )
                stop_all_stimuli(dots, audio, haptics)

                # Markers 106/107 (response) are sent inside run_response_detection.
                response = run_response_detection(
                    win, pad, background, eeg_trigger, lsl_markers)
                detection_correct = score_detection(trial, response)

                writer.writerow({
                    'trial_num': i + 1,
                    'target_modality': trial['modality'],
                    'level_percent': trial['change_pct'],
                    'change_occurred': 1 if trial['change_pct'] > 0 else 0,
                    'reported_detection': response['reported'],
                    'rt': response['rt'],
                    'detection_correct': detection_correct,
                })
                f.flush()

                # Markers 108/109 (feedback) and 101 (ITI) are sent inside these functions.
                run_feedback(win, detection_correct, background, eeg_trigger, lsl_markers)
                run_iti(win, background, eeg_trigger, lsl_markers)

                send_event_marker(122, eeg_trigger, lsl_markers)

        except KeyboardInterrupt:
            print('Threshold block aborted by user.')

        except Exception:
            print('\n===== ERROR DURING TRIALS =====')
            traceback.print_exc()
            print('===============================\n')

        finally:
            send_event_marker(110, eeg_trigger, lsl_markers)

            cleanup_and_quit(
                win, audio, haptics,
                eeg_trigger, lsl_markers,
            )


if __name__ == '__main__':
    main()