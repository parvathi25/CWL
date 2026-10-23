"""
MSPL.py
=======
Multisensory Perceptual Load Task (Main Script).
Updated with confidence ratings, focus checks, CSV logging,
and optional Cedrus Riponda response pad input.

Change sizes come from each participant's 50% hit-rate thresholds
(measured in threshold_block.py) and are entered at the start:
  - visual   : % coherent dots (rounded to a whole number, since there are 100 dots)
  - auditory : % amplitude increase (any value)
  - touch    : record only. The haptic tracks are edited per participant in
               Hapticlabs, so this value is NOT used to pick a track.
"""

from psychopy import visual, core, data, gui
from psychopy.hardware import keyboard
import random
import itertools
import os
import csv

# ---- Response input ----
USE_RESPONSE_PAD = True    # False = participant answers on the keyboard (1-4) instead

# ---- Marker outputs ----
USE_EEG_TRIGGERS = False   # Enable StimTracker TTL markers
USE_LSL_MARKERS = False    # Enable LSL marker stream

from common_task import (
    MODALITIES, DATA_DIR, QUIT_KEY,
    STIMTRACKER_PRODUCT_ID,
    RDKField, AuditoryStimulus, HapticsController,
    make_background, run_instructions,
    run_presentation, stop_all_stimuli,
    run_response_two_screen, run_confidence_screen, run_focus_screen,
    run_feedback, run_iti, cleanup_and_quit,
    send_event_marker,
    ResponsePadInput, find_xid_device, UNSURE_LABEL,
)

INSTRUCTIONS_TEXT = (
    "Welcome to the task!\n\n"
    "In each trial, you will first feel/see/hear baseline stimuli.\n"
    "Then, some of the stimuli may change (get faster / louder / stronger).\n\n"
    "After the stimuli, you will be asked:\n"
    "  Screen 1: How many stimuli changed? (0, 1, 2, or 3)\n"
    "            Press the yellow button if you are unsure.\n"
    "  Screen 2: If 1 or 2 changed, which ones changed?\n\n"
    "Once you press a button, your answer is locked in and cannot be changed.\n\n"
    "{BEGIN}"
)

N_TRIALS_PER_LEVEL = 10
LEVELS = [1, 2, 3, 4, 5]

# Placeholder values shown in the threshold dialog (replace with the participant's own).
DEFAULT_VISUAL_PCT = 10.0
DEFAULT_AUDIO_PCT = 10.0
DEFAULT_TOUCH_PCT = 20.0

LEVEL_DEFINITIONS = {
    0: (0, 'none'),
    1: (1, 'single'),
    2: (2, 'sequential'),
    3: (2, 'simultaneous'),
    4: (3, 'sequential'),
    5: (3, 'simultaneous'),
}

DISPLAY_NAMES = {'visual': 'Visual', 'auditory': 'Audio', 'touch': 'Tactile'}


def describe_trial(i, trial):
    """Console label, e.g. 'Trial 3 | Level 4 | Visual, Audio, Tactile'."""
    sep = ', ' if trial['timing_mode'] == 'sequential' else '+'
    names = sep.join(DISPLAY_NAMES[m] for m in trial['modalities_changed']) or 'None'
    return f"Trial {i + 1} | Level {trial['level']} | {names}"


def get_participant_thresholds():
    """
    Ask for each modality's 50% hit-rate change size (in %).
    Visual is rounded to a whole number (100 dots -> whole-dot steps).
    Touch is stored for the data file only; tracks are set in Hapticlabs.
    """
    threshold_info = {
        'visual_threshold_% (coherence)': DEFAULT_VISUAL_PCT,
        'auditory_threshold_% (amplitude)': DEFAULT_AUDIO_PCT,
        'touch_threshold_% (record only)': DEFAULT_TOUCH_PCT,
    }
    dlg = gui.DlgFromDict(threshold_info, title='Participant Thresholds (50% hit rate)')
    if not dlg.OK:
        core.quit()

    visual_pct = float(threshold_info['visual_threshold_% (coherence)'])
    audio_pct = float(threshold_info['auditory_threshold_% (amplitude)'])
    touch_pct = float(threshold_info['touch_threshold_% (record only)'])

    for name, v in (('visual', visual_pct), ('auditory', audio_pct)):
        if not 0 < v <= 100:
            raise ValueError(f"{name} threshold must be between 0 and 100 (got {v}).")

    return {
        'visual': int(round(visual_pct)),
        'auditory': audio_pct,
        'touch': touch_pct,
    }


def build_trial_list(thresholds, n_per_level=N_TRIALS_PER_LEVEL):
    trials = []
    for level in LEVELS:
        n_changing, timing_mode = LEVEL_DEFINITIONS[level]
        combos = list(itertools.combinations(MODALITIES, n_changing)) if n_changing > 0 else [()]
        for _ in range(n_per_level):
            modalities_changed = random.choice(combos)
            percents = {m: thresholds[m] for m in modalities_changed}
            trials.append({
                'level': level,
                'n_changing': n_changing,
                'timing_mode': timing_mode,
                'modalities_changed': modalities_changed,
                'percents': percents,
            })
    random.shuffle(trials)
    return trials


def score_response(trial, response):
    true_n = trial['n_changing']
    true_set = set(trial['modalities_changed'])

    # "Unsure" is not a correct answer: screen 2 is skipped and the trial is
    # scored incorrect. It is saved as reported_n = 'unsure' for separate analysis.
    if response['n_reported'] == UNSURE_LABEL:
        return False, False, False

    screen1_correct = (response['n_reported'] == true_n)

    if true_n in (0, 3) or response['n_reported'] in (0, 3):
        screen2_correct = True
    else:
        screen2_correct = (response['modalities_reported'] == true_set)

    return screen1_correct, screen2_correct, (screen1_correct and screen2_correct)


def initialize_marker_outputs():
    """Initialize only the marker outputs enabled in the configuration."""

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
            stream_name="MSPL_Task_Markers"
        )

    return eeg_trigger, lsl_markers


def show_title_screen(win, kb):
    """Display the task title until a keyboard key is pressed (experimenter-controlled)."""

    title = visual.TextStim(
        win,
        text="Multisensory Task",
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

        if any(k.name == QUIT_KEY for k in keys):
            raise KeyboardInterrupt

        if keys:
            break

    # Prevent the title-screen key from carrying over.
    kb.clearEvents()


def main():
    exp_info = {'participant': '', 'session': '001'}
    dlg = gui.DlgFromDict(exp_info, title='Multisensory Perceptual Load Task')
    if not dlg.OK:
        core.quit()

    thresholds = get_participant_thresholds()
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(
        DATA_DIR, f"sub-{exp_info['participant']}_ses-{exp_info['session']}_{data.getDateStr()}.csv"
    )

    # Initialize only the enabled marker outputs.
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
        cleanup_and_quit(
            win, audio, haptics,
            eeg_trigger, lsl_markers,
        )

    trials = build_trial_list(thresholds)

    fieldnames = [
        'trial_num', 'level', 'timing_mode',
        'correct_n', 'reported_n',
        'correct_modalities', 'reported_modalities',
        'percents_used',
        'rt_screen1', 'rt_screen2',
        'confidence', 'rt_confidence',
        'modality_focus', 'rt_focus',
        'screen1_correct', 'screen2_correct', 'overall_correct'
    ]

    with open(filename, 'w', newline='') as f:
        f.write(f"# thresholds used (50% hit rate): {thresholds}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Marker 100: task start, after instructions and before trial 1.
        send_event_marker(100, eeg_trigger, lsl_markers)

        try:
            for i, trial in enumerate(trials):
                print(describe_trial(i, trial))

                # 1. Presentation
                run_presentation(
                    win, dots, audio, haptics, trial, background,
                    eeg_trigger=eeg_trigger,
                    lsl_markers=lsl_markers,
                )
                stop_all_stimuli(dots, audio, haptics)

                # 2. Main Stimulus Response (response pad)
                response = run_response_two_screen(
                    win, pad, background,
                    eeg_trigger=eeg_trigger,
                    lsl_markers=lsl_markers,
                )
                s1_correct, s2_correct, overall_correct = score_response(trial, response)

                # 3. Confidence rating screen (1-5)
                if response['n_reported'] == UNSURE_LABEL:
                    confidence, rt_confidence = None, None
                else:
                    confidence, rt_confidence = run_confidence_screen(
                        win, pad, background,
                        eeg_trigger=eeg_trigger,
                        lsl_markers=lsl_markers,
                    )

                # 4. Modality focus screen
                modality_focus, rt_focus = run_focus_screen(win, pad, background)

                # 5. Feedback & ITI
                run_feedback(
                    win, overall_correct, background,
                    eeg_trigger=eeg_trigger,
                    lsl_markers=lsl_markers,
                )

                run_iti(
                    win, background,
                    eeg_trigger=eeg_trigger,
                    lsl_markers=lsl_markers,
                )

                # Format string variables before writing row
                percents_str = '+'.join(f'{m}:{trial["percents"][m]}' for m in trial['modalities_changed'])
                correct_mods_str = '+'.join(sorted(trial['modalities_changed']))
                reported_mods_str = '+'.join(sorted(response['modalities_reported'])) if response['modalities_reported'] else 'none'

                # 6. Write data row to CSV
                writer.writerow({
                    'trial_num': i + 1,
                    'level': trial['level'],
                    'timing_mode': trial['timing_mode'],
                    'correct_n': trial['n_changing'],
                    'reported_n': response['n_reported'],
                    'correct_modalities': correct_mods_str if correct_mods_str else 'none',
                    'reported_modalities': reported_mods_str,
                    'percents_used': percents_str,
                    'rt_screen1': response['rt_screen1'],
                    'rt_screen2': response['rt_screen2'],
                    'confidence': confidence if confidence is not None else ('skipped' if response['n_reported'] == UNSURE_LABEL else 'timeout'),
                    'rt_confidence': rt_confidence if rt_confidence is not None else ('skipped' if response['n_reported'] == UNSURE_LABEL else 'timeout'),
                    'modality_focus': modality_focus if modality_focus is not None else 'timeout',
                    'rt_focus': rt_focus if rt_focus is not None else 'timeout',
                    'screen1_correct': s1_correct,
                    'screen2_correct': s2_correct,
                    'overall_correct': overall_correct,
                })
                f.flush()

        except KeyboardInterrupt:
            print('Experiment aborted by user.')

        finally:
            # Marker 110: task end, including when the user aborts the task.
            send_event_marker(110, eeg_trigger, lsl_markers)

            cleanup_and_quit(
                win, audio, haptics,
                eeg_trigger, lsl_markers,
            )


if __name__ == '__main__':
    main()