#!/usr/bin/env python3
"""
Trial 2: batch-generate CCTV-style rig clips on fal (FLUX 3 Max by default, as in Trial 1).

65 clips: 50 normal-work clips and 15 falls. Normal work outnumbers falls on
purpose, because the main problem in Trials 1 and 2 was false alarms on
workers kneeling, crouching and lying under equipment. The fast get-downs and
near misses are the most valuable clips: they look most like a fall. Clips
51-65 target the Trial 2 false alarms directly: workers cut off by the frame
edge, fast crouches and kneels, bending low, crawling, and crowded scenes.

Generate in steps so you can check quality before spending more:
    1 test clip (--test), then 7 batches (--batch 1 ... --batch 7) of 10 clips,
    the last one 4 clips. Every batch mixes falls and normal work. Batch 1 has
    the hardest new normal-work clips, so problems show up early.

Setup (one time):
    pip install fal-client
    Put your fal API key in the repo's .env file (git ignores it), one line:
        FAL_KEY=your_fal_api_key
    A FAL_KEY environment variable also works. Never put the key in this file.

Usage (from the repo root):
    python scripts/generate_videos_trial2.py --list              # all 65 clips and their batch, no cost
    python scripts/generate_videos_trial2.py --dry-run --batch 1 # batch 1's prompts + cost, no cost
    python scripts/generate_videos_trial2.py --test              # step 1: the single test clip (about $1.70)
    python scripts/generate_videos_trial2.py --batch 1           # step 2: batch 1 (10 clips, about $17)
    python scripts/generate_videos_trial2.py --batch 2           # ...and so on up to --batch 7
    python scripts/generate_videos_trial2.py --only 1 33         # specific clip numbers
    python scripts/generate_videos_trial2.py --kind normal       # only the 50 normal-work clips
    python scripts/generate_videos_trial2.py                     # everything not generated yet
    python scripts/generate_videos_trial2.py --model ltx-fast --only 1   # LTX-2.3 Fast (cheaper)
    Every paid run shows the clip numbers and cost and asks before starting (skip with --yes).

--model picks the video model: flux (default, FLUX 3 Max at 720p, the same
model as the Trial 1 clips, about $1.70 a clip) or ltx-fast (about $0.40 a
clip). Each model saves into its own folder, so clips never mix.

Clips land in data/generated/trial2/<model>/rig_falls/ and .../normal_rig_work/
(a staging folder git ignores), named like fall_t2_03_slip_hose_forward_night.mp4
and normal_t2_33_fast_drop_to_knee.mp4. Watch each one, then copy the good ones
into data/input/rig_falls_t2/ and data/input/normal_rig_work_t2/ (drop any with
physically wrong motion, a slow-motion fall, or a normal clip where someone
actually falls). Clips already on disk are skipped, so rerunning never pays
twice. manifest.json in the same folder records every clip's prompt, seed,
scene and conditions (the scene is used for grouping during training).
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_fal_key() -> str:
    """FAL_KEY from the repo's .env (never committed), else from the environment.

    .env wins, so an old key left in a Windows environment variable (setx)
    cannot override the one in .env.
    """
    key = ""
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("FAL_KEY="):
                key = line.split("=", 1)[1]
    if not key.strip():
        key = os.environ.get("FAL_KEY", "")
    return key.strip().strip('"').strip("'").strip()

# Video models on fal. Prices are per second of video; check fal's pricing page.
MODELS = {
    "flux": {
        "endpoint": "blackforestlabs/flux-3-max/text-to-video",
        "arguments": {"resolution": "720p", "aspect_ratio": "16:9", "generate_audio": False},
        "price_per_second": 0.17,  # regular 720p rate from the Trial 1 script
    },
    "ltx-fast": {
        # LTX-2.3 Fast: 1080p is its lowest resolution; 24 fps samples evenly to our 12.
        "endpoint": "fal-ai/ltx-2.3/text-to-video/fast",
        "arguments": {"resolution": "1080p", "aspect_ratio": "16:9", "fps": 24, "generate_audio": False},
        "price_per_second": 0.04,
    },
}
# Raw clips go to a staging folder that git ignores. Only the clips that pass
# review get copied into data/input/ (which is versioned).
OUT_ROOT = REPO / "data" / "generated" / "trial2"
OUT_DIR = OUT_ROOT / "flux"  # set from --model in main()
FOLDERS = {"fall": "rig_falls", "normal": "normal_rig_work"}
DURATION = 10            # seconds
MAX_PARALLEL = 3         # how many clips run at once; lower to 1 if you hit rate limits

# ---------------------------------------------------------------------------
# Shared prompt pieces
# ---------------------------------------------------------------------------

CONTEXT = (
    "Static security-camera footage from a fixed, non-moving CCTV camera on an offshore oil platform. "
    "The camera is bolted in place: no pan, no tilt, no zoom, no handheld shake, no dolly, no cuts, "
    "no transitions, no slow motion, constant framing for the entire clip. Wide-angle lens, 16:9, "
    "24 fps, real-time playback speed, realistic surveillance look with mild lens distortion and flat "
    "industrial color. No on-screen text, timestamps, or overlays."
)

# Six scenes, each with its own camera height and angle, so the model sees
# several viewpoints. Clips that share a scene are grouped together in training.
SCENES = {
    "main_deck": (
        "Camera mounted 4 meters high, looking down at about 25 degrees. Scene: open steel work deck of an "
        "offshore drilling platform. Yellow-painted steel grating floor, pipe racks, valve manifolds, handrails, "
        "a crane base in mid-ground. Ocean visible past the deck edge railing, horizon in the top third of frame."
    ),
    "pump_skid": (
        "Camera mounted 5 meters high in a corner, looking down at about 35 degrees. Scene: pump skid area on an "
        "offshore platform. Several waist-high pumps and motors on steel skid frames, pipes running at knee and "
        "waist height, a control panel on a post, grey steel deck plate with yellow walkway markings."
    ),
    "pipe_rack": (
        "Camera mounted 3.5 meters high, looking down at about 20 degrees along a walkway. Scene: walkway beside "
        "a long pipe rack on an offshore platform. Bundles of pipes run alongside at waist to head height, with "
        "some pipes only 50 to 80 centimeters above the grating, and hoses coiled on the deck."
    ),
    "wellhead_bay": (
        "Camera mounted 4.5 meters high, looking down at about 30 degrees. Scene: wellhead bay on an offshore "
        "platform. A row of tall valve trees with large handwheels, small drain valves near deck level, "
        "steel grating floor, handrails, gauges mounted at different heights."
    ),
    "stair_ladder": (
        "Camera mounted 4 meters high, looking down at about 30 degrees. Scene: two deck levels on an offshore "
        "platform joined by a short steel staircase with handrails and a fixed vertical ladder, plus a raised "
        "steel maintenance platform about 1.5 meters high with an open gap in its handrail."
    ),
    "drill_floor": (
        "Camera mounted 6 meters high, looking down at about 40 degrees. Scene: drill floor of an offshore rig. "
        "Steel deck around the rotary table, drill pipe stands racked to one side, pipe handling equipment, "
        "toolboxes and chains on the floor, the derrick structure rising out of frame."
    ),
}

LIGHTING = {
    # day
    "overcast": "Flat, diffused daylight from an overcast sky, no harsh shadows, dry deck.",
    "sun": "Bright midday sun with a thin high haze, crisp but moderate shadows under the workers, dry deck.",
    "low_sun": "Late afternoon low sun from the side, warm light with long soft shadows, dry deck.",
    "dawn": "Soft dawn light, low pinkish-grey sky, long diffuse shadows, slightly damp deck.",
    # wet
    "drizzle": "Dim overcast light with light drizzle, wet reflective deck, muted grey-blue color.",
    "mist": "Hazy morning with light sea mist, low contrast, soft grey light, damp glistening deck.",
    "wet": "Overcast daylight just after heavy rain, puddles and a glossy wet deck with reflections.",
    # night
    "night": ("Night scene lit by white industrial floodlights mounted on the structure, strong contrast, dark "
              "sky, deck lit evenly enough that every worker is clearly visible head to boots."),
    "night_rain": ("Night scene lit by white industrial floodlights, light rain visible in the beams, glossy wet "
                   "deck with reflections, every worker clearly visible head to boots."),
}
CONDITION = {"overcast": "day", "sun": "day", "low_sun": "day", "dawn": "day",
             "drizzle": "wet", "mist": "wet", "wet": "wet", "night": "night", "night_rain": "night"}

WORKERS = (
    "Workers wear orange or red flame-resistant coveralls, white or yellow hard hats, safety glasses, gloves, "
    "and steel-toe boots. Every worker stays fully in frame, full body visible head to boots, between 4 and 12 "
    "meters from the camera, never cropped by the frame edge, never fully hidden by equipment for more than a "
    "few frames."
)

# For the frame-edge clips (theme "frame_edge"): real CCTV cameras often see a
# worker cut off by the edge of the picture, and one of the Trial 2 false
# alarms was exactly that, so these clips allow it on purpose.
WORKERS_EDGE = (
    "Workers wear orange or red flame-resistant coveralls, white or yellow hard hats, safety glasses, gloves, "
    "and steel-toe boots. One worker is close to the edge of the frame, so part of his body is cut off by the "
    "frame edge for much of the clip, as often happens on a real CCTV camera. Every other worker stays fully "
    "in frame, full body visible head to boots, between 4 and 12 meters from the camera."
)

FALL_MOVEMENT = (
    "Movement must be physically realistic and uncontrolled: real loss of balance, arms reaching out, hard "
    "contact with the deck, body ending up on the deck and staying down for the rest of the clip. The fall is "
    "fast, in real time: from losing balance to hitting the deck takes well under one second. No slow motion, "
    "no stunt-style rolls, no bracing into a clean push-up position, no bouncing back up. Clip length 10 "
    "seconds, continuous single take. The worker moves normally for the first 3 seconds, and the fall happens "
    "between second 3 and second 5."
)

NORMAL_MOVEMENT = (
    "Nobody falls. Every movement is deliberate, controlled and at natural real-life speed, not slow motion: "
    "workers get down to the deck and back up on purpose, the way experienced crew do during routine "
    "maintenance. Clip length 10 seconds, continuous single take. The main action starts after about second 2 "
    "and is clearly visible."
)

# ---------------------------------------------------------------------------
# The 50 clips: (kind, theme, slug, scene, lighting, scenario)
# ---------------------------------------------------------------------------

CLIPS = [
    # ---- FALLS (15) -------------------------------------------------------
    # Slips (4)
    ("fall", "slip", "slip_grating_backward_wet", "main_deck", "wet",
     "Scenario: one worker walks toward the camera across the grating while a second worker coils a hose about "
     "5 meters away. The walking worker steps onto a slick wet plate, both feet shoot forward, and he falls "
     "backward, landing hard on his back and hips. The other worker looks up and starts walking toward him."),
    ("fall", "slip", "slip_oil_sideways", "pump_skid", "overcast",
     "Scenario: two workers walk between the pumps, crossing the frame from left to right. The second worker "
     "steps on a dark oil sheen beside a pump, his legs slide out sideways, and he drops onto his hip and side "
     "on the deck. The first worker turns around and steps back toward him."),
    ("fall", "slip", "slip_hose_forward_night", "pipe_rack", "night",
     "Scenario: a worker walks away from the camera along the walkway carrying a flashlight. He steps on a wet "
     "coiled hose, his front foot slides forward and out, and he pitches forward onto his hands and chest on "
     "the grating. A second worker further down the walkway turns toward him."),
    ("fall", "slip", "slip_stepping_off_ladder", "stair_ladder", "sun",
     "Scenario: a worker climbs down the fixed ladder facing it while two coworkers talk nearby. As he steps "
     "off the bottom rung onto the deck, his boot slides out and he falls sideways onto his hip and shoulder. "
     "The two coworkers turn toward him, one steps closer."),
    # Trips (3)
    ("fall", "trip", "trip_cable_toward_camera", "main_deck", "sun",
     "Scenario: a worker walks briskly toward the camera holding a clipboard, looking at it. His boot catches "
     "on a black power cable lying across the grating, and he pitches forward toward the camera, arms out, "
     "landing on his hands, chest and knees. Two workers in the background stop and look."),
    ("fall", "trip", "trip_grating_edge_dawn", "wellhead_bay", "dawn",
     "Scenario: a worker walks diagonally across the bay between valve trees. The toe of his boot catches on "
     "a raised grating edge, he stumbles two quick steps with arms thrown out, and goes down hard on his knees "
     "and then flat on his stomach. A coworker at a valve tree turns toward him."),
    ("fall", "trip", "trip_pipe_sling_across", "drill_floor", "overcast",
     "Scenario: four workers on the drill floor. One walks across the frame from right to left, catches his "
     "foot on a lifting sling lying on the deck, and falls forward and slightly sideways onto his shoulder "
     "and chest. The other three stop what they are doing and look over."),
    # Backward falls (3)
    ("fall", "backward", "backward_valve_wheel_gives", "wellhead_bay", "overcast",
     "Scenario: a worker pulls hard on a large stuck valve handwheel with both hands, leaning back. The wheel "
     "suddenly gives, he stumbles backward two steps and falls onto his back and hard hat on the grating. A "
     "second worker nearby turns and steps toward him."),
    ("fall", "backward", "backward_pulling_hose_drizzle", "main_deck", "drizzle",
     "Scenario: a worker walks backward away from a manifold, pulling a heavy hose with both hands. The hose "
     "suddenly comes free, he lurches backward, loses his footing on the wet deck and falls flat on his back. "
     "Another worker at the manifold looks up."),
    ("fall", "backward", "backward_trip_toolbox_night", "drill_floor", "night",
     "Scenario: three workers on the drill floor. One walks backward guiding a swinging chain, his heels catch "
     "a toolbox on the deck behind him, and he falls backward over it onto his back. The other two turn toward "
     "him."),
    # Sideways / loss of balance (2)
    ("fall", "balance", "sudden_collapse_at_panel", "pump_skid", "sun",
     "Scenario: a worker stands at the control panel pressing buttons while another walks past. The worker at "
     "the panel suddenly goes limp and collapses sideways to the deck, without putting his hands out, and lies "
     "still. The passing worker stops and turns to him."),
    ("fall", "balance", "knocked_by_swinging_pipe", "pipe_rack", "low_sun",
     "Scenario: three workers guide a short pipe hanging from a sling. The pipe swings and its end bumps one "
     "worker's shoulder, he loses balance, staggers sideways and falls onto his side on the grating. The other "
     "two steady the pipe and look down at him."),
    # Partly hidden falls (2)
    ("fall", "hidden", "fall_behind_pump_partial", "pump_skid", "overcast",
     "Scenario: three workers in the pump area. One walks along the far side of a waist-high pump, so his legs "
     "are partly hidden but his upper body is visible. He slips and falls sideways beside the pump, his upper "
     "body still visible as he goes down and while lying on the deck. The other two look over."),
    ("fall", "hidden", "fall_between_coworkers", "main_deck", "mist",
     "Scenario: four workers stand in a tight group talking. The worker at the back of the group, furthest from "
     "the camera, slips and falls between two coworkers, briefly passing behind one of them, and ends lying on "
     "the deck still visible between their legs. The group steps back and turns toward him."),
    # From height (1)
    ("fall", "height", "fall_from_platform_night_rain", "stair_ladder", "night_rain",
     "Scenario: a worker on the raised maintenance platform steps backward toward the open gap in the handrail, "
     "loses his footing at the edge and drops about 1.5 meters to the deck, landing on his back and shoulder. "
     "A coworker on the lower deck steps back, then moves toward him."),

    # ---- NORMAL WORK (35) -------------------------------------------------
    # Lying down to work (6)
    ("normal", "lying", "lie_under_pipe_rack", "pipe_rack", "overcast",
     "Scenario: a worker sits down on the grating beside a low pipe, lies back, and slides on his back under "
     "the pipe to inspect a joint with a flashlight, staying there with only his legs and torso visible. A "
     "second worker stands nearby holding a toolbag."),
    ("normal", "lying", "lie_under_skid_wrench_night", "pump_skid", "night",
     "Scenario: a worker lies on his back under the edge of a pump skid, legs sticking out, working with a "
     "wrench above his chest for the whole clip. A second worker kneels beside him and hands him a tool."),
    ("normal", "lying", "lie_on_side_drain_valve", "wellhead_bay", "sun",
     "Scenario: a worker walks to a valve tree, kneels, then lies down on his side on the grating and reaches "
     "under the valve tree to open a small drain valve near deck level, staying lying on his side."),
    ("normal", "lying", "get_down_and_slide_under", "main_deck", "drizzle",
     "Scenario: a worker kneels beside a pipe manifold, puts one hand on the deck, lowers himself onto his "
     "back over about two seconds, and slides under the manifold to look at a fitting."),
    ("normal", "lying", "slide_out_and_stand", "pump_skid", "low_sun",
     "Scenario: a worker lying on his back under a pump skid slides out, sits up, pauses, then stands up and "
     "brushes off his coveralls. Another worker stands nearby reading a gauge."),
    ("normal", "lying", "lie_prone_flashlight", "drill_floor", "overcast",
     "Scenario: a worker lies on his stomach on the drill floor shining a flashlight down through a gap in the "
     "floor plates, then pushes himself up to his knees and stands. Two other workers move pipe in the "
     "background."),
    # Kneeling (6)
    ("normal", "kneeling", "kneel_one_knee_valve", "wellhead_bay", "overcast",
     "Scenario: a worker walks up to a low valve, goes down on one knee, turns the small handwheel several "
     "times, then stands back up and walks on."),
    ("normal", "kneeling", "kneel_both_knees_flange", "main_deck", "sun",
     "Scenario: a worker kneels on both knees at a pipe flange near the deck, tightening bolts with a wrench "
     "for the whole clip. A coworker stands beside him holding a torque chart."),
    ("normal", "kneeling", "kneel_tie_hose_night", "pipe_rack", "night",
     "Scenario: a worker walks along the walkway, kneels on one knee to tie off a loose hose to a pipe support, "
     "then stands up and keeps walking away from the camera."),
    ("normal", "kneeling", "kneel_check_gauge_rain", "pump_skid", "night_rain",
     "Scenario: two workers in rain gear. One kneels in front of a pump to read a low gauge with a flashlight, "
     "the other stands beside him writing on a notepad, then the kneeling worker stands."),
    ("normal", "kneeling", "kneel_group_hatch", "main_deck", "dawn",
     "Scenario: three workers walk to a floor hatch, kneel down around it one after another, and lift the hatch "
     "cover together while staying on their knees."),
    ("normal", "kneeling", "kneel_sit_back_on_heels", "drill_floor", "mist",
     "Scenario: a worker kneels to inspect a chain on the drill floor, sits back on his heels for a moment to "
     "look at it, then leans forward again to work on it."),
    # Crouching / squatting (5)
    ("normal", "crouching", "crouch_pump_gauge", "pump_skid", "sun",
     "Scenario: a worker walks up to a pump, crouches down low to read a gauge near the base, stays crouched "
     "for a few seconds writing a note, then stands."),
    ("normal", "crouching", "squat_pick_up_tools", "main_deck", "overcast",
     "Scenario: two workers walking together. One squats down to pick up several tools from the deck and put "
     "them in a bag, then stands and they walk on."),
    ("normal", "crouching", "crouch_walk_under_low_pipe", "pipe_rack", "wet",
     "Scenario: a worker crouches low and duck-walks under a low pipe crossing the walkway, then straightens up "
     "on the other side and continues walking."),
    ("normal", "crouching", "crouch_group_talk_night", "wellhead_bay", "night",
     "Scenario: two workers crouch side by side in front of a valve tree, pointing at a fitting and talking, "
     "then one stands up while the other stays crouched."),
    ("normal", "crouching", "deep_squat_bucket", "drill_floor", "low_sun",
     "Scenario: a worker squats deeply beside a bucket on the drill floor, wringing out a rag and wiping the "
     "deck, then stands and carries the bucket away."),
    # Fast get-downs (5) - the hardest cases, they look most like a fall
    ("normal", "fast_getdown", "fast_drop_to_knee", "main_deck", "overcast",
     "Scenario: a worker walking briskly sees a bolt rolling across the deck and quickly drops onto one knee in "
     "about half a second to grab it, stays kneeling for a moment, then stands and walks on."),
    ("normal", "fast_getdown", "fast_sit_on_toolbox", "pump_skid", "sun",
     "Scenario: a tired worker walks up to a toolbox and drops down quickly to sit on it, landing heavily, then "
     "leans forward with elbows on his knees. A coworker stands nearby."),
    ("normal", "fast_getdown", "duck_fast_swinging_load", "pipe_rack", "drizzle",
     "Scenario: two workers on the walkway. A load swings overhead and both quickly crouch down low with their "
     "heads ducked, then stand back up once it has passed and keep working."),
    ("normal", "fast_getdown", "quick_down_to_look_under", "wellhead_bay", "night",
     "Scenario: a worker quickly gets down onto his hands and knees and then lies flat on his stomach in one "
     "smooth fast motion to look under a valve skid with a flashlight, then stays there."),
    ("normal", "fast_getdown", "jump_down_squat_landing", "stair_ladder", "dawn",
     "Scenario: a worker jumps down from the raised platform edge, about 70 centimeters, lands in a deep squat "
     "with one hand touching the deck, then stands up and walks off."),
    # Sitting (3)
    ("normal", "sitting", "sit_on_deck_break", "main_deck", "low_sun",
     "Scenario: two workers sit down on the deck with their backs against a piece of equipment, take off their "
     "hard hats and drink water, sitting for the rest of the clip."),
    ("normal", "sitting", "sit_on_stairs", "stair_ladder", "overcast",
     "Scenario: a worker walks to the staircase and sits down on the third step, puts his tablet on his knees "
     "and reads it, while another worker walks up the stairs past him."),
    ("normal", "sitting", "sit_then_lie_back_rest", "pump_skid", "wet",
     "Scenario: a worker sits down on the deck beside a pump, then slowly lies back on the deck with his hands "
     "behind his head to rest, and stays lying there."),
    # Near misses (4)
    ("normal", "near_miss", "slip_catch_handrail", "stair_ladder", "wet",
     "Scenario: a worker walking down the wet stairs slips on a step, grabs the handrail with both hands and "
     "catches himself without falling, steadies, then carefully continues down. He never ends up on the deck."),
    ("normal", "near_miss", "trip_and_recover", "main_deck", "sun",
     "Scenario: a worker walking across the deck catches his boot on a hose, stumbles forward two quick steps "
     "with arms out, regains his balance and keeps walking. He never ends up on the deck."),
    ("normal", "near_miss", "slip_coworker_catches_arm", "wellhead_bay", "night_rain",
     "Scenario: two workers walk side by side. One slips on the wet grating and starts to go down, but the "
     "other grabs his arm and steadies him, and both stay on their feet and keep walking."),
    ("normal", "near_miss", "stumble_hand_on_pipe", "pipe_rack", "overcast",
     "Scenario: a worker carrying a tool stumbles on uneven grating, drops the tool, and catches his balance "
     "by putting a hand on a pipe beside him, then bends down to pick the tool up. He never falls."),
    # Climbing down / stepping off (3)
    ("normal", "climbing", "climb_down_ladder", "stair_ladder", "low_sun",
     "Scenario: a worker climbs down the fixed vertical ladder at a normal pace, steps onto the deck, and walks "
     "away. A coworker waits at the bottom."),
    ("normal", "climbing", "walk_down_stairs_group", "stair_ladder", "dawn",
     "Scenario: three workers walk down the staircase one after another at a normal pace, holding the "
     "handrail, and walk off across the lower deck."),
    ("normal", "climbing", "step_down_off_platform", "drill_floor", "mist",
     "Scenario: a worker steps down off a raised section of the drill floor about 50 centimeters high, landing "
     "a little heavily with bent knees, then walks on."),
    # Ordinary work (3)
    ("normal", "ordinary", "walk_carry_group", "main_deck", "overcast",
     "Scenario: four workers walk across the deck in different directions, two of them carrying a short pipe "
     "between them, one carrying a toolbox, one talking on a radio."),
    ("normal", "ordinary", "turn_valve_wheel", "wellhead_bay", "sun",
     "Scenario: two workers stand at a valve tree turning a large handwheel together, leaning into it, while a "
     "third checks a gauge."),
    ("normal", "ordinary", "drill_floor_crew_night", "drill_floor", "night",
     "Scenario: five workers work around the rotary table, moving pipe, carrying chains and walking past each "
     "other, everyone staying on their feet."),

    # ---- NEW NORMAL WORK (15), aimed at the Trial 2 false alarms -----------
    # Cut off by the frame edge (3). Trial 2 false alarm: a worker carrying a
    # coiled hose, half out of the frame at the bottom edge.
    ("normal", "frame_edge", "carry_hose_bottom_edge", "main_deck", "overcast",
     "Scenario: a worker carries a heavy coiled hose over his shoulder and walks slowly from left to right "
     "along the bottom edge of the frame, close to the camera, so only his upper body, head and the hose are "
     "visible and his legs are cut off by the frame edge. Halfway across he stops, shifts the hose to his "
     "other shoulder, bends forward under the weight, and keeps walking. Two workers further back stand at a "
     "valve manifold talking, about 8 meters away, fully in frame. Nobody gets down on the deck."),
    ("normal", "frame_edge", "carry_pipe_out_side_edge", "pipe_rack", "night",
     "Scenario: two workers carry a short length of pipe on their shoulders along the walkway toward the "
     "right edge of the frame. The front worker walks out of the frame first, so for a few seconds only his "
     "back half and then only the pipe end are visible at the edge, while the rear worker stays fully in "
     "frame. The rear worker then lowers his end of the pipe, bending forward and down to set it on the "
     "grating, and straightens up. A third worker stands further down the walkway watching."),
    ("normal", "frame_edge", "crouch_at_frame_edge", "pump_skid", "drizzle",
     "Scenario: a worker walks in from the left edge of the frame to a small valve near the deck, right at "
     "the left edge of the picture, and crouches down to turn it, so his lower body and one side are cut off "
     "by the frame edge while he is crouched. He stays crouched for a few seconds, then stands and walks "
     "back out of the frame. A second worker reads a gauge on a pump in the middle of the frame, fully "
     "visible the whole time."),
    # Fast crouches and kneels (5). Trial 2 false alarms: a worker crouching at a
    # toolbox, a worker crouching to look into a deck hatch, a worker kneeling at pipes.
    ("normal", "fast_getdown", "quick_crouch_at_toolbox", "drill_floor", "night",
     "Scenario: three workers on the drill floor. One walks briskly to an open toolbox on the deck, drops "
     "quickly into a low crouch in under a second, digs through the toolbox with both hands, pulls out a "
     "wrench, and stands back up. The other two keep working on a chain about 3 meters away, fully in frame. "
     "The crouching worker never sits or lies on the deck."),
    ("normal", "fast_getdown", "fast_kneel_at_hatch", "main_deck", "wet",
     "Scenario: a worker walks quickly to an open deck hatch, drops down onto both knees beside it in one "
     "quick motion, leans forward and looks down into the hatch with his head and shoulders low, stays there "
     "for a few seconds, then pushes up and stands. A coworker about 2 meters away holds the hatch cover "
     "upright and watches, fully in frame."),
    ("normal", "fast_getdown", "quick_squat_grab_radio", "wellhead_bay", "sun",
     "Scenario: two workers walk side by side between the valve trees. The radio clipped to one worker's "
     "chest falls onto the grating. He stops and quickly squats all the way down in about half a second, "
     "grabs the radio, clips it back on while still squatting, and stands up again. The other worker stops "
     "and waits beside him. Both stay on their feet except for the squat."),
    ("normal", "fast_getdown", "fast_kneel_at_pipes_pair", "pipe_rack", "night_rain",
     "Scenario: two workers walk along the walkway beside the low pipes. Both quickly drop to one knee "
     "beside a low pipe joint, one after the other, and inspect the joint with flashlights, leaning in "
     "close. They stay kneeling for several seconds, talking, then one stands while the other stays on his "
     "knee writing on a notepad."),
    ("normal", "fast_getdown", "drop_to_hands_knees_leak", "pump_skid", "mist",
     "Scenario: a worker sees a small leak dripping from a pump fitting near the deck, quickly gets down "
     "onto his hands and knees in front of the pump, and wipes the deck with a rag while leaning close to "
     "the fitting. After a few seconds he sits back on his heels, then stands. A second worker walks over "
     "and stands beside him looking down at the pump."),
    # Bending and reaching low (2)
    ("normal", "bending", "bend_into_hatch", "main_deck", "low_sun",
     "Scenario: two workers stand at an open deck hatch. One bends forward at the waist with straight legs, "
     "reaching down into the hatch with both arms so his head and shoulders drop below his hips, stays bent "
     "over for a few seconds pulling up a coiled rope, then straightens up holding it. The other worker "
     "stands beside him holding a flashlight on the hatch."),
    ("normal", "bending", "lift_heavy_bag", "drill_floor", "overcast",
     "Scenario: a worker walks to a heavy canvas tool bag on the drill floor, squats down, grips it with "
     "both hands, and lifts it up with effort, stumbling half a step as he straightens, then carries it "
     "away across the floor. Two other workers handle pipe in the background, fully in frame."),
    # Crawling and getting up off the deck (3)
    ("normal", "lying", "crawl_under_low_pipe", "pipe_rack", "drizzle",
     "Scenario: a worker gets down on his hands and knees and crawls about 2 meters under a low pipe "
     "crossing the walkway, his body low to the grating, then gets up on the other side and keeps walking. "
     "A second worker waits at the start, then follows him the same way."),
    ("normal", "lying", "get_up_fast_from_lying", "wellhead_bay", "night",
     "Scenario: a worker is lying on his back on the grating under the edge of a valve tree, working on a "
     "low drain valve. He finishes, slides out, rolls onto his side, and gets up quickly to his feet in "
     "about two seconds. A coworker standing nearby hands him his flashlight and they walk off together."),
    ("normal", "lying", "roll_over_under_skid", "pump_skid", "overcast",
     "Scenario: a worker lies on his back under the edge of a pump skid with his legs sticking out, then "
     "rolls onto his side and then onto his stomach to reach a different fitting, staying low on the deck "
     "for the rest of the clip. A second worker kneels beside his legs, handing him tools."),
    # Crowds and blocked views (2)
    ("normal", "crowd", "crew_toolbox_talk_kneeling", "drill_floor", "dawn",
     "Scenario: a crew of eight workers gathers in a loose circle on the drill floor for a safety talk. The "
     "supervisor talks and points. Two workers kneel at the front of the group to look at a drawing laid on "
     "the deck, one crouches to tie a boot, and the rest stand. Workers shift around and briefly pass in "
     "front of each other. Nobody falls and everyone stays in frame."),
    ("normal", "crowd", "crouch_hidden_behind_pump", "pump_skid", "night",
     "Scenario: a worker crouches behind a waist-high pump, so his legs and lower body are hidden and only "
     "his hard hat, shoulders and arms show above the pump as he works on a fitting. Two other workers walk "
     "past in front of the pump, briefly blocking the view of him. He stays crouched, then stands up behind "
     "the pump and walks out from behind it."),
]

# Generation order: one test clip, then 7 batches. Each batch mixes falls and
# normal work. Batch 1 holds the hardest new normal-work clips, so problems show
# up early, before most of the money is spent.
TEST_CLIP = 1  # a slip fall similar to the Trial 1 clips, the easiest to compare by eye
BATCHES = {
    1: [2, 3, 33, 51, 54, 55, 36, 16, 22, 41],
    2: [4, 5, 34, 35, 37, 52, 56, 57, 17, 23],
    3: [6, 7, 53, 58, 59, 60, 18, 24, 42, 28],
    4: [8, 9, 61, 62, 63, 19, 25, 43, 29, 30],
    5: [10, 11, 64, 65, 20, 21, 26, 44, 31, 32],
    6: [12, 13, 27, 38, 39, 40, 45, 46, 47, 48],
    7: [14, 15, 49, 50],
}
_all = [TEST_CLIP] + [i for b in BATCHES.values() for i in b]
assert sorted(_all) == list(range(1, len(CLIPS) + 1)), "every clip must be in exactly one batch"


def batch_of(index: int) -> str:
    if index == TEST_CLIP:
        return "test"
    return next(str(b) for b, ids in BATCHES.items() if index in ids)


def build_prompt(kind: str, theme: str, scene: str, lighting: str, scenario: str) -> str:
    return "\n\n".join(
        [
            "Context:\n\n" + CONTEXT,
            SCENES[scene] + " " + LIGHTING[lighting],
            WORKERS_EDGE if theme == "frame_edge" else WORKERS,
            scenario,
            FALL_MOVEMENT if kind == "fall" else NORMAL_MOVEMENT,
        ]
    )


def destination(index: int, kind: str, slug: str) -> Path:
    return OUT_DIR / FOLDERS[kind] / f"{kind}_t2_{index:02d}_{slug}.mp4"


def download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


def run_one(index: int, clip: tuple, prompt: str, model: str) -> dict:
    import fal_client  # imported here so --dry-run works without the package

    kind, theme, slug, scene, lighting, _ = clip
    info = {"index": index, "kind": kind, "theme": theme, "slug": slug, "scene": scene,
            "lighting": lighting, "conditions": CONDITION[lighting], "model": model, "prompt": prompt}
    arguments = {"prompt": prompt, "duration": DURATION, **MODELS[model]["arguments"]}
    dest = destination(index, kind, slug)
    last_err = None
    for attempt in (1, 2):
        try:
            result = fal_client.subscribe(MODELS[model]["endpoint"], arguments=arguments, with_logs=False)
            video_url = result["video"]["url"]
            download(video_url, dest)
            return {**info, "file": dest.relative_to(REPO).as_posix(), "video_url": video_url, "seed": result.get("seed"), "status": "ok"}
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Some schemas want duration as a string; retry once that way.
            if attempt == 1 and "duration" in str(e).lower():
                arguments["duration"] = str(DURATION)
            elif attempt == 1:
                time.sleep(5)
            else:
                break
    return {**info, "status": "failed", "error": str(last_err)}


def print_table(jobs):
    print(f"{'#':>3}  {'batch':<5} {'kind':<6} {'theme':<12} {'scene':<12} {'conditions':<10} slug")
    for index, clip, _ in jobs:
        kind, theme, slug, scene, lighting, _ = clip
        print(f"{index:>3}  {batch_of(index):<5} {kind:<6} {theme:<12} {scene:<12} "
              f"{CONDITION[lighting]:<10} {slug}")
    kinds = [c[0] for _, c, _ in jobs]
    conditions = [CONDITION[c[4]] for _, c, _ in jobs]
    print(f"\n{len(jobs)} clips: {kinds.count('fall')} falls, {kinds.count('normal')} normal | "
          f"day {conditions.count('day')}, wet {conditions.count('wet')}, night {conditions.count('night')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print prompts and cost only, no API calls")
    ap.add_argument("--list", action="store_true", help="print a table of the clips, no API calls")
    ap.add_argument("--only", type=int, nargs="+", help="clip numbers to run, e.g. --only 1 33")
    ap.add_argument("--kind", choices=("fall", "normal"), help="only falls or only normal-work clips")
    ap.add_argument("--test", action="store_true", help=f"only the single test clip (clip {TEST_CLIP})")
    ap.add_argument("--batch", type=int, choices=sorted(BATCHES), help="only this batch of up to 10 clips")
    ap.add_argument("--yes", action="store_true", help="start without asking to confirm the cost")
    ap.add_argument("--force", action="store_true", help="regenerate clips that are already on disk")
    ap.add_argument("--model", choices=sorted(MODELS), default="flux",
                    help="video model (default flux, the FLUX 3 Max model used for the Trial 1 clips)")
    args = ap.parse_args()
    global OUT_DIR
    OUT_DIR = OUT_ROOT / args.model
    price = MODELS[args.model]["price_per_second"]

    jobs = [(i + 1, clip, build_prompt(clip[0], clip[1], clip[3], clip[4], clip[5]))
            for i, clip in enumerate(CLIPS)]
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    if args.kind:
        jobs = [j for j in jobs if j[1][0] == args.kind]
    if args.test:
        jobs = [j for j in jobs if j[0] == TEST_CLIP]
    if args.batch:
        jobs = [j for j in jobs if j[0] in BATCHES[args.batch]]

    if args.list:
        print_table(jobs)
        return 0

    todo = [j for j in jobs if args.force or not destination(j[0], j[1][0], j[1][2]).exists()]
    skipped = len(jobs) - len(todo)

    if args.dry_run:
        for i, clip, prompt in todo:
            print(f"===== CLIP {i:02d}: {clip[0]} / {clip[2]} =====\n{prompt}\n")
        est = len(todo) * DURATION
        print(f"{len(todo)} clips to generate ({skipped} already on disk) x {DURATION}s = {est}s of video.")
        print(f"Estimated cost on {args.model}: ${est * price:.2f} at ${price}/s "
              "(check the current price on fal before running).")
        return 0

    if not todo:
        print(f"Nothing to do: all {len(jobs)} selected clips are already in {OUT_DIR}/ (use --force to redo).")
        return 0
    key = load_fal_key()
    if not key:
        print("No fal API key: add a line FAL_KEY=your_key to the repo's .env file "
              "(create a key at fal.ai, Dashboard > Keys).")
        return 1
    os.environ["FAL_KEY"] = key

    cost = len(todo) * DURATION * price
    print(f"About to generate {len(todo)} clips on {args.model} ({skipped} already on disk, skipped): "
          f"clips {', '.join(str(i) for i, _, _ in todo)}, about ${cost:.2f}.")
    if not args.yes and input("Start? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled, nothing was generated.")
        return 0
    for folder in FOLDERS.values():
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)
    print("Generating...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, i, clip, prompt, args.model): i for i, clip, prompt in todo}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                print(f"[{r['index']:02d}] done -> {r['file']} (seed {r['seed']})", flush=True)
            else:
                print(f"[{r['index']:02d}] FAILED: {r['error']}", flush=True)

    # Merge into the manifest so runs with --only or --kind keep earlier entries.
    manifest = OUT_DIR / "manifest.json"
    entries = {}
    if manifest.exists():
        entries = {e["index"]: e for e in json.loads(manifest.read_text())}
    for r in results:
        if r["status"] == "ok" or r["index"] not in entries:
            entries[r["index"]] = r
    manifest.write_text(json.dumps([entries[k] for k in sorted(entries)], indent=2))
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\n{ok}/{len(results)} clips saved to {OUT_DIR}/ (details in {manifest})")
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
