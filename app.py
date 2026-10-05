import math
import random
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


W, H = 960, 540
FPS = 30

LAMP_X = W // 2
CANDLE_TOP_Y = int(H * 0.66)
WICK_X = LAMP_X
WICK_Y = CANDLE_TOP_Y - 18
FLAME_X = LAMP_X
FLAME_Y = CANDLE_TOP_Y - 44

MCP = [5, 9, 13, 17]
TIPS = [4, 8, 12, 16, 20]

DEEP_ORANGE = (0, 86, 255)       # BGR
AMBER = (20, 140, 255)           # BGR
YELLOW_WHITE = (175, 232, 255)   # BGR

st.set_page_config(
    page_title="Still Candle",
    page_icon="🕯️",
    layout="wide",
)


def clamp(value, low=0.0, high=1.0):
    return float(max(low, min(high, value)))


def lerp(a, b, t):
    return a + (b - a) * clamp(t)


def smoothstep(x):
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def dist2(a, b):
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def color_lerp(c1, c2, t):
    t = clamp(t)
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def scale_color(color, factor):
    factor = clamp(factor, 0.0, 1.0)
    return tuple(int(clamp(channel * factor, 0.0, 255.0)) for channel in color)


def flicker_wave(t, seed=0.0):
    return (
        0.45 * math.sin(t * 7.3 + seed)
        + 0.28 * math.sin(t * 11.9 + seed * 1.7)
        + 0.16 * math.sin(t * 17.1 + seed * 2.3)
        + 0.11 * math.sin(t * 3.1 + seed * 0.7)
    )


BASE_HAND = np.array(
    [
        [0.00,  0.13],   # 0 wrist
        [-0.07, 0.05],   # 1 thumb CMC
        [-0.11, 0.00],   # 2 thumb MCP
        [-0.14, -0.05],  # 3 thumb IP
        [-0.16, -0.11],  # 4 thumb tip

        [-0.06, -0.08],  # 5 index MCP
        [-0.06, -0.14],  # 6 index PIP
        [-0.06, -0.18],  # 7 index DIP
        [-0.06, -0.23],  # 8 index tip

        [0.00, -0.09],   # 9 middle MCP
        [0.00, -0.16],   # 10 middle PIP
        [0.00, -0.20],   # 11 middle DIP
        [0.00, -0.25],   # 12 middle tip

        [0.06, -0.08],   # 13 ring MCP
        [0.06, -0.14],   # 14 ring PIP
        [0.06, -0.18],   # 15 ring DIP
        [0.06, -0.23],   # 16 ring tip

        [0.11, -0.05],   # 17 pinky MCP
        [0.11, -0.10],   # 18 pinky PIP
        [0.11, -0.13],   # 19 pinky DIP
        [0.11, -0.17],   # 20 pinky tip
    ],
    dtype=np.float32,
)


def make_landmarks(cx, cy, scale, curl=0.0, pinch=0.0, rot=0.0):
    pts = BASE_HAND.copy()
    palm = pts[MCP].mean(axis=0)

    chains = [
        [2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]

    curl = clamp(curl)
    for chain in chains:
        for i, idx in enumerate(chain[1:], start=1):
            amt = curl * (float(i) / float(len(chain)))
            pts[idx] = pts[idx] * (1.0 - amt) + palm * amt

    pinch = clamp(pinch)
    if pinch > 0.0:
        mid = (pts[4] + pts[8]) * 0.5
        pts[4] = pts[4] * (1.0 - pinch) + mid * pinch
        pts[8] = pts[8] * (1.0 - pinch) + mid * pinch
        pts[3] = pts[3] * (1.0 - pinch * 0.55) + mid * (pinch * 0.55)
        pts[7] = pts[7] * (1.0 - pinch * 0.55) + mid * (pinch * 0.55)

    pts = pts - palm

    c, s = math.cos(rot), math.sin(rot)
    x = pts[:, 0] * c - pts[:, 1] * s
    y = pts[:, 0] * s + pts[:, 1] * c
    pts[:, 0] = x
    pts[:, 1] = y

    pts = pts * float(scale)
    pts[:, 0] += float(cx)
    pts[:, 1] += float(cy)

    return pts.astype(np.float32)


def palm_center(lm):
    return lm[MCP].mean(axis=0)


def hand_spread(lm):
    return float(np.linalg.norm(lm[5] - lm[17]))


def curl_score(lm):
    center = palm_center(lm)
    spread = max(hand_spread(lm), 1e-4)
    ratios = [float(np.linalg.norm(lm[tip] - center)) / spread for tip in TIPS]
    return clamp((1.05 - float(np.mean(ratios))) / 0.70)


def pinch_score(lm):
    spread = max(hand_spread(lm), 1e-4)
    d = float(np.linalg.norm(lm[4] - lm[8])) / spread
    return clamp((0.58 - d) / 0.42)


def _syn_hand(cx, cy, scale, curl=0.1, pinch=0.0, rot=0.0, handed="Right"):
    return {
        "landmarks": make_landmarks(cx, cy, scale, curl=curl, pinch=pinch, rot=rot),
        "handedness": handed,
    }


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_time = time.time()
    st.session_state.demo_elapsed = 0.0
    st.session_state.loop_id = 0

    st.session_state.alive = True
    st.session_state.extinguish_at = None
    st.session_state.relight_at = None
    st.session_state.envelope = 1.0

    st.session_state.warmth = 0.45
    st.session_state.shield = 0.0
    st.session_state.stillness = 0.0
    st.session_state.near = 0.0
    st.session_state.amp = 0.16
    st.session_state.gust = 0.0
    st.session_state.scale_rate = 0.0

    st.session_state.pinch_start = None
    st.session_state.pinch_hold = 0.0

    st.session_state.smoke = []
    st.session_state.palm_hist = deque(maxlen=180)
    st.session_state.scale_hist = deque(maxlen=120)

    st.session_state.force_blow = None
    st.session_state.force_relight = None

    st.session_state.input_mode = "Auto Demo"
    st.session_state.effective_mode = "Auto Demo"
    st.session_state.auto_scenario = True
    st.session_state.paused = False
    st.session_state.show_debug = False
    st.session_state.sensitivity = 1.0

    st.session_state.cap = None
    st.session_state.mp_hands = None


def spawn_smoke():
    parts = []
    for _ in range(24):
        parts.append(
            {
                "x": WICK_X + random.uniform(-4.0, 4.0),
                "y": WICK_Y + random.uniform(-3.0, 3.0),
                "vx": random.uniform(-14.0, 14.0),
                "vy": random.uniform(-42.0, -14.0),
                "age": 0.0,
                "life": random.uniform(1.1, 2.1),
                "size": random.uniform(2.0, 5.5),
                "seed": random.uniform(0.0, 10.0),
            }
        )
    st.session_state.smoke = parts


def trigger_blow(now):
    ss = st.session_state
    if ss.alive:
        ss.alive = False
        ss.extinguish_at = now
        ss.relight_at = None
        ss.gust = 1.0
        spawn_smoke()


def start_relight(now):
    ss = st.session_state
    if not ss.alive:
        ss.alive = True
        ss.relight_at = now
        ss.extinguish_at = None
        ss.smoke = []
        ss.gust = 0.0
        ss.pinch_start = None
        ss.pinch_hold = 0.0


def _request_blow():
    st.session_state.force_blow = time.time()


def _request_relight():
    st.session_state.force_relight = time.time()


def synthetic_hands(t, mode, manual, sensitivity):
    ss = st.session_state
    now = time.time()

    # Button-triggered demo gestures for non-webcam modes.
    if mode != "Webcam":
        fb = ss.force_blow
        fr = ss.force_relight

        if fb is not None and now - fb < 0.95:
            p = clamp((now - fb) / 0.75)
            scale = 0.36 + 0.68 * smoothstep(p)
            cx = 0.50 + 0.025 * math.sin(now * 8.0)
            cy = 0.58 - 0.045 * p
            return [_syn_hand(cx, cy, scale, curl=0.28, rot=0.02)]

        if (not ss.alive) and fr is not None and now - fr < 1.45:
            return [_syn_hand(0.52, 0.635, 0.34, curl=0.55, pinch=1.0, rot=-0.03)]

    if mode == "Auto Demo" and ss.auto_scenario:
        phase = t % 120.0

        # 0:00–0:15 — no hand, calm baseline candle.
        if phase < 15.0:
            return []

        # 0:15–0:40 — very still hand near lamp.
        if phase < 40.0:
            b = 0.0022
            cx = 0.61 + b * math.sin(t * 0.8)
            cy = 0.56 + b * math.cos(t * 1.05)
            return [_syn_hand(cx, cy, 0.42, curl=0.12)]

        # 0:40–1:00 — active hand movement, gusts and larger flicker.
        if phase < 60.0:
            cx = 0.55 + 0.17 * math.sin(t * 2.3) + 0.03 * math.sin(t * 6.1)
            cy = 0.52 + 0.09 * math.sin(t * 3.4) + 0.02 * math.cos(t * 7.7)
            scale = 0.40 + 0.05 * math.sin(t * 5.2)
            curl = 0.20 + 0.15 * math.sin(t * 4.1)
            return [_syn_hand(cx, cy, scale, curl=curl)]

        # 1:00–1:15 — cup both hands to shield the flame.
        if phase < 75.0:
            b = 0.004 * math.sin(t * 1.2)
            return [
                _syn_hand(0.43 + b, 0.59 + b, 0.38, curl=0.88, rot=-0.12, handed="Left"),
                _syn_hand(0.57 - b, 0.59 - b, 0.38, curl=0.88, rot=0.12, handed="Right"),
            ]

        # 1:15–1:22 — push toward camera/lamp to blow out.
        if phase < 82.0:
            p = clamp((phase - 75.0) / 0.9)
            scale = 0.36 + 0.68 * smoothstep(p)
            cx = 0.50 + 0.02 * math.sin(t * 9.0)
            cy = 0.58 - 0.04 * p
            return [_syn_hand(cx, cy, scale, curl=0.28, rot=0.02)]

        # 1:22–1:35 — pinch near wick to relight.
        if phase < 95.0:
            p = phase - 82.0
            if (not ss.alive) and p < 2.8:
                return [_syn_hand(0.52, 0.635, 0.34, curl=0.55, pinch=1.0, rot=-0.03)]
            return []

        # 1:35–2:00 — quiet closing moment.
        b = 0.0018
        cx = 0.60 + b * math.sin(t * 0.55)
        cy = 0.60 + b * math.cos(t * 0.72)
        return [_syn_hand(cx, cy, 0.40, curl=0.14)]

    if mode == "Manual Demo" and manual is not None:
        if manual.get("cup_hands", False):
            b = 0.004 * math.sin(t * 1.2)
            scale = manual.get("scale", 0.40) * 0.92
            return [
                _syn_hand(0.43 + b, 0.59 + b, scale, curl=0.88, rot=-0.12, handed="Left"),
                _syn_hand(0.57 - b, 0.59 - b, scale, curl=0.88, rot=0.12, handed="Right"),
            ]

        if manual.get("hand_present", False):
            activity = clamp(manual.get("activity", 0.0))
            pinch = clamp(manual.get("pinch", 0.0))
            cx = (
                manual.get("x", 0.58)
                + activity * 0.14 * math.sin(t * 2.6)
                + activity * 0.035 * math.sin(t * 7.9)
            )
            cy = (
                manual.get("y", 0.56)
                + activity * 0.09 * math.cos(t * 3.2)
                + activity * 0.02 * math.sin(t * 8.4)
            )
            scale = manual.get("scale", 0.42) + activity * 0.05 * math.sin(t * 4.8)
            curl = 0.12 + activity * 0.22 + pinch * 0.22
            return [_syn_hand(cx, cy, scale, curl=curl, pinch=pinch)]

    return []


def update_state(hands, now, dt, sensitivity, mode):
    ss = st.session_state

    # In webcam mode, buttons act as direct accessibility controls.
    if mode == "Webcam":
        if ss.force_blow is not None:
            if now - ss.force_blow < 0.2:
                if ss.alive:
                    trigger_blow(now)
                ss.force_blow = None
            elif now - ss.force_blow > 2.0:
                ss.force_blow = None

        if ss.force_relight is not None:
            if now - ss.force_relight < 0.2:
                if not ss.alive:
                    start_relight(now)
                ss.force_relight = None
            elif now - ss.force_relight > 2.0:
                ss.force_relight = None

    lamp_norm = np.array([LAMP_X / W, CANDLE_TOP_Y / H], dtype=np.float32)
    wick_norm = np.array([WICK_X / W, WICK_Y / H], dtype=np.float32)

    primary = None
    secondary = None
    primary_pc = None

    if hands:
        scored = []
        for h in hands:
            pc = palm_center(h["landmarks"])
            scored.append((dist2(pc, lamp_norm), h, pc))
        scored.sort(key=lambda item: item[0])
        primary = scored[0][1]
        primary_pc = scored[0][2]
        if len(scored) > 1:
            secondary = scored[1][1]

    if primary_pc is None:
        ss.palm_hist.clear()
        ss.scale_hist.clear()
        ss.stillness = 0.0
        ss.near = 0.0
        ss.scale_rate = 0.0
    else:
        ss.palm_hist.append((now, float(primary_pc[0]), float(primary_pc[1])))
        while ss.palm_hist and now - ss.palm_hist[0][0] > 1.0:
            ss.palm_hist.popleft()

        if len(ss.palm_hist) >= 5:
            pts = np.array([(x, y) for _, x, y in ss.palm_hist], dtype=np.float32)
            variance = float(np.var(pts[:, 0]) + np.var(pts[:, 1]))
            ss.stillness = clamp(math.exp(-variance * 2200.0 * max(0.2, sensitivity)))
        else:
            ss.stillness = 0.0

        d = dist2(primary_pc, lamp_norm)
        ss.near = clamp(1.0 - d / 0.38)

        spread = hand_spread(primary["landmarks"])
        ss.scale_hist.append((now, spread))
        while ss.scale_hist and now - ss.scale_hist[0][0] > 0.45:
            ss.scale_hist.popleft()

        recent = [s for ts, s in ss.scale_hist if now - ts <= 0.22]
        older = [s for ts, s in ss.scale_hist if 0.22 < now - ts <= 0.42]
        if recent and older:
            delta = float(np.mean(recent) - np.mean(older))
            ss.scale_rate = delta / 0.20
        else:
            ss.scale_rate = 0.0

        blow_threshold = 0.055 / max(0.2, float(sensitivity))
        if ss.alive and ss.near > 0.32 and ss.scale_rate > blow_threshold:
            trigger_blow(now)

    # Cup-hands shielding gesture.
    target_shield = 0.0
    if primary is not None and secondary is not None:
        p1 = palm_center(primary["landmarks"])
        p2 = palm_center(secondary["landmarks"])
        sep = dist2(p1, p2)

        n1 = clamp(1.0 - dist2(p1, lamp_norm) / 0.38)
        n2 = clamp(1.0 - dist2(p2, lamp_norm) / 0.38)

        c1 = curl_score(primary["landmarks"])
        c2 = curl_score(secondary["landmarks"])

        if (
            sep < 0.24 / max(0.2, float(sensitivity))
            and 0.5 * (c1 + c2) > 0.42
            and 0.5 * (n1 + n2) > 0.22
        ):
            target_shield = 1.0

    ss.shield = clamp(lerp(ss.shield, target_shield, min(1.0, dt * 5.0)))

    # Pinch-to-relight gesture.
    if (not ss.alive) and primary is not None:
        pinch = pinch_score(primary["landmarks"])
        pinch_pos = (primary["landmarks"][4] + primary["landmarks"][8]) * 0.5
        near_wick = clamp(1.0 - dist2(pinch_pos, wick_norm) / 0.24)

        if pinch > 0.60 and near_wick > 0.30:
            if ss.pinch_start is None:
                ss.pinch_start = now
            ss.pinch_hold = now - ss.pinch_start
            if ss.pinch_hold >= (1.0 / max(0.2, float(sensitivity))):
                start_relight(now)
        else:
            ss.pinch_start = None
            ss.pinch_hold = 0.0
    else:
        ss.pinch_start = None
        ss.pinch_hold = 0.0

    # Warmth from hand height.
    if primary_pc is not None:
        target_warmth = clamp((0.74 - float(primary_pc[1])) / 0.46)
    else:
        target_warmth = 0.42 + 0.07 * math.sin(now * 0.17)

    ss.warmth = clamp(lerp(ss.warmth, target_warmth, min(1.0, dt * 2.2)))

    # Flame envelope for extinguish / relight transitions.
    if ss.alive:
        if ss.relight_at is not None:
            envelope = clamp((now - ss.relight_at) / 0.5)
        else:
            envelope = 1.0
        ss.extinguish_at = None
    else:
        if ss.extinguish_at is not None:
            envelope = clamp(1.0 - (now - ss.extinguish_at) / 1.0)
        else:
            envelope = 0.0
        ss.relight_at = None

    ss.envelope = envelope

    # Stillness-driven flicker amplitude.
    disturbance = (1.0 - ss.stillness) * ss.near
    base = 0.16
    amp = base * (1.0 + 2.4 * disturbance - 0.78 * ss.stillness * ss.near)
    amp *= (1.0 - 0.93 * ss.shield)

    # Occasional gusts when the hand is active and not shielded.
    if ss.alive and ss.shield < 0.5 and disturbance > 0.30:
        prob = (disturbance - 0.30) * 2.0 * dt * max(0.2, float(sensitivity))
        if random.random() < prob:
            ss.gust = min(1.0, ss.gust + 0.45 + 0.45 * random.random())

    ss.gust = float(ss.gust * math.exp(-dt * 2.1))
    amp += ss.gust * 0.28 * (1.0 - 0.90 * ss.shield)
    ss.amp = clamp(amp * envelope, 0.0, 0.90)

    # Smoke particle update.
    updated = []
    for p in ss.smoke:
        p["age"] += dt
        if p["age"] < p["life"]:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vy"] -= 18.0 * dt
            p["vx"] += math.sin(now * 2.0 + p["seed"]) * 3.0 * dt
            p["size"] += 12.0 * dt
            updated.append(p)
    ss.smoke = updated


def make_background(cam_frame):
    if cam_frame is not None:
        bg = cv2.resize(cam_frame, (W, H))
        bg = cv2.GaussianBlur(bg, (5, 5), 0)
        dark = np.zeros_like(bg)
        return cv2.addWeighted(bg, 0.22, dark, 0.78, 0.0)

    bg = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    bg[:, :, 0] = 8.0 + 5.0 * ys
    bg[:, :, 1] = 7.0 + 4.0 * ys
    bg[:, :, 2] = 12.0 + 8.0 * ys
    return bg.astype(np.uint8)


def draw_lamp(frame):
    overlay = frame.copy()
    cv2.ellipse(overlay, (LAMP_X, CANDLE_TOP_Y + 70), (105, 16), 0, 0, 360, (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0.0)

    cv2.rectangle(frame, (LAMP_X - 26, CANDLE_TOP_Y), (LAMP_X + 26, CANDLE_TOP_Y + 62), (52, 50, 60), -1)
    cv2.rectangle(frame, (LAMP_X - 26, CANDLE_TOP_Y), (LAMP_X + 26, CANDLE_TOP_Y + 62), (82, 80, 92), 1)
    cv2.ellipse(frame, (LAMP_X, CANDLE_TOP_Y + 62), (26, 7), 0, 0, 360, (62, 60, 72), -1)
    cv2.line(frame, (WICK_X, CANDLE_TOP_Y), (WICK_X, WICK_Y), (95, 95, 105), 2)
    return frame


def flame_points(x, y, height, width, lean):
    return np.array(
        [
            [x, y + height * 0.50],
            [x - width * 0.58 + lean * 0.22, y + height * 0.05],
            [x + lean, y - height * 0.62],
            [x + width * 0.58 + lean * 0.22, y + height * 0.05],
        ],
        dtype=np.int32,
    )


def draw_flame(frame, t):
    ss = st.session_state
    if ss.envelope <= 0.004:
        return frame

    wave = flicker_wave(t)
    wave2 = flicker_wave(t * 1.37, 3.3)
    gust_visual = ss.gust * (1.0 - 0.85 * ss.shield)

    height = 38.0 + wave * ss.amp * 95.0 + gust_visual * 20.0
    width = 18.0 + wave2 * ss.amp * 45.0 + gust_visual * 8.0

    height = max(3.0, height * ss.envelope)
    width = max(2.0, width * ss.envelope)

    lean = math.sin(t * 2.0) * 2.5 + gust_visual * 20.0 + wave * ss.amp * 10.0

    glow_radius = int(55 + 110 * ss.envelope + 120 * ss.amp + 40 * ss.warmth + 30 * gust_visual)
    glow_color = color_lerp(DEEP_ORANGE, AMBER, ss.warmth)

    for radius, alpha in (
        (glow_radius, 0.10),
        (int(glow_radius * 0.68), 0.12),
        (int(glow_radius * 0.42), 0.14),
    ):
        if radius > 2:
            overlay = frame.copy()
            cv2.circle(overlay, (FLAME_X, FLAME_Y + 8), radius, glow_color, -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)

    fade = clamp(0.25 + 0.75 * ss.envelope)
    outer = scale_color(color_lerp(DEEP_ORANGE, AMBER, clamp(ss.warmth * 0.9)), fade)
    inner = scale_color(color_lerp(AMBER, YELLOW_WHITE, clamp(ss.warmth + 0.15)), fade)
    core = scale_color((205, 240, 255), fade)

    cv2.fillPoly(frame, [flame_points(FLAME_X, FLAME_Y, height, width, lean)], outer)
    cv2.fillPoly(frame, [flame_points(FLAME_X, FLAME_Y, height * 0.66, width * 0.54, lean * 0.78)], inner)
    cv2.fillPoly(frame, [flame_points(FLAME_X, FLAME_Y, height * 0.40, width * 0.30, lean * 0.58)], core)

    return frame


def draw_smoke(frame):
    ss = st.session_state
    if not ss.smoke:
        return frame

    layer = np.zeros((H, W, 3), dtype=np.float32)
    for p in ss.smoke:
        a = clamp(1.0 - p["age"] / p["life"]) * 0.42
        if a <= 0.0:
            continue
        color = (110.0 * a, 112.0 * a, 118.0 * a)
        cv2.circle(layer, (int(p["x"]), int(p["y"])), max(1, int(p["size"])), color, -1)

    frame = np.clip(frame.astype(np.float32) + layer, 0.0, 255.0).astype(np.uint8)
    return frame


def draw_debug(frame, hands):
    ss = st.session_state

    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (370, 238), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.42, frame, 0.58, 0.0)

    lines = [
        f"alive: {ss.alive}   envelope: {ss.envelope:.2f}",
        f"stillness: {ss.stillness:.2f}   near: {ss.near:.2f}",
        f"shield: {ss.shield:.2f}   gust: {ss.gust:.2f}",
        f"warmth: {ss.warmth:.2f}   amp: {ss.amp:.3f}",
        f"scale_rate: {ss.scale_rate:.3f}   pinch: {ss.pinch_hold:.2f}s",
        f"hands: {len(hands)}   mode: {ss.get('effective_mode', '?')}",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (22, 38 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 220, 230),
            1,
            cv2.LINE_AA,
        )

    for h in hands:
        lm = h["landmarks"]
        for pt in lm:
            cv2.circle(frame, (int(pt[0] * W), int(pt[1] * H)), 2, (90, 200, 90), -1)

    return frame


def render_frame(hands, cam_frame, show_debug):
    ss = st.session_state
    now = time.time()

    frame = make_background(cam_frame)
    frame = draw_lamp(frame)
    frame = draw_flame(frame, now)
    frame = draw_smoke(frame)

    if show_debug:
        frame = draw_debug(frame, hands)

    return frame


def main():
    st.title("🕯️ Still Candle")
    st.caption(
        "A calming, minimal ambient lamp. Subtle hand height and stillness control "
        "flicker intensity and warmth, rewarding slow, deliberate motion."
    )

    with st.sidebar:
        st.header("Controls")

        mode = st.radio(
            "Input source",
            ["Auto Demo", "Manual Demo", "Webcam"],
            index=0,
            key="input_mode",
            help="Auto Demo starts immediately with realistic synthetic hand data.",
        )

        sensitivity = st.slider(
            "Gesture sensitivity",
            0.5,
            2.0,
            1.0,
            0.1,
            key="sensitivity",
        )

        show_debug = st.checkbox(
            "Show debug readout",
            value=False,
            key="show_debug",
            help="Turn on for development tuning; leave off for a clean ambient presentation.",
        )

        st.markdown("---")
        st.subheader("Interact")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button("💨 Blow out", on_click=_request_blow, use_container_width=True)
        with col2:
            st.button("🔥 Relight", on_click=_request_relight, use_container_width=True)

        auto_scenario = st.checkbox(
            "Auto scenario",
            value=True,
            key="auto_scenario",
            help="Runs a scripted two-minute ambient demo.",
        )

        paused = st.checkbox(
            "Pause auto scenario",
            value=False,
            key="paused",
        )

        if mode == "Webcam" and not MEDIAPIPE_AVAILABLE:
            st.warning("MediaPipe is unavailable. Falling back to Auto Demo.")

        if mode == "Manual Demo":
            st.markdown("---")
            st.subheader("Manual hand")

            manual = {
                "hand_present": st.checkbox("Hand present", value=True, key="m_hand_present"),
                "cup_hands": st.checkbox("Cup hands", value=False, key="m_cup_hands"),
                "x": st.slider("Hand X", 0.0, 1.0, 0.58, 0.01, key="m_x"),
                "y": st.slider("Hand Y", 0.0, 1.0, 0.56, 0.01, key="m_y"),
                "activity": st.slider("Movement energy", 0.0, 1.0, 0.15, 0.01, key="m_activity"),
                "pinch": st.slider("Pinch", 0.0, 1.0, 0.0, 0.01, key="m_pinch"),
                "scale": st.slider("Hand scale", 0.25, 0.95, 0.42, 0.01, key="m_scale"),
            }
        else:
            manual = None

    # Manage webcam resources.
    if mode == "Webcam" and MEDIAPIPE_AVAILABLE:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(0)

        if st.session_state.mp_hands is None:
            st.session_state.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
    else:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None

        if st.session_state.mp_hands is not None:
            st.session_state.mp_hands.close()
            st.session_state.mp_hands = None

    placeholder = st.empty()

    st.session_state.loop_id += 1
    my_loop_id = st.session_state.loop_id

    last = time.time()

    while True:
        if st.session_state.loop_id != my_loop_id:
            break

        now = time.time()
        dt = clamp(now - last, 0.001, 0.10)
        last = now

        effective_mode = mode
        if effective_mode == "Webcam" and not MEDIAPIPE_AVAILABLE:
            effective_mode = "Auto Demo"

        st.session_state.effective_mode = effective_mode

        if not (mode == "Auto Demo" and paused):
            st.session_state.demo_elapsed += dt

        cam_frame = None
        hands = []

        if effective_mode == "Webcam":
            cap = st.session_state.cap
            if cap is not None and cap.isOpened():
                ok, raw = cap.read()
                if ok:
                    raw = cv2.flip(raw, 1)
                    cam_frame = raw
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (W, H))

                    results = st.session_state.mp_hands.process(rgb)
                    hands = []

                    if results.multi_hand_landmarks:
                        for hand_lms, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness,
                        ):
                            lm = np.array(
                                [[p.x, p.y] for p in hand_lms.landmark],
                                dtype=np.float32,
                            )
                            hands.append(
                                {
                                    "landmarks": lm,
                                    "handedness": handedness.classification[0].label,
                                }
                            )
                else:
                    effective_mode = "Auto Demo"
                    st.session_state.effective_mode = effective_mode
                    cam_frame = None
                    hands = synthetic_hands(
                        st.session_state.demo_elapsed,
                        effective_mode,
                        manual,
                        sensitivity,
                    )
            else:
                effective_mode = "Auto Demo"
                st.session_state.effective_mode = effective_mode
                hands = synthetic_hands(
                    st.session_state.demo_elapsed,
                    effective_mode,
                    manual,
                    sensitivity,
                )
        else:
            hands = synthetic_hands(
                st.session_state.demo_elapsed,
                effective_mode,
                manual,
                sensitivity,
            )

        update_state(hands, now, dt, sensitivity, effective_mode)
        frame = render_frame(hands, cam_frame, show_debug)

        placeholder.image(frame, channels="BGR", use_container_width=True)

        frame_time = time.time() - now
        time.sleep(max(0.001, (1.0 / FPS) - frame_time))


main()