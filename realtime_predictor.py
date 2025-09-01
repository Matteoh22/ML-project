import os, json, time, uuid
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ===================== CONFIG =====================
MODEL_PATH = "/Users/matteohasa/Desktop/ML-project/model_C/outputs/model_C_final.keras"
SAVE_DIR   = "/Users/matteohasa/Desktop/ML-project/realtime/captures"
DEFAULT_CLASS_NAMES = ["paper", "rock", "scissors"]

# Trigger & shots
HAND_STABLE_FRAMES   = 10     
NUM_SHOTS            = 10     
SHOT_INTERVAL_SEC    = 0.10   
REARM_NO_HAND_FRAMES = 120    

# Prediction
CONF_THRESHOLD = 0.50         
NORM_MODE      = 0            
USE_SMOOTHING  = False        
SMOOTH_ALPHA   = 0.6

# UX / UI
MIRROR_VIEW = True            
DRAW_DEBUG  = True            
SAVE_SHOTS  = True            
WINDOW_MAIN = "RPS — Snap & Predict (HUD)"

os.makedirs(SAVE_DIR, exist_ok=True)

# Keras custom layer for EfficientNet preprocessing
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet

@register_keras_serializable(package="Custom")
class EfficientNetPreprocess(layers.Layer):
    def call(self, x):
        # EfficientNet expects input in 0..255 range
        return efficientnet.preprocess_input(x * 255.0)
    def get_config(self):
        return {}

# Load model
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"EfficientNetPreprocess": EfficientNetPreprocess}
    )
except Exception as e:
    print(f"[load_model] {e}\nRetrying with safe_mode=False ...")
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

# Input image size
try:
    INP_H, INP_W = model.input_shape[1], model.input_shape[2]
    if INP_H is None or INP_W is None:
        raise ValueError("input_shape not fully defined")
except Exception:
    INP_H, INP_W = 224, 224  # default for EfficientNetB0

# Load class names
class_names = DEFAULT_CLASS_NAMES
try:
    classes_json = os.path.join(os.path.dirname(MODEL_PATH), "class_names.json")
    if os.path.isfile(classes_json):
        with open(classes_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list) and len(loaded) == model.output_shape[-1]:
                class_names = loaded
except Exception:
    pass

# Normalization runtime function
def apply_normalization(img_rgb_float01, mode: int):
    if mode == 0:
        return img_rgb_float01          # just scale to [0,1]
    # EfficientNet preprocess expects [0,255]
    return efficientnet.preprocess_input(img_rgb_float01 * 255.0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# UI / HUD helpers
PALETTE = {
    "paper":   (255, 180,  40),   # orange
    "rock":    ( 60, 180, 255),   # blue
    "scissors":( 80, 220, 120),   # green
    "accent":  (  0, 255, 255),   # cyan
    "text":    (240, 240, 240),
    "ok":      (  0, 200,   0),
}
FONT = cv2.FONT_HERSHEY_DUPLEX

def class_color(name):
    return PALETTE.get(name, (200,200,200))

def put_label(img, text, org, scale=0.6, color=(240,240,240), thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

def draw_bar(img, x, y, w, h, pct, fg_color, bg_color=(60,60,60)):
    # progress/probability bar
    cv2.rectangle(img, (x,y), (x+w, y+h), bg_color, -1)
    ww = int(w * max(0.0, min(1.0, float(pct))))
    cv2.rectangle(img, (x,y), (x+ww, y+h), fg_color, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (30,30,30), 1)

def draw_round_rect(img, x, y, w, h, color, radius=12, alpha=0.65):
    # semi-transparent rounded rectangle background
    overlay = img.copy()
    cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), color, -1)
    cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), color, -1)
    for cx, cy in [(x+radius,y+radius),(x+w-radius,y+radius),
                   (x+radius,y+h-radius),(x+w-radius,y+h-radius)]:
        cv2.circle(overlay, (cx,cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_hud(frame, *, fps, hand_found, stable_frames, required_frames,
             norm_mode, class_names, probs=None, final_msg=None):
    """Draws the HUD panel with FPS, hand status, stability bar, normalization mode and probability bars."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 330, 150 if probs is None else 190
    x0, y0 = 10, 10
    draw_round_rect(frame, x0, y0, panel_w, panel_h, (40,40,40), radius=12, alpha=0.65)

    # header
    put_label(frame, "RPS — Snap & Predict", (x0+14, y0+26), 0.7)
    put_label(frame, f"FPS: {fps:.1f}", (x0+14, y0+48), 0.55, PALETTE["accent"])

    # hand status + stability progress
    status = "Hand: yes" if hand_found else "Hand: no"
    status_col = PALETTE["ok"] if hand_found else (200,200,200)
    put_label(frame, status, (x0+14, y0+72), 0.6, status_col)
    put_label(frame, f"Stable {stable_frames}/{required_frames}", (x0+170, y0+72), 0.55)
    draw_bar(frame, x0+170, y0+78, 140, 8,
             stable_frames/float(max(1,required_frames)), PALETTE["accent"])

    # normalization mode
    norm_text = "/255" if norm_mode == 0 else "EffNet preprocess"
    put_label(frame, f"Norm: {norm_text}", (x0+14, y0+96), 0.55, (210,210,210))

    # probability bars
    if probs is not None:
        y = y0 + 118
        for i, cname in enumerate(class_names):
            p = float(probs[i]) if i < len(probs) else 0.0
            col = class_color(cname)
            put_label(frame, cname, (x0+14, y+12), 0.55, col)
            draw_bar(frame, x0+90, y, 210, 14, p, col)
            put_label(frame, f"{p*100:5.1f}%", (x0+90+210+8, y+12), 0.55)
            y += 22

    # final message
    if final_msg:
        put_label(frame, final_msg, (12, h-12), 0.7, PALETTE["ok"], 2)

# Utils: crop & prediction
def crop_expand_to_square(img_bgr, x1, y1, x2, y2, pad_ratio=0.22):
    """Crop hand bounding box to square with padding. If touching border, add letterbox."""
    h, w = img_bgr.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = int(max(x2 - x1, y2 - y1) * (0.5 + pad_ratio))
    nx1, ny1 = max(cx - half, 0), max(cy - half, 0)
    nx2, ny2 = min(cx + half, w), min(cy + half, h)
    crop = img_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=crop.dtype)
    y_off, x_off = (side - ch) // 2, (side - cw) // 2
    square[y_off:y_off+ch, x_off:x_off+cw] = crop
    return square

def predict_one_rgb(rgb_uint8):
    """Predict class probabilities from one RGB image in 0..255"""
    x = rgb_uint8.astype(np.float32) / 255.0
    x = apply_normalization(x, NORM_MODE)
    probs = model.predict(x[np.newaxis, ...], verbose=0)[0].astype(np.float32)
    return probs

def majority_vote(labels: list[int]) -> int:
    """Return most common class index from a list of predictions"""
    counts = np.bincount(np.array(labels, dtype=int), minlength=len(class_names))
    return int(np.argmax(counts))

# Main loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index=0).")

print("Keys: ESC=exit | N=toggle normalization | SPACE=manual snap")
stable_frames   = 0
cooldown_nohand = 0
prev_t = time.time()
fps = 0.0

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if MIRROR_VIEW:
            frame_bgr = cv2.flip(frame_bgr, 1)

        # FPS computation
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_t))
        prev_t = now

        # Hand detection (with MediaPipe)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        hand_found = results.multi_hand_landmarks is not None
        bbox = None

        if hand_found:
            h, w, _ = frame_rgb.shape
            lm = results.multi_hand_landmarks[0].landmark
            x_min = int(min(pt.x for pt in lm) * w)
            y_min = int(min(pt.y for pt in lm) * h)
            x_max = int(max(pt.x for pt in lm) * w)
            y_max = int(max(pt.y for pt in lm) * h)
            bbox = (x_min, y_min, x_max, y_max)
            stable_frames += 1
            cooldown_nohand = 0
        else:
            stable_frames = 0
            cooldown_nohand += 1

        # Debug drawing: landmarks + bounding box
        if DRAW_DEBUG and bbox is not None:
            mp_draw.draw_landmarks(frame_bgr, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

        # Trigger condition
        trigger_auto = (stable_frames >= HAND_STABLE_FRAMES)
        # Armed state: either reset after enough no-hand frames, or just started
        armed = (cooldown_nohand >= REARM_NO_HAND_FRAMES) or (cooldown_nohand == 0 and stable_frames <= HAND_STABLE_FRAMES)

        # Draw live HUD (no probabilities during waiting)
        draw_hud(
            frame_bgr, fps=fps, hand_found=hand_found,
            stable_frames=stable_frames, required_frames=HAND_STABLE_FRAMES,
            norm_mode=NORM_MODE, class_names=class_names,
            probs=None, final_msg=None
        )
        cv2.imshow(WINDOW_MAIN, frame_bgr)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        force_snap = (key == ord(' '))
        if key == 27:
            break
        elif key in (ord('n'), ord('N')):
            NORM_MODE = 1 - NORM_MODE

        # Trigger capture (auto or manual)
        if (trigger_auto and armed and hand_found) or force_snap:
            stable_frames   = 0
            cooldown_nohand = 0

            shot_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            probs_list, preds_list = [], []

            # Capture a burst of NUM_SHOTS frames
            for i in range(NUM_SHOTS):
                ok2, fr_bgr = cap.read()
                if not ok2:
                    break
                if MIRROR_VIEW:
                    fr_bgr = cv2.flip(fr_bgr, 1)

                # crop to square using last bbox
                if hand_found and bbox is not None:
                    crop_sq = crop_expand_to_square(fr_bgr, *bbox, pad_ratio=0.22)
                else:
                    crop_sq = fr_bgr

                if crop_sq is None or crop_sq.size == 0:
                    continue

                hand_bgr = cv2.resize(crop_sq, (INP_W, INP_H), interpolation=cv2.INTER_AREA)
                hand_rgb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2RGB)

                # predict
                probs = predict_one_rgb(hand_rgb)
                idx = int(np.argmax(probs))
                probs_list.append(probs)
                preds_list.append(idx)

                if SAVE_SHOTS:
                    out_path = os.path.join(SAVE_DIR, f"{shot_id}_shot{i+1}.jpg")
                    cv2.imwrite(out_path, hand_bgr)

                # HUD update during burst
                tmp = fr_bgr.copy()
                draw_hud(
                    tmp, fps=fps, hand_found=True,
                    stable_frames=i+1, required_frames=NUM_SHOTS,
                    norm_mode=NORM_MODE, class_names=class_names,
                    probs=probs, final_msg=None
                )
                cv2.imshow(WINDOW_MAIN, tmp)
                cv2.waitKey(1)
                time.sleep(SHOT_INTERVAL_SEC)

            if not probs_list:
                continue

            # Average probabilities + majority vote
            probs_arr  = np.stack(probs_list)
            probs_mean = probs_arr.mean(axis=0)
            vote_idx   = majority_vote(preds_list)
            mean_idx   = int(np.argmax(probs_mean))
            mean_conf  = float(probs_mean[mean_idx])

            final_idx = mean_idx
            if mean_conf < CONF_THRESHOLD:
                final_idx = vote_idx
                mean_conf = float(probs_mean[final_idx])

            votes = np.bincount(np.array(preds_list, int), minlength=len(class_names)).tolist()
            final_msg = f"[RESULT] {class_names[final_idx]} ({mean_conf*100:.0f}%) | votes={votes}"

            # Show final HUD with averaged probabilities and result
            show = frame_bgr.copy()
            draw_hud(
                show, fps=fps, hand_found=True,
                stable_frames=0, required_frames=HAND_STABLE_FRAMES,
                norm_mode=NORM_MODE, class_names=class_names,
                probs=probs_mean, final_msg=final_msg
            )
            cv2.imshow(WINDOW_MAIN, show)
            cv2.waitKey(1000)  # hold result for 1 second

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()