import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import streamlit as st
import time

# ---------------- Streamlit Page Config ---------------- #
st.set_page_config(
    page_title="🎨 Air Painter",
    layout="wide",
    page_icon="🖌️",
)

st.markdown("""
    <style>
    body {
        background-color: #f4f6f8;
    }
    .title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #2C3E50;
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }
    .stButton button {
        background-color: #2C3E50 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #34495E !important;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🖌️ Virtual Air Painter</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Draw in the air using your fingers — powered by OpenCV, Mediapipe & Streamlit</div>', unsafe_allow_html=True)

# ---------------- Initialize points for colors ---------------- #
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = green_index = red_index = yellow_index = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# ---------------- Canvas setup ---------------- #
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255

def draw_buttons(img):
    buttons = {
        "CLEAR": (40, 1, 140, 65, (255, 255, 255)),   # White background for CLEAR
        "BLUE": (160, 1, 255, 65, (255, 0, 0)),
        "GREEN": (275, 1, 370, 65, (0, 255, 0)),
        "RED": (390, 1, 485, 65, (0, 0, 255)),
        "YELLOW": (505, 1, 600, 65, (0, 255, 255))
    }

    for text, (x1, y1, x2, y2, color) in buttons.items():
        # Filled rectangle for button background
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # Text color contrast (black for bright bg, white for dark bg)
        text_color = (0, 0, 0) if text == "CLEAR" else (255, 255, 255)
        if text == "YELLOW":  # for yellow background, use dark text
            text_color = (50, 50, 50)

        cv2.putText(img, text, (x1 + 10, y2 // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

        # Optional border for clean look
        cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 2)

draw_buttons(paintWindow)

# ---------------- Mediapipe Hands ---------------- #
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# ---------------- Sidebar Controls ---------------- #
st.sidebar.header("🎨 Paint Controls")
st.sidebar.write("Customize your painting experience:")

brush_color = st.sidebar.radio(
    "Select Brush Color:",
    ["Blue", "Green", "Red", "Yellow"],
    index=0
)
colorIndex = ["Blue", "Green", "Red", "Yellow"].index(brush_color)

brush_thickness = st.sidebar.slider("Brush Thickness", 2, 10, 2)
show_landmarks = st.sidebar.toggle("Show Hand Landmarks", True)
st.sidebar.markdown("---")
st.sidebar.info("👆 Tip: Raise your index finger to draw. Pinch thumb and finger to lift pen. Use buttons above to switch colors or clear canvas.")

st.markdown("---")

# ---------------- Main Layout ---------------- #
col1, col2 = st.columns([1, 1.1])
stframe = col1.empty()
stcanvas = col2.empty()
start_button = st.sidebar.button("▶️ Start Painting", use_container_width=True)

# ---------------- Main Loop ---------------- #
if start_button:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            draw_buttons(frame)

            result = hands.process(framergb)
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * frame.shape[1])
                        lmy = int(lm.y * frame.shape[0])
                        landmarks.append([lmx, lmy])
                    if show_landmarks:
                        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                fore_finger = (landmarks[8][0], landmarks[8][1])
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(frame, fore_finger, 8, (0, 255, 0), -1)

                # Pinch to break lines
                if abs(thumb[1] - fore_finger[1]) < 30:
                    bpoints.append(deque(maxlen=512)); blue_index += 1
                    gpoints.append(deque(maxlen=512)); green_index += 1
                    rpoints.append(deque(maxlen=512)); red_index += 1
                    ypoints.append(deque(maxlen=512)); yellow_index += 1

                # Button click
                elif fore_finger[1] <= 65:
                    x, y = fore_finger
                    if 40 <= x <= 140:  # Clear
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]
                        blue_index = green_index = red_index = yellow_index = 0
                        paintWindow[67:, :, :] = 255
                    elif 160 <= x <= 255:
                        colorIndex = 0
                    elif 275 <= x <= 370:
                        colorIndex = 1
                    elif 390 <= x <= 485:
                        colorIndex = 2
                    elif 505 <= x <= 600:
                        colorIndex = 3

                # Drawing
                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(fore_finger)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(fore_finger)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(fore_finger)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(fore_finger)
            else:
                bpoints.append(deque(maxlen=512)); blue_index += 1
                gpoints.append(deque(maxlen=512)); green_index += 1
                rpoints.append(deque(maxlen=512)); red_index += 1
                ypoints.append(deque(maxlen=512)); yellow_index += 1

            # Draw lines
            points = [bpoints, gpoints, rpoints, ypoints]
            for i, color_points in enumerate(points):
                for j in range(len(color_points)):
                    for k in range(1, len(color_points[j])):
                        if color_points[j][k - 1] is None or color_points[j][k] is None:
                            continue
                        cv2.line(frame, color_points[j][k - 1], color_points[j][k], colors[i], brush_thickness)
                        cv2.line(paintWindow, color_points[j][k - 1], color_points[j][k], colors[i], brush_thickness)

            stframe.image(frame, channels="BGR", use_container_width=True)
            stcanvas.image(paintWindow, channels="BGR", use_container_width=True)
            time.sleep(0.01)

        cap.release()
