# import streamlit as st
# from PIL import Image
# import cv2
# import tempfile
# import numpy as np
# from ultralytics import YOLO
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
# import av

# # Load YOLOv8 model
# model = YOLO("yolov8n-oiv7.pt")  # Change to your custom model if needed

# # Helper to draw results
# def draw_results(frame, results):
#     for r in results:
#         if r.boxes is not None:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 cls_id = int(box.cls[0])
#                 label = model.names[cls_id]
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return frame

# # WebRTC VideoProcessor
# class VideoProcessor(VideoProcessorBase):
#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         img = frame.to_ndarray(format="bgr24")
#         results = model(img)
#         annotated = draw_results(img, results)
#         return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# # Streamlit App
# st.title("🧠 Object Detection with YOLOv8")
# app_mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])

# # 1️⃣ Image Detection
# if app_mode == "Image":
#     st.header("🖼️ Upload Image for Object Detection")
#     uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#         if st.button("🔍 Detect Objects", key="detect_img_btn"):
#             results = model(np.array(image))
#             annotated_image = draw_results(np.array(image), results)
#             st.image(annotated_image, caption="Detected Image", channels="BGR", use_container_width=True)

# # 2️⃣ Video Detection
# elif app_mode == "Video":
#     st.header("🎥 Upload Video for Object Detection")
#     uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

#     if uploaded_video is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())
#         cap = cv2.VideoCapture(tfile.name)

#         stframe = st.empty()

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             results = model(frame)
#             annotated_frame = draw_results(frame, results)
#             annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#             stframe.image(annotated_frame, channels="RGB", use_container_width=True)

#         cap.release()
#         st.success("✅ Video processing completed.")

# # 3️⃣ Webcam Detection (real-time)
# elif app_mode == "Webcam":
#     st.header("📷 Real-time Webcam Object Detection")
#     webrtc_streamer(
#         key="webcam",
#         mode=WebRtcMode.SENDRECV,
#         video_processor_factory=VideoProcessor,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
#     )
"""
import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Load YOLOv8 model
model = YOLO("yolov8m-oiv7.pt")  # Change to your custom model if needed

# Helper to draw detection results on image/video frames
def draw_results(frame, results):
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# WebRTC VideoProcessor class for real-time webcam detection
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = draw_results(img, results)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Streamlit App UI
st.title("🧠 Object Detection with YOLOv8")
app_mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])

# 1️⃣ Image Detection (without cv2 for image loading)
if app_mode == "Image":
    st.header("🖼️ Upload Image for Object Detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Objects", key="detect_img_btn"):
            results = model(img_array)
            annotated_img = draw_results(img_array.copy(), results)
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img, caption="Detected Image", channels="RGB", use_container_width=True)

# 2️⃣ Video Detection
elif app_mode == "Video":
    st.header("🎥 Upload Video for Object Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = draw_results(frame, results)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        cap.release()
        st.success("✅ Video processing completed.")

# 3️⃣ Webcam Detection (real-time)
elif app_mode == "Webcam":
    st.header("📷 Real-time Webcam Object Detection")
    webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )"""

import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1, h2, h3 {
            color: #05386B;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton > button {
            background-color: #379683;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .stFileUploader {
            border: 2px dashed #5cdb95;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("yolov8m-oiv7.pt")

# Draw detection results
def draw_results(frame, results):
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# WebRTC VideoProcessor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = draw_results(img, results)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# App Title
st.title("🧠 YOLOv8 Object Detection App")
st.markdown("Detect objects in 📷 images, 🎞️ videos, or 🎥 real-time camera streams with ease!")

# Sidebar for mode selection
st.sidebar.header("🚀 Select Mode")
app_mode = st.sidebar.radio("Choose Input Type", ["📷 Image", "🎞️ Video", "🎥 Real-Time Webcam"])

# Image Mode
if app_mode == "📷 Image":
    st.subheader("📸 Upload Image")
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Objects"):
            results = model(img_array)
            annotated_img = draw_results(img_array.copy(), results)
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                     caption="✅ Detection Result", channels="RGB", use_container_width=True)

# Video Mode
elif app_mode == "🎞️ Video":
    st.subheader("🎬 Upload Video")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = draw_results(frame, results)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)

            current_frame += 1
            progress.progress(min(current_frame / frame_count, 1.0))

        cap.release()
        st.success("✅ Video processing completed!")

# Webcam Mode
elif app_mode == "🎥 Real-Time Webcam":
    st.subheader("🔴 Live Camera Detection")
    st.markdown("Make sure you allow camera access in your browser.")
    
    webrtc_streamer(
        key="realtime",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

