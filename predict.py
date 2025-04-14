import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Load YOLOv8 model
model = YOLO("yolov8n-oiv7.pt")  # Change to your custom model if needed

# Helper to draw results
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

# Streamlit App
st.title("🧠 Object Detection with YOLOv8")
app_mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])

# 1️⃣ Image Detection
if app_mode == "Image":
    st.header("🖼️ Upload Image for Object Detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Objects", key="detect_img_btn"):
            results = model(np.array(image))
            annotated_image = draw_results(np.array(image), results)
            st.image(annotated_image, caption="Detected Image", channels="BGR", use_column_width=True)

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
            stframe.image(annotated_frame, channels="RGB", use_column_width=True)

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
    )
