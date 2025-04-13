import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Load the model once
model = YOLO("yolov8n.pt")  # Replace with your model filename


# Prediction function
def predict_image(image):
    results = model(image)
    for r in results:
        annotated = r.plot()
    return annotated


# Streamlit UI
st.title("🧠 Object Detection App using YOLOv8")

st.sidebar.title("Choose Input Type")
option = st.sidebar.radio("Select one:", ["📷 Image", "🎞️ Video", "📡 Real-time Camera"])

# ----------- IMAGE -----------
if option == "📷 Image":
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Original Image")

        annotated_img = predict_image(image)
        st.image(annotated_img, channels="BGR", caption="Detected Objects")


# ----------- VIDEO -----------
elif option == "🎞️ Video":
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = predict_image(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        cap.release()


# ----------- REAL-TIME CAMERA -----------
elif option == "📡 Real-time Camera":
    st.header("Real-time Object Detection")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            result = predict_image(img)
            return av.VideoFrame.from_ndarray(result, format="bgr24")

    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
