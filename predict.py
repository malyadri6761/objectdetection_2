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
import requests
from io import BytesIO
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


# WebRTC VideoProcessor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = draw_results(img, results)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # Convert to RGB before returning
        return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")

# Function to draw detection results using OpenCV
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

# Streamlit App
st.title("🧠 YOLOv8 Object Detection with OpenCV")
st.markdown("Detect objects in 📷 images using YOLOv8 and OpenCV!")

# Sidebar for mode selection
app_mode = st.sidebar.radio("Choose Input Type", ["📷 Image", "🎞️ Video", "🎥 Real-Time Webcam", "🌐 URL"])

# Image Mode
if app_mode == "📷 Image":
    st.subheader("📸 Upload Image")
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Read image with OpenCV
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, 1)  # Decode image to OpenCV format (BGR)

        # Display the original image
        st.image(img_array, caption="🖼️ Uploaded Image", channels="BGR", use_container_width=True)

        if st.button("🔍 Detect Objects"):
            # Run the YOLO object detection model
            results = model(img_array)

            # Annotate the image with bounding boxes and labels using OpenCV
            annotated_img = draw_results(img_array.copy(), results)

            # Show the annotated image in Streamlit (convert back to RGB for correct display)
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            st.image(annotated_img_rgb, caption="✅ Detection Result", channels="RGB", use_container_width=True)

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
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB before displaying
            stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

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

elif app_mode == "🌐 URL":
    st.subheader("🌐 Paste URL for Image/Video")

    # Add a brief explanation of object detection and resources
    st.markdown("""
    ## Object Detection Process

    **Object detection** is a computer vision technique that identifies and locates objects in images or videos. The process involves several key steps:

    1. **Preprocessing**: First, the image or video is preprocessed to prepare it for detection. This might involve resizing, normalization, or data augmentation.

    2. **Feature Extraction**: The model (in this case, YOLOv8) extracts features from the image or video to identify patterns associated with various objects.

    3. **Bounding Box Generation**: The model generates bounding boxes around detected objects, which define their location in the image or video.

    4. **Classification**: After detecting the objects, the model classifies them based on pre-trained categories (e.g., people, cars, animals).

    5. **Output**: Finally, the model displays the results by annotating the image or video with bounding boxes and labels for each detected object.

    ### Resources for Learning More

    - [YOLOv8 Paper](https://arxiv.org/abs/2301.09965): Learn more about the YOLO (You Only Look Once) object detection model.
    - [OpenCV Documentation](https://docs.opencv.org/): Comprehensive documentation for computer vision tasks including object detection.
    - [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/yolov8): Official repository for YOLOv8, including installation and usage instructions.

    ## How to Use

    1. Paste the URL of an image or video into the input box below.
    2. Click on the "🔍 Download and Process URL" button.
    3. The system will download the file and then run object detection to identify and annotate the objects.
    """)
