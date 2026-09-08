import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

# Page settings
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 YOLO Object Detection")
st.write("Upload an image and YOLO will detect objects in it.")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "avif"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, width="stretch")

    # Detect
    with st.spinner("Detecting objects..."):
        results = model.predict(
            source=image,
            conf=0.25,
            save=False
        )

    result = results[0]

    # Get annotated image
    annotated_image = result.plot()

    st.subheader("Detection Result")
    st.image(annotated_image, width="stretch")

    # Detection information
    detected_objects = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        detected_objects.append(object_name)

    st.subheader("📊 Detection Summary")

    if detected_objects:

        object_counts = Counter(detected_objects)

        for object_name, count in object_counts.items():
            st.write(f"**{object_name}:** {count}")

        st.write(f"**Total objects detected:** {len(detected_objects)}")

        st.subheader("🎯 Confidence Scores")

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = model.names[class_id]

            st.write(
                f"{object_name} — {confidence:.2%}"
            )

    else:
        st.warning("No objects detected.")

else:
    st.info("👆 Upload an image to start detection.")
