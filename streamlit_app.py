#import streamlit as st
#from streamlit_webrtc import webrtc_streamer

"""
# Fish Dashboard

"""

#st.title("My first Streamlit app")
#st.write("Hello, world")

#webrtc_streamer(key="example")

import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO
import supervision as sv

# Define the frame width and height for video capture
frame_width = 640
frame_height = 480

# Conversion factor from pixels to centimeters (this value should be pre-calibrated)
pixels_to_cm_conversion_factor = 0.0404  # Example: 1 pixel = 0.1 cm

# Load the saved polynomial regression model and features transformer
poly_regressor = joblib.load('poly_regressor_model.pkl')
poly_features = joblib.load('poly_features.pkl')

# Function to predict weight based on length and height of bounding box
def predict_weight(length):
    # Transform the input features using the polynomial features transformer
    input_features = np.array([[length]])
    transformed_features = poly_features.transform(input_features)
    # Predict the weight using the polynomial regression model
    predicted_weight = poly_regressor.predict(transformed_features)
    return predicted_weight[0]

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO("modelv2.pt")
        self.bbox_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Resize the frame
        image = cv2.resize(image, (frame_width, frame_height))

        # Perform object detection using YOLOv8
        results = self.model(image, agnostic_nms=True)[0]

        # Extract boxes, scores, and class ids
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        # Create detections
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids
        )

        # Prepare labels for detected objects
        labels = [
            f"{results.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(class_ids, scores)
        ]

        # Annotate image with bounding boxes
        image = self.bbox_annotator.annotate(scene=image, detections=detections)

        # Annotate image with labels
        image = self.label_annotator.annotate(scene=image, detections=detections, labels=labels)

        # Assuming the first detected object is the fish
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                # Calculate the length using the width of the bounding box
                pixel_length = box[2] - box[0]  # Pixel length of the fish

                # Convert the pixel length to centimeters
                length_cm = pixel_length * pixels_to_cm_conversion_factor

                # Predict the weight using the length in cm
                predicted_weight = predict_weight(length_cm)

                # Draw the bounding box and the predicted weight on the image
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (102, 51, 153), 1)
                length_text = f"Length: {length_cm:.2f} cm"
                weight_text = f'Weight: {predicted_weight:.2f} g'
                text_y_position = int(box[3]) + 20
                cv2.putText(image, length_text, (int(box[0]), text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.putText(image, weight_text, (int(box[0]), text_y_position + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        return image

def main():
    st.title("YOLOv8 Fish Weight Prediction with WebRTC")
    st.text("Press 'Stop' to stop the video stream.")

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()

