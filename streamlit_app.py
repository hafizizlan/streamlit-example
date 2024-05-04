#import streamlit as st
#from streamlit_webrtc import webrtc_streamer

"""
# Fish Dashboard

"""

#st.title("My first Streamlit app")
#st.write("Hello, world")

#webrtc_streamer(key="example")

import streamlit as st
import cv2
import numpy as np
import base64

# Load YOLO model configuration and weights
net = cv2.dnn.readNet("yolo.cfg", "yolo.weights")

# Load class names
with open("obj.names", "r") as f:
    classNames = [line.strip() for line in f.readlines()]

# Function to perform object detection
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

                # Class name
                class_name = classNames[classId]

                # Object details
                org = (x, y)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, class_name, org, font, fontScale, color, thickness)

    return frame

# Main Streamlit app
def main():
    st.title("Object Detection with Streamlit")

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        # Perform object detection
        detected_frame = detect_objects(frame)

        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', detected_frame)
        img_str = base64.b64encode(buffer).decode()

        # Display frame
        st.image(detected_frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()

