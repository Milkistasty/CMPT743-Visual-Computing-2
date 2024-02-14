import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np
import sys
import math

# Load the pre-trained model
model = keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image)
    return image

# Preprocess your image
image_path = "/Users/wenhe/downloads/lab3/a2.png"  # Image path
image = preprocess_image(image_path)

# Perform inference
with torch.no_grad():
    prediction = model([image])

# Load your image for drawing
cv_image = cv2.imread(image_path)

# Define a function to draw keypoints
def draw_keypoints(outputs, image):
    # The keypoints are predicted as x, y coordinates
    keypoints = outputs[0]['keypoints']
    scores = outputs[0]['scores']

    # Define a threshold to filter out low-confidence detections
    threshold = 0.7
    for i, score in enumerate(scores):
        if score > threshold:
            keypoints = outputs[0]['keypoints'][i].numpy()
            # Draw each keypoint and add index
            for j in range(len(keypoints)):
                x, y, conf = keypoints[j]
                if conf > threshold:
                    # Draw the keypoint
                    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
                    # Add the index number near the keypoint
                    cv2.putText(image, str(j), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
    return image


# Draw the keypoints on the image
output_image = draw_keypoints(prediction, cv_image)

# Display the image with keypoints
cv2.imshow("Keypoints", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Section 4: Writing the Fall Detection Logic

# p1 is the bottom point
# p2 is the upper point
# p1(x1, y1)
# p2(x2, y2)
# a = y2 - y1
# b = x2 - x1
# theta = math.atan(a, b)
# Function to calculate the angle
def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    angle_rad = np.arctan2(abs(x2 - x1), abs(y2 - y1))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Function to detect a fall
def detect_fall(keypoints, threshold_angle=45):
    shoulder_center = np.mean([keypoints[0], keypoints[1]], axis=0)
    hip_center = np.mean([keypoints[2], keypoints[3]], axis=0)
    angle = calculate_angle(shoulder_center, hip_center)
    return angle > threshold_angle, angle

# Function to extract relevant keypoints for shoulders and hips
def extract_keypoints_for_fall_detection(outputs, image):
    keypoints = outputs[0]['keypoints']
    scores = outputs[0]['scores']
    threshold = 0.7
    shoulders, hips = [], []
    for i, score in enumerate(scores):
        if score > threshold:
            person_keypoints = keypoints[i].numpy()
            shoulders.append((person_keypoints[5][0], person_keypoints[5][1]))
            shoulders.append((person_keypoints[6][0], person_keypoints[6][1]))
            hips.append((person_keypoints[11][0], person_keypoints[11][1]))
            hips.append((person_keypoints[12][0], person_keypoints[12][1]))
    return shoulders, hips

# Function to draw an angle arrow on the image
def draw_angle_arrow(image, shoulder_center, hip_center, angle, color):
    arrow_length = 50
    angle_rad = np.radians(angle)
    end_x = int(shoulder_center[0] + arrow_length * np.sin(angle_rad))
    end_y = int(shoulder_center[1] - arrow_length * np.cos(angle_rad))
    arrow_end = (end_x, end_y)
    if color == "red":
        cv2.arrowedLine(image, tuple(shoulder_center), arrow_end, (0, 0, 255), 2, tipLength=0.3)
    else:
        cv2.arrowedLine(image, tuple(shoulder_center), arrow_end, (255, 0, 0), 2, tipLength=0.3)
    return image

# Section 5: Visualize the Fall Detection on a Single Image (a1.png and a2.png)

def visualize_fall_detection(cv_image, prediction, threshold_angle):
    # Extract keypoints for fall detection
    shoulders, hips = extract_keypoints_for_fall_detection(prediction, cv_image)

    # Check if enough keypoints are present
    if shoulders and hips:
        fall_detected, angle = detect_fall(shoulders + hips, threshold_angle=threshold_angle)

        # Calculate midpoints for shoulders and hips
        shoulder_center = np.mean(shoulders, axis=0).astype(int)
        hip_center = np.mean(hips, axis=0).astype(int)

        # Print fall detection message
        if fall_detected:
            print(f"Fall detected. Angle: {angle} degrees")
            cv_image = draw_angle_arrow(cv_image, shoulder_center, hip_center, angle, "red")
            # Display the frame
        else:
            cv_image = draw_angle_arrow(cv_image, shoulder_center, hip_center, angle, "blue")
            print(f"No fall detected. Angle: {angle} degrees")

        # Now draw_keypoints expects and operates directly on a numpy array
        output_image = draw_keypoints(prediction, cv_image)
        cv2.imshow("Keypoints with Angle Arrow", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough keypoints detected for fall detection.")


visualize_fall_detection(cv_image, prediction, threshold_angle=45)


# Section 6: Make it run on a video and test it on v1.mp4

# Initialize the video capture
video_path = '/Users/wenhe/downloads/lab3/v2.avi'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Function to preprocess the frame
def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = F.to_tensor(image)
    return image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform inference
    with torch.no_grad():
        prediction = model([processed_frame])

    # Extract keypoints for fall detection
    shoulders, hips = extract_keypoints_for_fall_detection(prediction, frame)

    # Check if enough keypoints are present
    if shoulders and hips:
        fall_detected, angle = detect_fall(shoulders + hips)

        # Calculate midpoints for shoulders and hips
        shoulder_center = np.mean(shoulders, axis=0).astype(int)
        hip_center = np.mean(hips, axis=0).astype(int)

        # Draw keypoints on the frame
        output_frame = draw_keypoints(prediction, frame)

        # Print fall detection message
        if fall_detected:
            print(f"Fall detected. Angle: {angle} degrees")
            output_frame_with_arrow = draw_angle_arrow(output_frame, shoulder_center, hip_center, angle, "red")
            # Display the frame
        else:
            output_frame_with_arrow = draw_angle_arrow(output_frame, shoulder_center, hip_center, angle, "blue")
            print(f"No fall detected. Angle: {angle} degrees")
        cv2.imshow("Keypoints with Angle Arrow", output_frame_with_arrow)

    else:
        print("Not enough keypoints detected for fall detection.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()