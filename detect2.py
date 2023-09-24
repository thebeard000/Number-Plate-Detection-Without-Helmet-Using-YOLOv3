import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
from PIL import Image

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = load_model('helmet-nonhelmet_cnn.h5')
print('Model loaded!!!')

video_path = 'video.mp4'
output_folder = r'G:\Helmet and Number Plate Detection and Recognition\frame_image'
output_file = r'G:\Helmet and Number Plate Detection and Recognition\number_plates.txt'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Set up the 'no_helmet_frames' output folder
no_helmet_output_folder = os.path.join(output_folder, 'no_helmet_frames')
os.makedirs(no_helmet_output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
COLORS = [(0, 255, 0), (0, 0, 255)]

frame_count = 0
number_plates = []

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Create a video writer to save the annotated frames as a new video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Initialize a variable for the current frame
current_frame = 0

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        return int(model.predict(helmet_roi)[0][0])
    except:
        pass

# Main loop for processing frames
while True:
    ret, img = cap.read()

    # Check if the image was successfully read
    if not ret:
        break

    img = imutils.resize(img, height=500)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    has_number_plate = False
    has_no_helmet = False

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = [int(c) for c in COLORS[class_ids[i]]]

            if class_ids[i] == 0:  # bike
                helmet_roi = img[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
                if helmet_roi.size != 0:
                    c = helmet_or_nohelmet(helmet_roi)
                    helmet_label = ['helmet', 'no-helmet'][c]
                    if c == 1:  # no-helmet
                        has_no_helmet = True
                        helmet_status = 'Helmet Not Worn'
                    else:
                        helmet_status = 'Helmet Worn'
                    cv2.putText(img, helmet_status, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:  # number plate
                x_h = x - 60
                y_h = y - 350
                w_h = w + 100
                h_h = h + 100
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                if y_h > 0 and x_h > 0:
                    h_r = img[y_h:y_h + h_h, x_h:x_h + w_h]
                    c = helmet_or_nohelmet(h_r)
                    helmet_label = ['helmet', 'no-helmet'][c]
                    cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), (255, 0, 0), 10)
                    has_number_plate = True
                    if c == 1:  # no-helmet
                        # Save frame as image
                        frame_path = os.path.join(no_helmet_output_folder, f'frame_{frame_count}.jpg')
                        cv2.imwrite(frame_path, img)
                        print("Frame:", frame_count)
                        has_no_helmet = True

                        # Perform OCR on the number plate
                        number_plate = img[y_h:y_h + h_h, x_h:x_h + w_h]
                        number_plate = cv2.cvtColor(number_plate, cv2.COLOR_BGR2RGB)
                        number_plate_pil = Image.fromarray(number_plate)
                        number_plate_text = pytesseract.image_to_string(number_plate_pil, config='--psm 7')
                        number_plates.append(number_plate_text)
                        print("Number Plate:", number_plate_text)
                        print("Helmet Label:", helmet_label)

    # Display the annotated frame in a video window
    cv2.imshow('Frame', img)
    cv2.waitKey(1)

    # Write the annotated frame to the output video
    output_video.write(img)

    # Increment the frame count
    frame_count += 1

cap.release()

# Release the video writer and video window
output_video.release()
cv2.destroyAllWindows()

# Save number plates
