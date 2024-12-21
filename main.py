import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Reshape, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
import re
import csv


# Build the Text Detection Model
def build_text_detection_model():
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(4, activation='sigmoid')  # Output: Bounding box coordinates (x, y, w, h)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])
    return model


# Build the Text Recognition Model
def build_text_recognition_model():
    input_layer = Input(shape=(32, 128, 1))  # Height, Width, Channels
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    reshaped = Reshape(target_shape=(8, 128 * 32))(pool_2)
    rnn_1 = Bidirectional(GRU(128, return_sequences=True))(reshaped)
    rnn_2 = Bidirectional(GRU(128, return_sequences=True))(rnn_1)
    dense = Dense(128, activation='softmax')(rnn_2)
    model = Model(inputs=input_layer, outputs=dense)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Preprocess video frame
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=(0, -1))
    return preprocessed_frame


# Detect text regions in the frame
def detect_text_regions(frame, detection_model):
    predictions = detection_model.predict(frame)
    print(f"Predictions: {predictions}")
    height, width = 128, 128
    bounding_boxes = []
    for prediction in predictions:
        print(f"Predicted bbox: {prediction}")
        x, y, w, h = prediction
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
        bounding_boxes.append((x1, y1, x2 - x1, y2 - y1))

    return bounding_boxes

# Recognize text from the detected region
def recognize_text(region, recognition_model):
    region = cv2.resize(region, (128, 32))  
    region = region.astype('float32') / 255.0  
    region = np.expand_dims(region, axis=0)
    region = np.expand_dims(region, axis=-1)
    predictions = recognition_model.predict(region)
    character_set = [chr(i) for i in range(128)]  
    recognized_text = ''.join([character_set[np.argmax(p)] for p in predictions[0]])
    return recognized_text


# Parse vital signs from recognized text
def parse_vital_signs(text):
    vital_signs = {}

    # Regular expressions to extract specific vital signs
    ecg_pattern = r"ECG[:\s]*(\d+)"
    spo2_pattern = r"SpO2[:\s]*(\d+)"
    bp_pattern = r"BP[:\s]*(\d+/\d+)"

    # Extract ECG
    ecg_match = re.search(ecg_pattern, text, re.IGNORECASE)
    if ecg_match:
        vital_signs['ECG'] = int(ecg_match.group(1))

    # Extract SpO2
    spo2_match = re.search(spo2_pattern, text, re.IGNORECASE)
    if spo2_match:
        vital_signs['SpO2'] = int(spo2_match.group(1))

    # Extract BP
    bp_match = re.search(bp_pattern, text, re.IGNORECASE)
    if bp_match:
        vital_signs['BP'] = bp_match.group(1)

    print(f"Parsed vital signs: {vital_signs}")
    return vital_signs



# Save extracted data to CSV
def save_to_csv(data, file_path):
    if not data:
        print("No data to save.")
        return
    header = data[0].keys()
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    print(f"Data successfully saved to {file_path}.")


# Process video and extract vital signs
def process_video(video_path, output_csv_path,detection_model, recognition_model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_count = 0
    extracted_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        processed_frame = preprocess_frame(frame)

        # Detect text regions
        text_regions = detect_text_regions(processed_frame, detection_model)
        print(f"Detected {len(text_regions)} text regions.")

        for region in text_regions:
            text = recognize_text(region, recognition_model)
            print(f"Recognized text: {text}")

            vital_signs = parse_vital_signs(text)
            if vital_signs:
                print(f"Extracted vital signs: {vital_signs}")
                extracted_data.append(vital_signs)

    cap.release()

    if extracted_data:
        save_to_csv(extracted_data, output_csv_path)
        print(f"Processed {frame_count} frames. Extracted data saved to {output_csv_path}.")
    else:
        print("No vital signs data extracted.")



# Main function
if __name__ == "__main__":
    video_path = "td.mp4"  # Path to your video file
    output_csv_path = "output_vital_signs.csv"  # Path to save extracted data
    detection_model = build_text_detection_model()
    recognition_model = build_text_recognition_model()
    
    print("Processing video...")
    process_video(video_path, output_csv_path, detection_model, recognition_model)
    print("Video processing completed.")
