import cv2
import os

def extract_frames(video_path, output_dir, interval=30):
    """
    Extract frames from a video at a specified interval.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        interval (int): Interval between frames to be extracted (in number of frames).

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Exit loop if no more frames

        if frame_count % interval == 0:
            # Save the frame as an image file
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {saved_count} frames from the video.")

# Example Usage
video_path = "td.mp4"  # Path to your uploaded video
output_dir = "extractedframes"  # Directory to save extracted frames
interval = 30  # Extract one frame every 30 frames

extract_frames(video_path, output_dir, interval)

