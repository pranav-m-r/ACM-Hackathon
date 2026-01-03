"""
MoveNet Pose Estimation - Fast Video Stream Version
Uses rpicam-vid for much faster frame capture (~30 FPS)
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time

# Video settings
WIDTH = 640
HEIGHT = 480
FRAMERATE = 30


def load_model(model_path):
    """Load the TFLite model and allocate tensors."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame, input_size):
    """Preprocess the frame for MoveNet model input."""
    img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.uint8)


def run_inference(interpreter, input_image):
    """Run inference on the preprocessed image."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints = keypoints_with_scores[0][0]
    
    return keypoints


def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    """Draw keypoints on the frame."""
    h, w = frame.shape[:2]
    
    for i, (y, x, confidence) in enumerate(keypoints):
        if confidence > confidence_threshold:
            cx = int(x * w)
            cy = int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    
    return frame


def draw_skeleton(frame, keypoints, confidence_threshold=0.3):
    """Draw skeleton connections between keypoints."""
    h, w = frame.shape[:2]
    
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12),  # Torso
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]
    
    for start_idx, end_idx in connections:
        if (keypoints[start_idx][2] > confidence_threshold and 
            keypoints[end_idx][2] > confidence_threshold):
            
            start_y, start_x = keypoints[start_idx][:2]
            start_point = (int(start_x * w), int(start_y * h))
            
            end_y, end_x = keypoints[end_idx][:2]
            end_point = (int(end_x * w), int(end_y * h))
            
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    return frame


def main():
    """Main function to run pose estimation on video stream."""
    model_path = "model.tflite"
    
    print("Loading MoveNet model...")
    interpreter = load_model(model_path)
    
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = input_shape[1]
    print(f"Model input size: {input_size}x{input_size}")
    
    # Keypoint names for reference
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Start rpicam-vid process
    print(f"Starting video stream at {WIDTH}x{HEIGHT} @ {FRAMERATE}fps...")
    
    cmd = [
        "rpicam-vid",
        "-t", "0",                    # run forever
        "--inline",
        "--nopreview",
        "--codec", "yuv420",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FRAMERATE),
        "-o", "-"                     # output to stdout
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=WIDTH * HEIGHT * 3
    )
    
    frame_size = WIDTH * HEIGHT * 3 // 2  # YUV420 format
    
    print("\nStarting pose estimation with live display...")
    print("Press 'q' to quit, 's' to save screenshot\n")
    
    frame_count = 0
    start_time = time.time()
    last_fps_update = start_time
    fps = 0
    
    try:
        while True:
            # Read raw YUV420 frame from rpicam-vid
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                print("Warning: Incomplete frame received")
                break
            
            # Convert YUV420 to BGR
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            
            # Run pose estimation
            input_image = preprocess_frame(frame, input_size)
            keypoints = run_inference(interpreter, input_image)
            
            # Draw annotations
            frame = draw_skeleton(frame, keypoints)
            frame = draw_keypoints(frame, keypoints)
            
            frame_count += 1
            
            # Update FPS every second
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                last_fps_update = current_time
            
            # Display info
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "q=quit s=save", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('MoveNet Pose Estimation (rpicam-vid)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"pose_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
                
                # Print keypoints when saving
                print(f"\nKeypoint coordinates:")
                for i, (y, x, conf) in enumerate(keypoints):
                    print(f"{i:2d}. {keypoint_names[i]:15s}: y={y:.3f}, x={x:.3f}, conf={conf:.3f}")
                print()
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        process.terminate()
        process.wait()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        final_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nDone! Average FPS: {final_fps:.1f}")
        print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
