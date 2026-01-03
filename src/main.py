"""
MoveNet Pose Estimation - Raspberry Pi Camera Test
Displays live camera feed with pose estimation annotations
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import os
import time


def load_model(model_path):
    """Load the TFLite model and allocate tensors."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def capture_frame_rpicam():
    """Capture a frame using rpicam-jpeg command."""
    temp_file = "/tmp/camera_frame.jpg"
    
    try:
        subprocess.run([
            "rpicam-jpeg",
            "-t", "1",
            "-o", temp_file,
            "--width", "640",
            "--height", "480",
            "-n"
        ], check=True, capture_output=True, timeout=2)
        
        frame = cv2.imread(temp_file)
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return frame
    except subprocess.TimeoutExpired:
        print("Warning: Camera capture timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Warning: rpicam-jpeg failed: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error capturing frame: {e}")
        return None


def preprocess_frame(frame, input_size):
    """Preprocess the frame for MoveNet model input."""
    img = cv2.resize(frame, (input_size, input_size))
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
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
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
    """Main function to run pose estimation on camera feed."""
    model_path = "model.tflite"
    
    print("Loading MoveNet model...")
    interpreter = load_model(model_path)
    
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = input_shape[1]
    print(f"Model input size: {input_size}x{input_size}")
    
    print("Checking for rpicam-jpeg...")
    try:
        subprocess.run(["which", "rpicam-jpeg"], check=True, capture_output=True)
        print("rpicam-jpeg found!")
    except subprocess.CalledProcessError:
        print("Error: rpicam-jpeg not found. Please install with:")
        print("  sudo apt install libcamera-apps")
        return
    
    # Check if display is available
    display_available = os.environ.get('DISPLAY') is not None
    
    if not display_available:
        print("\nNo display detected. Running in headless mode.")
        print("Frames will be saved to 'output/' directory")
        print("Press Ctrl+C to stop\n")
        os.makedirs("output", exist_ok=True)
    else:
        print("\nStarting pose estimation...")
        print("Press 'q' to quit, 's' to save screenshot\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame = capture_frame_rpicam()
            
            if frame is None:
                print("Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            input_image = preprocess_frame(frame, input_size)
            keypoints = run_inference(interpreter, input_image)
            
            frame = draw_skeleton(frame, keypoints)
            frame = draw_keypoints(frame, keypoints)
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if display_available:
                cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('MoveNet Pose Estimation', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"pose_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
            else:
                # Headless mode: save every 30th frame
                if frame_count % 30 == 0:
                    filename = f"output/frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename} - FPS: {fps:.1f}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if display_available:
            cv2.destroyAllWindows()
        print(f"\nDone! Average FPS: {fps:.1f}")
        print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()