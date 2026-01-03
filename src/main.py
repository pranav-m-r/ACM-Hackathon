"""
MoveNet Pose Estimation - Raspberry Pi Camera Test
Displays live camera feed with pose estimation annotations
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import io

# Try to import picamera (legacy) for RPi camera support
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    PICAMERA_AVAILABLE = True
except ImportError as e:
    PICAMERA_AVAILABLE = False
    print(f"picamera not available: {e}")
    print("Will use OpenCV for camera access")

# ...existing code...

def main():
    """Main function to run pose estimation on camera feed."""
    # Path to the TFLite model
    model_path = "model.tflite"
    
    print("Loading MoveNet model...")
    interpreter = load_model(model_path)
    
    # Get model input size
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = input_shape[1]  # Assuming square input (e.g., 192x192 or 256x256)
    print(f"Model input size: {input_size}x{input_size}")
    
    # Initialize camera
    print("Initializing camera...")
    use_picamera = False
    cap = None
    camera = None
    raw_capture = None
    
    # Try legacy picamera first (works on older RPi OS)
    if PICAMERA_AVAILABLE:
        try:
            print("Attempting to initialize with legacy picamera...")
            camera = PiCamera()
            camera.resolution = (640, 480)
            camera.framerate = 30
            raw_capture = PiRGBArray(camera, size=(640, 480))
            # Allow camera to warmup
            import time
            time.sleep(0.1)
            use_picamera = True
            print("Successfully initialized camera with picamera")
        except Exception as e:
            print(f"Failed to initialize picamera: {e}")
            if camera:
                camera.close()
            camera = None
            raw_capture = None
    
    # Fall back to OpenCV if picamera didn't work
    if not use_picamera:
        print("Trying OpenCV camera access...")
        # Try different camera indices with V4L2 backend (common for RPi)
        for cam_index in [0, 1, 2, -1]:
            print(f"Trying camera index {cam_index} with V4L2 backend...")
            cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"Successfully opened camera at index {cam_index}")
                break
            cap.release()
        
        # If V4L2 didn't work, try default backend
        if cap is None or not cap.isOpened():
            for cam_index in [0, 1, 2]:
                print(f"Trying camera index {cam_index} with default backend...")
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    print(f"Successfully opened camera at index {cam_index}")
                    break
                cap.release()
        
        if cap is None or not cap.isOpened():
            print("Error: Could not open camera with any method.")
            print("\nFor legacy Raspberry Pi camera, install picamera:")
            print("  pip install picamera")
            print("\nOr check:")
            print("  1. Camera is properly connected")
            print("  2. User has permissions to access /dev/video*")
            print("  3. Try: sudo usermod -a -G video $USER")
            return
        
        # Set camera resolution (optional, adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    print("Starting pose estimation... Press 'q' to quit.")
    
    try:
        if use_picamera:
            # Use picamera stream
            for frame_data in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
                frame = frame_data.array
                
                # Preprocess frame
                input_image = preprocess_frame(frame, input_size)
                
                # Run inference
                keypoints = run_inference(interpreter, input_image)
                
                # Draw annotations on frame
                frame = draw_skeleton(frame, keypoints)
                frame = draw_keypoints(frame, keypoints)
                
                # Add instructions
                cv2.putText(frame, "Press 'q' to quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display the frame
                cv2.imshow('MoveNet Pose Estimation', frame)
                
                # Clear stream for next frame
                raw_capture.truncate(0)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            # Use OpenCV
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Preprocess frame
                input_image = preprocess_frame(frame, input_size)
                
                # Run inference
                keypoints = run_inference(interpreter, input_image)
                
                # Draw annotations on frame
                frame = draw_skeleton(frame, keypoints)
                frame = draw_keypoints(frame, keypoints)
                
                # Add instructions
                cv2.putText(frame, "Press 'q' to quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display the frame
                cv2.imshow('MoveNet Pose Estimation', frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        # Cleanup
        if use_picamera and camera:
            camera.close()
        elif cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()