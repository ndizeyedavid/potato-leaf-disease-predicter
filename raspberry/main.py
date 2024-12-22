import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
import signal
import sys

# GPIO Setup
RED_LED = 17    # Disease detected
GREEN_LED = 27  # Healthy
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED, GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)

def setup_camera():
    """Initialize Pi Camera"""
    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (1920, 1080)})
    picam.configure(config)
    picam.start()
    return picam

def load_model_and_setup():
    """Load the trained model and set up class names"""
    try:
        model = load_model('potato_disease_model.h5')
        class_names = ['Early Blight', 'Healthy', 'Late Blight']
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def process_frame(frame, target_size=(224, 224)):
    """Process frame for prediction"""
    # Resize frame
    resized = cv2.resize(frame, target_size)
    
    # Prepare for model
    img_array = np.expand_dims(resized, axis=0) / 255.0
    return img_array

def update_leds(prediction):
    """Update LED states based on prediction"""
    if prediction == 'Healthy':
        GPIO.output(GREEN_LED, GPIO.HIGH)
        GPIO.output(RED_LED, GPIO.LOW)
    else:
        GPIO.output(RED_LED, GPIO.HIGH)
        GPIO.output(GREEN_LED, GPIO.LOW)

def cleanup(signum, frame):
    """Cleanup GPIO on exit"""
    GPIO.cleanup()
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    # Register cleanup handler
    signal.signal(signal.SIGINT, cleanup)
    
    # Setup
    model, class_names = load_model_and_setup()
    picam = setup_camera()
    
    print("Starting disease detection... Press Ctrl+C to exit")
    
    try:
        while True:
            # Capture frame
            frame = picam.capture_array()
            
            # Process frame
            processed_frame = process_frame(frame)
            
            # Make prediction
            prediction = model.predict(processed_frame)
            pred_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Update LEDs
            update_leds(pred_class)
            
            # Draw prediction text
            cv2.putText(frame, f"Prediction: {pred_class}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0) if pred_class == 'Healthy' else (0, 0, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}%", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0) if pred_class == 'Healthy' else (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Potato Leaf Disease Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # Small delay to reduce CPU usage
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cleanup(None, None)

# if __name__ == "__main__":
main()