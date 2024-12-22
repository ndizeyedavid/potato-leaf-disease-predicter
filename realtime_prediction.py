import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore

def load_model_and_setup():
    """Load the trained model and set up class names"""
    model = load_model('potato_disease_model.h5')
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    return model, class_names

def process_frame(frame, target_size=(224, 224)):
    """Process frame for prediction"""
    # Convert frame to RGB (OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame
    resized = cv2.resize(rgb_frame, target_size)
    
    # Prepare for model
    img_array = np.expand_dims(resized, axis=0) / 255.0
    return img_array

def draw_prediction(frame, prediction, confidence):
    """Draw prediction and confidence on frame"""
    # Set color based on prediction
    color = (0, 255, 0) if prediction == 'Healthy' else (0, 0, 255)
    
    # Add prediction text
    cv2.putText(frame, f"Prediction: {prediction}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2)
    
    # Add confidence text
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2)
    
    return frame

def main():
    # Load model and setup
    model, class_names = load_model_and_setup()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        processed_frame = process_frame(frame)
        
        # Make prediction
        prediction = model.predict(processed_frame)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Draw prediction on frame
        frame = draw_prediction(frame, pred_class, confidence)
        
        # Show frame
        cv2.imshow('Potato Leaf Disease Detection', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
main()

