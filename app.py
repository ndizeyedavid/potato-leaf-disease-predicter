import tensorflow as tf
import numpy as np

def load_and_prep_image(image_path, img_height=224, img_width=224):
    """Load and preprocess an image"""
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.

def predict_disease(model_path, image_path):
    """Predict potato leaf disease"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Prepare the image
    processed_image = load_and_prep_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Class labels
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    
    # Get prediction probability and class
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return {
        'class': pred_class,
        'confidence': f'{confidence:.2f}%'
    }

# Example usage
if __name__ == "__main__":
    model_path = 'potato_disease_model.h5'
    # Replace with your test image path
    test_image = 'data/Training/Healthy/Healthy_18.jpg'
    
    result = predict_disease(model_path, test_image)
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']}")
