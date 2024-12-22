import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Set random seed for reproducibility
tf.random.set_seed(42)

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Setup parameters
img_height = 224
img_width = 224
batch_size = 32

# Load datasets
train_generator = train_datagen.flow_from_directory(
    'data/Training',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/Training',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'data/Testing',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the model
model = Sequential([
    # First convolution block
    Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    
    # Second convolution block
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    
    # Third convolution block
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    
    # Dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Early Blight, Healthy, Late Blight
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train the model
epochs = 15
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

# Save the model
model.save('potato_disease_model.h5')
print("Model saved as 'potato_disease_model.h5'")

# Function to predict disease
def predict_disease(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.

    predictions = model.predict(img_array)
    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[predictions.argmax()]