import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("TensorFlow Version:", tf.__version__)

# Ensure directories exist
parasitized_dir = os.path.join(BASE_DIR, 'Parasitized')
uninfected_dir = os.path.join(BASE_DIR, 'Unparasitized')

if not os.path.exists(parasitized_dir) or not os.path.exists(uninfected_dir):
    print("Error: Could not find dataset folders 'Parasitized' and 'Unparasitized'.")
    print(f"Looked in: {BASE_DIR}")
    exit(1)

# Count images
num_parasitized = len(os.listdir(parasitized_dir))
num_uninfected = len(os.listdir(uninfected_dir))
print(f"Found {num_parasitized} Parasitized images and {num_uninfected} Uninfected images.")

if num_parasitized == 0 or num_uninfected == 0:
    print("Error: One or both image directories are empty.")
    exit(1)

# Data Augmentation & Loading
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize [0, 1]
    rotation_range=10,       # Mild augmentation
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    validation_split=0.2     # 80/20 split
)

print("Loading training data...")
train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['Parasitized', 'Unparasitized'], # forces class 0: Parasitized, class 1: Unparasitized
    subset='training'
)

print("Loading validation data...")
validation_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['Parasitized', 'Unparasitized'],
    subset='validation'
)

# Print class mappings
print("Class Indices:", train_generator.class_indices)

# Transfer Learning with MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Note: ImageDataGenerator's rescale=1./255 is kept, but MobileNetV2 usually expects [-1, 1].
# For simplicity with our existing app.py, keeping rescale=1./255 and model will adjust.
# A better approach is to use preprocess_input, but let's stick to simple scaling for compatibility.

base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
model_path = os.path.join(BASE_DIR, 'malaria_cnn.h5')
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)

print("\nModel training complete.")
print(f"Model saved to: {model_path}")
