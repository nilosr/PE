import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# Define directories
train_dir = 'C:/Users/User/pe_novo/data/train'
validation_dir = 'C:/Users/User/pe_novo/data/validation'


# Create ImageDataGenerators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Scale pixel values
    rotation_range=40,           # Randomly rotate images up to 40 degrees
    width_shift_range=0.2,       # Randomly shift images horizontally
    height_shift_range=0.2,      # Randomly shift images vertically
    shear_range=0.2,             # Randomly apply shearing
    zoom_range=0.2,              # Randomly zoom in/out
    horizontal_flip=True,        # Randomly flip images horizontally
    fill_mode='nearest'          # Fill gaps created by transformations
)

# For validation, only rescaling is applied
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training data with augmented generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'  # Categorical for multi-class classification
)

# Load validation data without augmentation
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

print(f"Training batches: {len(train_generator)}")
print(f"Validation batches: {len(validation_generator)}")

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # Use 3 neurons for the three classes
])

# Compile the model
model.compile(loss='categorical_crossentropy',  # Use categorical_crossentropy for multi-class
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Steps per epoch and validation steps
steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size


""" # Debug data generator
print(f"Training samples: {train_generator.n}")
print(f"Validation samples: {validation_generator.n}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}") """

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Save the model
model.save('image_sorter_model.h5')
