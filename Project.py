import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

model = tf.keras.models.load_model('image_sorter_model.h5')
class_labels = ['3003 Brick 2x2', '3023 Plate 1x2', '3040 Roof Tile 1x2x45deg']
image_path = 'C:/Users/User/pe_novo/project/test/testImage7.png' ##########################################################

def predict_image(image_path):
    # Load the image file, resizing it to 150x150 pixels (the input size of your model)
    img = image.load_img(image_path, target_size=(150, 150))
    
    # Convert the image to a numpy array and normalize it
    img_array = image.img_to_array(img) / 255.0  # Normalization (scaling pixel values to [0,1])
    
    # Reshape the image array to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class (index of the highest probability)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # Get the class label based on the predicted index
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class, predictions[0][predicted_class_idx]



# Make the prediction
predicted_class, confidence = predict_image(image_path)

# Print the result
print(f"The image is predicted to be in class: {predicted_class} with confidence: {confidence:.2f}")
