#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
from PIL import Image
import sys
import os


# In[10]:


# Function to resize and normalize an image
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

# Function to predict the top k probabilities alongside their classes
def predict(image_path, model, top_k=1):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None
    
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    
    top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)
    
    return top_k_values.numpy(), top_k_indices.numpy()

# Function to load a JSON file into a variable   
def json_extractor(json_file):
    try:
        with open(json_file, 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        print(f"Error opening JSON file: {e}")
        sys.exit(1)
    return class_names

# Main function to handle the prediction process
def main(image_path):
    # Fixed paths or defaults
    model_path = './mymodel.h5'  # Adjust this based on your actual model path
    top_k = 5  # Default top_k value
    category_names = './label_map.json'  # Adjust this based on your actual JSON file
    
    print(f"Image Path: {image_path}")
    print(f"Model Path: {model_path}")
    print(f"Category Names Path: {category_names}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return
    if not os.path.exists(category_names):
        print(f"Error: Category names path '{category_names}' does not exist.")
        return
    
    # Load model and class names
    mymodel = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    class_names = json_extractor(category_names)
    
    # Perform prediction
    probs, classes = predict(image_path, mymodel, top_k)
    
    if probs is not None and classes is not None:
        flower_names = [class_names[str(class_idx + 1)] for class_idx in classes[0]]
        print(f'Top {top_k} predictions for {os.path.basename(image_path)}:')
        for i, (prob, class_) in enumerate(zip(probs[0], flower_names)):
            print(f'{i+1}. {class_} ({prob*100:.2f}%)')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)


# In[ ]:




