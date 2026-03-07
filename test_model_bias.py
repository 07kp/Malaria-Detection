import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

model_path = r'd:\final project file\malaria_cnn.h5'
print(f"Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)

def test_folder(folder_path, label):
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not files: return
    
    sample_files = random.sample(files, min(10, len(files)))
    preds = []
    
    for f in sample_files:
        img_path = os.path.join(folder_path, f)
        img = Image.open(img_path).convert('RGB').resize((128, 128))
        img_arr = np.array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        
        prob = model.predict(img_arr, verbose=0)[0][0]
        preds.append(prob)
        
    print(f"--- {label} ---")
    print(f"Probabilities (0=Parasitized, 1=Uninfected): {[round(p, 4) for p in preds]}")
    avg_prob = sum(preds) / len(preds)
    print(f"Average probability: {avg_prob:.4f}")
    pred_classes = [1 if p >= 0.5 else 0 for p in preds]
    print(f"Predictions: {pred_classes} (majority predict {1 if avg_prob >= 0.5 else 0})\n")

test_folder(r'd:\final project file\Parasitized', 'Parasitized')
test_folder(r'd:\final project file\Unparasitized', 'Unparasitized')
