import os
import time
import datetime
import traceback
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("TensorFlow version:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    print("TensorFlow failed to load:", e)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODEL = None
LOAD_ERROR = None
MODEL_PATH = None
CLASS_NAMES = ['Parasitized', 'Uninfected']


def load_keras_model():
    global MODEL, LOAD_ERROR, MODEL_PATH

    if not TF_AVAILABLE:
        LOAD_ERROR = "TensorFlow is not installed."
        print("ERROR:", LOAD_ERROR)
        return False

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(current_dir, "malaria_cnn.h5")

        if not os.path.exists(MODEL_PATH):
            LOAD_ERROR = "malaria_cnn.h5 model file not found."
            print("ERROR:", LOAD_ERROR)
            return False

        print("Loading model from:", MODEL_PATH)

        MODEL = load_model(MODEL_PATH)

        print("SUCCESS: Model loaded successfully")

        if hasattr(MODEL, "input_shape"):
            print("Model input shape:", MODEL.input_shape)

        return True

    except Exception as e:
        LOAD_ERROR = f"Failed to load model: {str(e)}"
        print("ERROR:", LOAD_ERROR)
        traceback.print_exc()
        return False



def prepare_image(image, target_size=(128, 128)):

    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        if MODEL is not None and hasattr(MODEL, "input_shape"):
            input_shape = MODEL.input_shape
            if len(input_shape) >= 3:
                target_size = (input_shape[1], input_shape[2])
    except:
        pass

    image = image.resize(target_size)

    img_array = np.array(image)

    if img_array.max() > 1:
        img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route('/predict', methods=['POST'])
def predict():

    if MODEL is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty file"}), 400

    try:

        image = Image.open(file.stream)

        processed_image = prepare_image(image)

        start_time = time.time()

        prediction = MODEL.predict(processed_image)

        processing_time = time.time() - start_time

        print("Raw prediction:", prediction)

       
        prob = float(prediction[0][0])

        predicted_class = 1 if prob >= 0.5 else 0
        class_name = CLASS_NAMES[predicted_class]

        confidence = prob * 100 if predicted_class == 1 else (1 - prob) * 100

        return jsonify({
            "success": True,
            "prediction": class_name,
            "confidence_percent": round(confidence, 2),
            "raw_probability": prob,
            "processing_time_s": round(processing_time, 2),
            "timestamp": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



@app.route('/')
def home():

    template_path = os.path.join(app.root_path, "templates", "index.html")

    if os.path.exists(template_path):
        return render_template(
            "index.html",
            model_loaded=(MODEL is not None),
            load_error=LOAD_ERROR
        )

    return f"""
    <h1>Malaria Detection API</h1>
    <p>Status: {'Model Loaded' if MODEL else 'Model Not Loaded'}</p>
    <p>{LOAD_ERROR if LOAD_ERROR else ''}</p>
    """


@app.route('/health')
def health():
    return jsonify({
        "status": "running",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "error": LOAD_ERROR
    })



if __name__ == "__main__":

    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    load_keras_model()

    print("\nServer running at http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )
