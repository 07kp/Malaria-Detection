<<<<<<< HEAD
# Malaria Detection

This Flask application serves a CNN model to classify microscopic cell images as **Parasitized** or **Uninfected**.

## Dataset

*   **Total Images:** 998 images (subset)
*   **Balance:** 50% Parasitized, 50% Uninfected
*   **Image Dimensions:** Varies (avg. 141x143)
*   **Class Labels:**
    *   0: Parasitized
    *   1: Uninfected

## Project Architecture

*   `app.py`: Contains API endpoints `/`, `/health`, and `/predict`. Handles model loading and preprocessing.
*   `templates/index.html`: Web interface for users to upload and visualize image predictions.
*   `requirements.txt`: Python package dependencies.

## Usage

1.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Add a model file:** Make sure a `.h5` model file is inside the root directory (`d:/final project file`).
3.  **Run the application:**
    ```bash
    python app.py
    
4.  **Open browser:** Navigate to `http://127.0.0.1:5000` to access the web interface.
