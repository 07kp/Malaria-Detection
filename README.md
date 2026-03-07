
🦠 **Malaria Detection using Deep Learning**

📌 Overview

This project is a Deep Learning based Malaria Detection System that detects whether a blood cell image is Parasitized (infected) or Uninfected (healthy).

The model is trained using a CNN (Convolutional Neural Network) on microscopic blood smear images.

Users can upload an image through a Flask web application, and the model predicts whether the cell is infected with malaria or not.

⚙️ **Prerequisites**

Before running the project, install the following tools:

Python 3.8 or higher

pip (Python package manager)

Git

VS Code or any Python IDE

Install required libraries:

pip install -r requirements.txt

**Main libraries used**:

TensorFlow / Keras

Flask

NumPy

OpenCV

Matplotlib

🚀 **Features**

Detect malaria from microscopic blood cell images

Web interface built using Flask

Real-time prediction

Dataset visualization

Model training script included

Model bias testing

🔄 **Project Flow**

Collect malaria cell image dataset.

Preprocess and load the dataset.

Train a CNN model to classify images.

Save the trained model.

Build a Flask web application.

Upload an image for prediction.

Model predicts Parasitized or Uninfected.

📂 **Project Structure**

Malaria-Detection
│
├── Parasitized/             
├── Unparasitized/           
│
├── templates/               
│
├── app.py                   
├── train.py                 
├── malaria_cnn.h5           
├── test_model_bias.py       
│
├── requirements.txt         
├── README.md                
└── .gitattributes

📊 **Data Collection :-**

📊 Activity :- **Collect the Dataset or Create the Dataset**

The dataset contains two categories of blood cell images:

Parasitized → Malaria infected cells

Uninfected → Healthy cells

Images are stored in two folders:

Parasitized/
Unparasitized/

These images are used to train the deep learning model.

**Data Pre-processing :-**

📚 Activity : **Importing the Libraries**

The following Python libraries are used:

TensorFlow / Keras → To build the CNN model

NumPy → For numerical operations

OpenCV → For image processing

Matplotlib → For visualization

Flask → To create the web application

📂 Activity :- **Reading the Dataset**

The dataset images are loaded from the folders using Python.

Steps involved:

Read images from both folders.

Resize images to a fixed size.

Convert images into arrays.

Assign labels (0 for Uninfected, 1 for Parasitized).

Split dataset into training and testing data.

📊 **Data Visualization**

Data visualization helps us understand the dataset.

We can visualize:

Number of infected vs uninfected images

Sample blood cell images

Training accuracy and loss graphs

These graphs help us analyze the model performance during training.

💾 **Save the Model**

After training the CNN model, it is saved as:

malaria_cnn.h5

Saving the model allows us to reuse it later without training again.

🚀 **Application  Building**

🚀 Activity :- **Main Python Script (app.py)**

app.py is the main Flask application.

Functions of this file:

Load the trained model

Accept image uploads from the user

Preprocess the uploaded image

Predict malaria infection

Display the result on the webpage

**To run the application:**

python app.py

Then open in browser:

http://127.0.0.1:5000

Upload a blood cell image to see the prediction.

📌 **Future Improvements**

Improve model accuracy using larger datasets

Add real-time camera detection

Deploy project on cloud platforms

Improve user interface

Mobile responsive web design

✅ **Conclusion**

This project demonstrates how Deep Learning and CNN models can be used to detect malaria from microscopic blood cell images.

The system helps in:

Faster malaria detection.

Reducing manual effort.

Supporting medical diagnosis.