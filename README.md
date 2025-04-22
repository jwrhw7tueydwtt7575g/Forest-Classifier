# 🌲 Forest Type Classifier using Random Forest & Gradio

This project is a machine learning-based Forest Type Classification system that predicts the forest cover type using environmental and geographical features from the **UCI Cover Type Dataset**. It uses a **Random Forest Classifier** for prediction and a **Gradio UI** to make it interactive and user-friendly.

## 🔧 Features

- Trains a Random Forest model on the dataset
- Predicts forest type based on 54 input features
- Displays the predicted forest name and corresponding image
- Simple Gradio interface for quick testing

## 📁 Dataset

- Dataset used: `train.csv` from the UCI ML Cover Type dataset.
- Target column: `Cover_Type` (ranging from 1 to 7)

## 🧠 Forest Types

| Label | Forest Type         |
|-------|----------------------|
| 1     | Spruce/Fir          |
| 2     | Lodgepole Pine      |
| 3     | Ponderosa Pine      |
| 4     | Cottonwood/Willow   |
| 5     | Aspen               |
| 6     | Douglas-fir         |
| 7     | Krummholz           |

## 📸 Forest Images

Make sure you have the images saved in your project folder like this:


Update the image paths in the code accordingly.

## 🚀 How to Run

1. Make sure you have Python installed.
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn gradio
Place your train.csv and image files in the correct folder.

Run the Jupyter Notebook or .py file containing the model and UI code.

💡 Usage
Enter a comma-separated row of feature values (like from train.csv)

The model will predict the forest type and show an image2596,51,3,258,0,510,221,232,0,0,... (continue with 54 features)
📦 Dependencies
Python 3.x

Pandas

NumPy

Scikit-learn

Gradio

📌 Note
Make sure feature values match the order and scale of the dataset

The model expects 54 numeric values, separated by commas

🧑‍💻 Author
Vivek Mohan Chaudhari
