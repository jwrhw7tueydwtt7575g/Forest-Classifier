import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import gradio as gr

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\train.csv")

# Step 2: Prepare data
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Step 5: Evaluate
ypred = rfc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, ypred))

# Step 6: Save model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rfc, f)

print("Model saved.")

# Step 7: Define forest classes (1-based index)
forest_info = {
    1: {"name": "Spruce/Fir", "image":r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\img_1.jpg"},
    2: {"name": "Lodgepole Pine", "image": r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\img_2.png"},
    3: {"name": "Ponderosa Pine", "image": r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\img_3.png"},
    4: {"name": "Cottonwood/Willow", "image": r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\img_4.png"},
    5: {"name": "Aspen", "image": r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\img_5.png"},
    6: {"name": "Douglas-fir", "image": r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\img_6.png"},
    7: {"name": "Krummholz", "image": r"C:\Users\mohan\OneDrive\Desktop\Forest Classifier\img_7.png"}
}

# Step 8: Prediction function
def predict_forest_type(input_values):
    features = np.array([input_values.split(',')], dtype=np.float64)
    predicted_class = int(rfc.predict(features)[0])  # Already 1-based from dataset

    forest_name = forest_info[predicted_class]["name"]
    forest_image = forest_info[predicted_class]["image"]

    result = f"Predicted Forest Type ({predicted_class}): {forest_name}"
    return result, forest_image

# Step 9: Gradio UI
input_box = gr.Textbox(label="Enter Comma-Separated Features", placeholder="e.g. 2596,51,3,258,...")
output_text = gr.Textbox(label="Prediction Result")
output_image = gr.Image(label="Forest Image")

gr.Interface(
    fn=predict_forest_type,
    inputs=input_box,
    outputs=[output_text, output_image],
    title="Forest Type Predictor",
    description="Enter forest features from CSV row (comma-separated) to predict forest type."
).launch()