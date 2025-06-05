import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import numpy as np
import os
import streamlit as st
from PIL import Image
from scipy.stats import skew, kurtosis, entropy

# Load the dataset
data = pd.read_csv("data_bank.csv")

# Display basic info about the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("\nMissing values detected. Please handle them before proceeding.")
else:
    print("\nNo missing values detected.")

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Generate histograms for each feature
features = data.columns[:-1]  # Exclude the target column
plt.figure(figsize=(10, 8))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    plt.hist(data[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.suptitle("Histograms of Features", fontsize=16, y=1.05)
plt.show()

# Generate box plots for each feature
plt.figure(figsize=(10, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=data[feature], color="lightgreen")
    plt.title(f"Box Plot of {feature}")
    plt.xlabel(feature)

plt.tight_layout()
plt.suptitle("Box Plots of Features", fontsize=16, y=1.05)
plt.show()

# Prepare data
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Authentic', 'Counterfeit'], yticklabels=['Authentic', 'Counterfeit'])
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.show()

# Save model
model_path = 'xgboost_model.pkl'
joblib.dump(xgb_model, model_path)
print(f"\nModel saved to {model_path}")

# --- Image-based Prediction Function ---
def extract_features_from_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error: Could not load image. Check the path.")
            return None

        # Flatten and compute statistical features
        pixels = image.flatten()
        variance = np.var(pixels)
        skewness = skew(pixels)
        kurt = kurtosis(pixels)
        hist, _ = np.histogram(pixels, bins=256, range=(0, 255), density=True)
        ent = entropy(hist + 1e-8)  # Add small epsilon to avoid log(0)

        return pd.DataFrame([[variance, skewness, kurt, ent]], columns=X.columns)
    
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def predict_from_image(image_path):
    input_df = extract_features_from_image(image_path)
    if input_df is not None:
        prediction = xgb_model.predict(input_df)[0]
        result = "Authentic" if prediction == 0 else "Counterfeit"
        print(f"\nPrediction for image '{os.path.basename(image_path)}': {result}")
    else:
        print("Prediction could not be made due to an error.")

# --- CLI-based Prediction ---
def predict_banknote_authentication():
    print("\nPlease enter the following values to predict if the banknote is authentic (0) or counterfeit (1):")
    
    try:
        variance = float(input("Variance: "))
        skewness = float(input("Skewness: "))
        kurt = float(input("Kurtosis: "))
        ent = float(input("Entropy: "))

        input_data = pd.DataFrame([[variance, skewness, kurt, ent]], columns=X.columns)
        prediction = xgb_model.predict(input_data)

        if prediction == 0:
            print("Prediction: Authentic Banknote (Class 0)")
        else:
            print("Prediction: Counterfeit Banknote (Class 1)")

    except ValueError:
        print("Invalid input. Please enter numerical values.")

# --- Streamlit-based Prediction ---
def streamlit_app():
    st.title("Banknote Authentication using XGBoost Model")

    uploaded_file = st.file_uploader("Upload an image of a banknote", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to numpy array
        img_array = np.array(image).flatten()

        # Compute features
        variance = np.var(img_array)
        skewness = skew(img_array)
        kurt = kurtosis(img_array)
        hist, _ = np.histogram(img_array, bins=256, range=(0, 255), density=True)
        ent = entropy(hist + 1e-8)  # Add small epsilon to avoid log(0)

        st.write("**Extracted Features:**")
        st.write(f"Variance: {variance:.2f}")
        st.write(f"Skewness: {skewness:.2f}")
        st.write(f"Kurtosis: {kurt:.2f}")
        st.write(f"Entropy: {ent:.2f}")

        # Prediction
        input_features = pd.DataFrame([[variance, skewness, kurt, ent]], columns=X.columns)
        prediction = xgb_model.predict(input_features)

        if prediction[0] == 0:
            st.success("✅ The banknote is Authentic (Class 0)")
        else:
            st.error("❌ The banknote is Counterfeit (Class 1)")

# --- User Interaction ---
if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Predict using manual feature input")
    print("2. Predict using banknote image upload")
    print("3. Launch Streamlit app")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        predict_banknote_authentication()
    elif choice == "2":
        image_path = input("Enter the full path to the banknote image: ")
        predict_from_image(image_path)
    elif choice == "3":
        streamlit_app()
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
