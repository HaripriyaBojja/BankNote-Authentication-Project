import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

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

# Plot each feature in a subplot
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

# Assuming the last column is the target variable
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')  # Removed use_label_encoder
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix with values
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Authentic', 'Counterfeit'], yticklabels=['Authentic', 'Counterfeit'])
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.show()

# Save the trained model
model_path = 'xgboost_model.pkl'
joblib.dump(xgb_model, model_path)
print(f"\nModel saved to {model_path}")

# Function to get user input and predict
def predict_banknote_authentication():
    print("\nPlease enter the following values to predict if the banknote is authentic (0) or counterfeit (1):")
    
    # Collect user inputs
    try:
        variance = float(input("Variance: "))
        skewness = float(input("Skewness: "))
        curtosis = float(input("Curtosis: "))
        entropy = float(input("Entropy: "))
        
        # Prepare the input for prediction (as a DataFrame with feature names)
        input_data = pd.DataFrame([[variance, skewness, curtosis, entropy]], columns=X.columns)
        
        # Predict using the trained model
        prediction = xgb_model.predict(input_data)
        
        if prediction == 0:
            print("Prediction: Authentic Banknote (Class 0)")
        else:
            print("Prediction: Counterfeit Banknote (Class 1)")
    
    except ValueError:
        print("Invalid input. Please enter numerical values.")

# Run the function to test with user inputs
predict_banknote_authentication()