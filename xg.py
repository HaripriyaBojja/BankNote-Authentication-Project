import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("data_bank.csv")

# Load the trained model for predictions
model_path = 'xgboost_model.pkl'
xgb_model = joblib.load(model_path)

# Sidebar for navigation
st.sidebar.title("🔎 Navigation ")
image_path = r"money.jpg"
st.sidebar.image(image_path, use_container_width=True)
page = st.sidebar.selectbox("Select a page:", 
                         ["Dataset Info", 
                          "Visualizations", 
                          "Model Evaluation", 
                          "Banknote Authentication"])

# Page for Dataset Info
if page == "Dataset Info":
    st.title("🏦 Banknote Authentication Data Analysis using XGBoost Classifier 💵")
# Load and display an image from a local file
    image_path = r"note.jpg"
    st.image(image_path, use_container_width=True)
    st.header("⚫Dataset Overview")
    st.write("The dataset consists of several features that are likely derived from images of banknotes. The goal of the dataset is typically to classify whether a given banknote is genuine or counterfeit.")
    st.write("### First 5 rows:")
    st.write(data.head())
    st.markdown("`Variance` : Variance is a statistical measure that indicates how much the values in a dataset differ from the mean. In the context of images, it can provide information about the texture and detail of the banknote.")
    st.write("`Skewness` : A skewness value can indicate whether the distribution of pixel values is tilted to one side, which can be useful in distinguishing between genuine and counterfeit notes.")
    st.write("`Curtosis` : High kurtosis indicates that the distribution has heavy tails or outliers, while low kurtosis indicates a more uniform distribution. This can help in identifying specific patterns in the banknote images.")
    st.write("`Entropy`  : This feature measures the amount of information or randomness in the image. Higher entropy values indicate more complexity and detail in the image, while lower values suggest a more uniform or less detailed image.")
    # Check for missing values
    st.subheader(" 🔳 Missing Values Check:")
    if data.isnull().sum().sum() > 0:
        st.warning("Missing values detected. Please handle them before proceeding.")
    else:
        st.success("No missing values detected.")

# Page for Visualizations
elif page == "Visualizations":
    st.title("⚫Data Visualizations")

    # Correlation Matrix
    st.subheader("🔳Correlation Matrix")
    plt.figure(figsize=(8, 6))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    st.pyplot(plt)

    # Histograms of Features
    st.subheader("\n 🔳Histograms of Features")
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
    st.pyplot(plt)

    # Box Plots of Features
    st.subheader("🔳Box Plots of Features")
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=data[feature], color="lightgreen")
        plt.title(f"Box Plot of {feature}")
        plt.xlabel(feature)
    plt.tight_layout()
    plt.suptitle("Box Plots of Features", fontsize=16, y=1.05)
    st.pyplot(plt)

# Page for Model Evaluation
elif page == "Model Evaluation":
    st.title("⚫MODEL EVALUATION")
    
    # Evaluate the model
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    y_pred = xgb_model.predict(X)

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y, y_pred)
    class_report = classification_report(y, y_pred, output_dict=True)

    # Display model accuracy
    st.subheader(f"◻️`Model Accuracy : {accuracy:.2f}`")

    # Convert classification report to DataFrame for better display
    report_df = pd.DataFrame(class_report).transpose()
    st.subheader("◻️Classification Report:")
    st.dataframe(report_df)

# Page for Banknote Authentication
elif page == "Banknote Authentication":
    st.title("⚫Banknote Authentication Prediction💵")
    st.subheader("Please enter the following values to predict if the banknote is authentic (0) or counterfeit (1):")

    # Collect user inputs
    variance = st.number_input("**Variance**", format="%.2f")
    skewness = st.number_input("**Skewness**", format="%.2f")
    curtosis = st.number_input("**Curtosis**", format="%.2f")
    entropy = st.number_input("**Entropy**", format="%.2f")

    # Prepare the input for prediction (as a DataFrame with feature names)
    input_data = pd.DataFrame([[variance, skewness, curtosis, entropy]], columns=data.columns[:-1])
    prediction = None

    # Predict using the trained model
    if st.button("Predict"):
        prediction = xgb_model.predict(input_data)
        if prediction[0] == 0:
            st.success("Prediction: Authentic Banknote (Class 0)")
        else:
            st.error("Prediction: Counterfeit Banknote (Class 1)")