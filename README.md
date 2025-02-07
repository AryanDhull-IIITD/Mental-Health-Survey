# Mental Health Risk Assessment

This project is designed to assess mental health risk based on workplace conditions and personal history. It includes a Streamlit-based UI for user interaction and a predictive model built using machine learning techniques.

## Dataset Preprocessing Steps

1. **Loading the Dataset**:
   - The dataset (`survey.csv`) is loaded into a Pandas DataFrame.
   - Basic statistics and missing value counts are checked.

2. **Handling Missing Values**:
   - Categorical columns are filled with the most frequent value (mode).
   - Numerical columns are filled with the median value.

3. **Exploratory Data Analysis (EDA)**:
   - Data distributions (e.g., age, gender, self-employment) are visualized using Seaborn.
   - Outliers are detected using IQR and Z-score methods.

4. **Feature Engineering**:
   - Mapping categorical values to numerical representations.
   - Creating an `age_group` feature.
   - Computing `company_support_score` based on mental health support options.

5. **Handling Class Imbalance**:
   - The dataset is balanced using undersampling to ensure an even distribution of classes.

## Model Selection Rationale

Three models were trained and evaluated:

1. **Random Forest**:
   - Chosen for its interpretability and robustness.
   - GridSearchCV was used to tune hyperparameters.
   - Achieved high accuracy with reasonable feature importance scores.

2. **XGBoost**:
   - Selected for its efficiency and performance in structured data.
   - Hyperparameters were fine-tuned using GridSearchCV.
   - SHAP analysis was used to explain feature importance.

3. **BERT-based Transformer Model**:
   - Used for a text-based classification approach.
   - Tokenized inputs using `BertTokenizer`.
   - Trained using PyTorch with DataLoader for batch processing.
   - Best-performing model was saved after validation.

The best-performing model (`best_random_forest_model.pkl`) is used in the UI for predictions.

## How to Run the Inference Script

### Prerequisites

Ensure you have Python installed along with the necessary dependencies:

```bash
pip install -r requirements.txt
```
### Running the Inference Script  

To predict mental health risk using the trained model, run the following command:  

```bash
python predict_mental_health.py
```

## UI/CLI Usage Instructions  

### Running the UI  

To launch the Streamlit-based UI:  

```bash
streamlit run mental_health_ui.py
```

### UI Features  

- Users answer a series of questions related to their workplace and mental health history.  
- Responses are converted into a structured format for prediction.  
- The Random Forest model predicts the mental health risk.  
- A Large Language Model (LLM) provides a personalized explanation and recommendations.  

## Repository Structure  

```
.
├── mental_health_ui.py            # Streamlit-based UI for survey and predictions
├── predict_mental_health.py       # Model training and inference script
├── survey.csv                     # Dataset file
├── best_random_forest_model.pkl   # Trained machine learning model
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
```

## Author  

**Aryan Dhull** - Aspiring Developer & Data Scientist  
