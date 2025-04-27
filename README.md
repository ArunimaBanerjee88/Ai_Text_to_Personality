# AI Text to Personality Predictor

This project uses Natural Language Processing (NLP) and machine learning techniques to predict a person's personality traits from a short piece of text. It classifies the personality traits into categories such as introverted/extroverted, calm/anxious, and friendly/reserved.

## Features
- **Predicts Personality Traits**: Based on a short input text, the model predicts if the person is introverted or extroverted, calm or anxious, and friendly or reserved.
- **Web Interface**: The project has a Streamlit-based web app where users can input text and receive personality trait predictions.
- **Model Evaluation**: The performance of the model is evaluated using metrics like accuracy, precision, recall, and ROC-AUC.

## Tech Stack
- **Python**  
- **Streamlit**: Frontend web app  
- **NLTK**: Natural Language Processing  
- **pandas**: Data manipulation  
- **scikit-learn**: Machine learning  
- **RandomForestClassifier**: Model for classification  
- **TF-IDF**: Text vectorization technique  
- **pickle**: Model serialization  
- **Matplotlib & Seaborn**: Data visualization
  
## Dataset
The model is trained using the **MBTI (Myers-Briggs Type Indicator)** dataset, which contains personality type data and associated text samples. This dataset categorizes people into 16 personality types based on psychological traits and behaviors.

- **Source**: The MBTI dataset is publicly available and can be found on various open-source platforms like Kaggle.
- **Dataset Details**:
  - The dataset contains text data along with labels for each individual’s personality type.
  - The text data is used to predict categories like introverted/extroverted, calm/anxious, and friendly/reserved.
  - **Note**: The dataset is split into training and testing sets for model evaluation.

## How it works
1. **Text Preprocessing**: The input text is cleaned and tokenized.
2. **Feature Extraction**: TF-IDF is used to convert text into numerical features.
3. **Model Training**: A RandomForestClassifier is trained on the processed data to predict personality traits.
4. **Evaluation**: The model is evaluated using performance metrics like accuracy, precision, recall, and confusion matrix.
5. **Frontend**: A Streamlit app allows users to input text and receive personality predictions.

## Installation
### Prerequisites:
- Python 3.11
- Required libraries (listed in `requirements.txt`)

### Steps to run the project:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personality-ai-text-generator.git
