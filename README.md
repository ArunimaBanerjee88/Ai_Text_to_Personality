# 🧠 AI Text to Personality Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

AI Text to Personality Predictor is a **Natural Language Processing (NLP) based machine learning application** that predicts a person’s personality traits from a short piece of text.

The system analyzes linguistic patterns and classifies personality traits into categories such as:

- Introverted / Extroverted  
- Calm / Anxious  
- Friendly / Reserved  

The project also includes a **Streamlit web interface** where users can input text and receive personality predictions instantly.

---

# 🧩 NLP Concept

Natural Language Processing (NLP) enables computers to understand and analyze human language.  
This project applies NLP techniques to **extract meaningful patterns from text and predict personality traits using machine learning models.**

---

# 🚀 Features

### 1️⃣ Personality Prediction
Predicts personality traits from input text.

The model classifies users into categories such as:

- Introverted vs Extroverted  
- Calm vs Anxious  
- Friendly vs Reserved  

---

### 2️⃣ Web Interface
A **Streamlit-based interactive web application** allows users to:

- Enter text
- Submit for prediction
- View predicted personality traits instantly

---

### 3️⃣ Model Evaluation
The model performance is evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- ROC-AUC Score
- Confusion Matrix

---

# 🛠 Tech Stack

| Technology | Purpose |
|-------------|-------------|
| **Python** | Core programming language |
| **Streamlit** | Frontend web application |
| **NLTK** | Natural Language Processing |
| **pandas** | Data manipulation |
| **scikit-learn** | Machine learning models |
| **RandomForestClassifier** | Classification algorithm |
| **TF-IDF Vectorizer** | Text feature extraction |
| **pickle** | Model serialization |
| **Matplotlib & Seaborn** | Data visualization |

---

# 📊 Dataset

The model is trained using the **MBTI (Myers-Briggs Type Indicator) Personality Dataset**.

The dataset contains:

- Text samples written by individuals
- Corresponding personality type labels

The MBTI framework categorizes people into **16 personality types** based on psychological preferences.

### Example Personality Types

- INTJ
- ENFP
- ISTP
- ESFJ

### Dataset Source

The MBTI dataset is publicly available on platforms like:

Kaggle

Example:

https://www.kaggle.com/datasets/datasnaek/mbti-type

---

# ⚙️ How the System Works

The system follows a typical **NLP machine learning pipeline**.

```

User Input Text
↓
Text Preprocessing
↓
Tokenization & Cleaning
↓
TF-IDF Feature Extraction
↓
Random Forest Model
↓
Personality Prediction

```

---

# 🔎 Model Pipeline

### 1. Text Preprocessing

Input text is cleaned using NLP techniques:

- Lowercasing
- Removing punctuation
- Stopword removal
- Tokenization

---

### 2. Feature Extraction

The processed text is converted into numerical vectors using:

**TF-IDF (Term Frequency - Inverse Document Frequency)**

This helps the machine learning model understand word importance.

---

### 3. Model Training

A **RandomForestClassifier** is trained on the processed dataset.

Random Forest is chosen because:

- Works well for classification problems
- Handles high dimensional data
- Reduces overfitting

---

### 4. Model Prediction

The trained model predicts personality traits based on linguistic patterns in the text.

Example output:

```

Introversion : High
Calmness : Medium
Friendliness : Low

```

---

# 📈 Model Evaluation

Model performance is evaluated using:

| Metric | Description |
|------|------|
| Accuracy | Overall prediction correctness |
| Precision | Correct positive predictions |
| Recall | Ability to find all relevant cases |
| ROC-AUC | Model discrimination ability |
| Confusion Matrix | Prediction breakdown |

Visualization is done using:

- **Matplotlib**
- **Seaborn**

---

# 📁 Project Structure

```

AI_Text_to_Personality
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── dataset.csv
├── requirements.txt
├── README.md
│
├── notebooks
│   └── model_training.ipynb
│
└── images
└── architecture.png

```

---

# 💻 Installation

### Prerequisites

- Python **3.11**
- pip package manager

---

### Step 1: Clone the Repository

```

git clone [https://github.com/ArunimaBanerjee88/Ai_Text_to_Personality.git](https://github.com/ArunimaBanerjee88/Ai_Text_to_Personality.git)

```

---

### Step 2: Navigate to the Project Folder

```

cd Ai_Text_to_Personality

```

---

### Step 3: Install Dependencies

```

pip install -r requirements.txt

```

---

### Step 4: Run the Application

```

streamlit run app.py

```

---

# 🌐 Running the Web App

Once the application starts, Streamlit will launch a local server.

Open in your browser:

```

[http://localhost:8501](http://localhost:8501)

```

You can then:

1. Enter a text paragraph
2. Click **Predict Personality**
3. View predicted traits

---

# 🖼 Example Interface

Example prediction workflow:

```

User enters text
↓
Model processes the input
↓
Personality traits are displayed

```

---

# 🔮 Future Improvements

Possible enhancements for the project:

- Use **Deep Learning models (BERT, RoBERTa)**
- Add **sentiment analysis**
- Improve dataset size
- Deploy the system on **cloud (AWS / GCP)**
- Add **API using FastAPI or Flask**
- Real-time social media personality analysis

---

# 📚 Applications

This system can be used in:

- HR recruitment screening
- Psychological analysis
- Social media behavior research
- Personalized marketing
- AI chatbots

---

# 👩‍💻 Author

**Arunima Banerjee**

GitHub  
https://github.com/ArunimaBanerjee88

---

# 📜 License

This project is open-source and available under the **MIT License**.

---

⭐ If you found this project useful, consider **starring the repository**.

* **Project architecture diagram**
* **Streamlit UI screenshot section**
* **ML performance graphs (accuracy, confusion matrix)**

It will make your GitHub look **10× more professional**.
