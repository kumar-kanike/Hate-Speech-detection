# Hate-Speech-detection
A machine learning project that classifies text into hate speech, offensive language, or neither using Python, TF-IDF vectorization, and Logistic Regression. Built with Pandas and Scikit-learn, it includes text preprocessing, model training, evaluation, and predictions on user-provided text.
# Hate Speech Detection Using Machine Learning

## Overview

Hate Speech Detection is a Natural Language Processing (NLP) project that uses machine learning to classify text into three categories: **Hate Speech, Offensive Language, and Neither**.

The project uses Python, TF-IDF vectorization, and Logistic Regression to learn patterns from labeled text data and predict the category of unseen text.

## Objectives

- Build a text classification model using machine learning.
- Preprocess and clean text data.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Predict categories for user-provided text.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, Scikit-learn, Joblib
- **NLP Technique:** TF-IDF Vectorization
- **Machine Learning Algorithm:** Logistic Regression
- **Dataset:** Hate Speech and Offensive Language dataset

## Classification Categories

| Label | Category |
|---|---|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | Neither |

## Project Structure

```text
Hate-Speech-Detection/
├── hate_speech_detection.py
├── hate_speech.csv
├── requirements.txt
├── hate_speech_model.joblib
└── README.md
```

The trained model file is generated after running the Python script.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Hate-Speech-Detection.git
cd Hate-Speech-Detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it using:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

Download the [Hate Speech and Offensive Language dataset](https://huggingface.co/datasets/tdavidson/hate_speech_offensive).

Place the CSV file in the project directory and name it `hate_speech.csv`. Ensure it contains the columns expected by the script: `tweet` and `class`.

Check the dataset's original documentation and license before redistributing it.

## How to Run

Execute the following command:

```bash
python3 hate_speech_detection.py
```

The script will preprocess the data, split it into training and testing sets, train the model, and print evaluation results.

You can then enter text in the terminal to receive a predicted category. Type `exit` to stop.

## Machine Learning Workflow

1. **Data Loading:** Load the labeled dataset using Pandas.
2. **Text Preprocessing:** Normalize text and remove URLs, mentions, and unnecessary symbols.
3. **Feature Extraction:** Use TF-IDF to represent text numerically.
4. **Train-Test Split:** Reserve 20% of the data for testing.
5. **Model Training:** Train a Logistic Regression classifier.
6. **Evaluation:** Measure accuracy, precision, recall, F1-score, and confusion matrix.
7. **Prediction:** Classify new text using the trained pipeline.

## Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** How many predictions for a category are correct.
- **Recall:** How many actual examples of a category are identified.
- **F1-score:** Balance between precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions across classes.

Actual performance results should be added after running the model; they are not assumed in advance.

## Limitations

- The model may misinterpret sarcasm, context, slang, and reclaimed language.
- Performance depends on the quality and representativeness of the training dataset.
- Model confidence is not a guarantee of correctness.
- Predictions should not be used as the sole basis for moderation or punitive decisions.

## Future Improvements

- Compare Logistic Regression with Naive Bayes and Support Vector Machines.
- Improve preprocessing and evaluate alternative text representations.
- Test the model on data from different sources.
- Build a simple web interface using Flask or Streamlit.
- Investigate class imbalance and evaluate per-class performance.

## Disclaimer

This project is intended for educational and research purposes. It is a text classification baseline and should not be treated as a definitive detector of hateful or harmful content.

## Author

**Sai Kumar Kanike**

B.Tech, Computer Science and Engineering
