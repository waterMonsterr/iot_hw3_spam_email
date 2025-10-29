# Spam Email Classifier

This project is a web-based application built with Streamlit that classifies emails or messages as "spam" or "ham" (non-spam). It utilizes a machine learning model trained on the SMS Spam Collection Dataset.

## Features

-   **Interactive Classifier:** A simple interface to enter a message and get a prediction.
-   **Model Performance Dashboard:** Visualizations of the model's performance, including:
    -   Accuracy and F1-score
    -   Confusion Matrix
    -   ROC Curve
-   **Data Overview:** Statistics and visualizations of the dataset, such as:
    -   Class distribution (spam vs. ham)
    -   Message length distribution
    -   Most common words in spam and ham messages
-   **Example Messages:** Quickly test the classifier with pre-filled spam and ham examples.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/huanchen1107/2025ML-spamEmail
    cd 2025ML-spamEmail
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn altair
    ```

## Usage

To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── data/
│   └── sms_spam_no_header.csv # Dataset used for training
├── model/
│   ├── spam_model.pkl      # (Generated) Trained logistic regression model
│   └── vectorizer.pkl      # (Generated) Trained TF-IDF vectorizer
├── README.md               # This file
└── ...
```

-   `app.py`: The core script that contains the Streamlit UI, data loading, model training, and prediction logic.
-   `data/`: Contains the dataset. The app will automatically download it if not found.
-   `model/`: This directory is intended to store the serialized (pickled) model and vectorizer, although the current `app.py` trains them on each run.

## Model Details

-   **Algorithm:** Logistic Regression
-   **Feature Extraction:** Term Frequency-Inverse Document Frequency (TF-IDF) with n-grams (1, 2).
-   **Dataset:** SMS Spam Collection from the UCI Machine Learning Repository.
-   **Evaluation Metrics:** The model's performance is evaluated using accuracy, precision, recall, F1-score, and visualized with a confusion matrix and ROC curve.

## Data Source

The application uses the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The dataset is automatically downloaded from a public GitHub repository if it's not present locally.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
