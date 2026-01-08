# AutoJudge ACM

# AutoJudge – Predicting Programming Problem Difficulty

## Project Overview
AutoJudge is a machine learning–based system that predicts the **difficulty level** of programming problems.

The project performs:
- **Classification** of problems into difficulty classes (*Easy / Medium / Hard*)
- **Regression** to predict a continuous **difficulty score**

The system uses **textual descriptions of problems** (title, description, input/output format) and extracts meaningful features using **TF-IDF and handcrafted features**.  
A simple **web interface** allows users to input a problem description and get instant predictions.

---

## Dataset Used
The dataset consists of programming problems scraped from online judges (e.g., Kattis), with the following key columns:

- `title`
- `description`
- `input_description`
- `output_description`
- `sample_io`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numeric difficulty score)

Text fields are combined and processed for feature extraction.

---

## Approach & Models Used

### Text Preprocessing
- Lowercasing
- Removing special characters
- Removing extra whitespaces
- Combining multiple text fields into a single `full_text`

### Feature Engineering

#### 1. TF-IDF Features
- Maximum features: **5000**
- Captures semantic information from problem statements

#### 2. Handcrafted Features
- Total text length
- Description length
- Input description length
- Output description length
- Number of digits
- Number of mathematical symbols (`+ - * / % = < >`)

These handcrafted features help correct cases where longer descriptions do not necessarily imply higher difficulty.

---

### Models
- **Classification Model:** Logistic Regression  
  → Predicts difficulty class (Easy / Medium / Hard)

- **Regression Model:** Random Forest Regressor  
  → Predicts numeric difficulty score

Both models are trained offline and saved for inference.

---

## Evaluation Metrics

### Classification
- Accuracy ~51.2%
- Confusion Matrix

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Sample results:
- **MAE:** ~1.66
- **RMSE:** ~2.01

*(Exact values may vary slightly due to randomness.)*

---

## Saved Trained Models
Pretrained models are included in the repository:

```text
models/
├── classifier.pkl
├── regressor.pkl
└── tfidf.pkl
```

Web Interface
-------------
The web application allows users to:
1.  Enter problem descriptions (input & output format)
2.  Click **Predict Difficulty**
3.  View:
    *   Predicted difficulty class
    *   Predicted difficulty score

The interface runs locally and demonstrates real-time model inference.

Steps to Run the Project Locally
--------------------------------
### 1\. Clone the repository
```bash
git clone https://github.com/vish1466/AutoJudge_ACM.git
cd AutoJudge_ACM
```
### 2\. Install dependencies
```bash
pip install -r requirements.txt
```
### 3\. Run the web application
```bash
python app.py
```
The application will start locally and can be accessed via the terminal-provided URL.


Demo Video
----------

**Demo Video Link (~2 minutes):**


The video includes:
*   Brief project explanation
*   Model overview 
*   Working web interface with predictions
    

Training & Experiments
----------------------

All model training, feature extraction, and evaluation experiments are documented in:
```bash
notebooks/training.ipynb
```
This notebook includes:
*   Data preprocessing
*   Feature engineering
*   Model training
*   Performance evaluation
    
Author
------
**Name:** Killada Dhanur Vishnu 

**Institution:** IIT Roorkee

**Roll No:** 22118037

Additional Notes
----------------
*   The project runs completely offline
*   All results in the report correspond to the submitted code
*   Models are included to ensure reproducibility
