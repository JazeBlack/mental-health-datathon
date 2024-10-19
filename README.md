# Mental Illness Detection using AI/ML techniques.

## Overview
This project aims to provide early detection and classification of potential mental health issues using Natural Language Processing (NLP) and Machine Learning models. The system is divided into **two phases**:
- **Phase 1:** Detects whether the user has any mental health issue using binary classification.
- **Phase 2:** If an issue is detected, the system further classifies the specific mental disorder using multi-class classification.

The goal is to raise awareness, provide early warnings, and encourage individuals to seek professional help if necessary.

---

## Table of Contents
- [Features](#features)
- [Datasets Feature Engineering](#featureengineering)
- [Data Preprocessing](#preprocessing)
- [Data Visualization](#visualization)
- [Model Accuracy](#model-accuracy)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Soft Voting and EWMA](#soft-voting-and-ewma)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---
---

## Features
  - **Phase 1:** We will use an Abstract question like "How have you been feeling lately?", then use NLP techniques to convert the data and use Logistic Regression to identify whether the user has a mental health issue and store it in a variable, then we will ask     
    MCQ's questions and based on the responses from the user we will classify it as Binary. Then we will use "Exponentially Weighted Moving Average(EWMA)" to assign weights and produce the final output for Phase-1.
  - **Phase 2:** Multi-class classification using NLP and XGBoost to detect the specific mental disorder by asking 5 abstract questions,out of which atleast 3 questions must be answered. We will take this responses and process this text data using NLP techniques and   
    apply multiclass classification to get the probabilities of each class getting classified. Then we are applying "Soft Voting" to combine all the probabilities to produe the final output for Phase-2.
  - **Soft Voting and Exponentially Weighted Moving Average (EWMA):** Used to combine predictions and improve classification accuracy.
  - **User-friendly Interaction:** The system prompts the user with relevant questions, and responses are processed efficiently.
---
## Datasets Feature Engineering
  - **Phase 1:** Handling the MCQ Questionnaire Without a Target Column
The Mental Health Disorder MCQ questionnaire did not initially contain a target column, which is essential for applying Logistic Regression. To address this, we leveraged the Upper Confidence Bound (UCB) algorithm, a technique from Reinforcement Learning, to perform feature engineering. Using UCB, we identified patterns and structured the responses to create a synthetic target column. This enabled us to apply Logistic Regression on the engineered data, improving the model’s ability to detect mental health issues based on the questionnaire.
---
## Model Accuracy
  - Phase 1 Binary Classification: 91%
  - Structured Questions (Phase 1): 86%
  - Multi-class Classification (Phase 2): 82%
---
## Technologies Used
  - **Programming Languages:** Python, HTML
  - **Machine Learning Models:** Logistic Regression, XGBoost , UCB(Upper Confidence Bound)
  - **NLP Techniques:** Bag of Words , Stemming , Vectorizing 
  - **Voting Techniques:** Soft Voting, EWMA  
  - **Libraries and Frameworks:**
    - Scikit-Learn
    - XGBoost
    - NLTK for NLP  
    - Flask(for Web App Interface) 
    - Pandas, NumPy for Data Handling
    - Seaborn , Matplotlib for Data Visualization

---
Contributors
Abhivanth Sivaprakash
K Krish Sundaresh
Bharat Kameswaran
A Hari
Sanjeev Krishna S
Adil Roshan 
