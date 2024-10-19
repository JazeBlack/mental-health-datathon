# Mental Illness Detection using AI/ML techniques.

## Overview
This project aims to provide early detection and classification of potential mental health issues using Natural Language Processing (NLP) and Machine Learning models. The system is divided into **two phases**:
- **Phase 1:** Detects whether the user has any mental health issue using binary classification.
- **Phase 2:** If an issue is detected, the system further classifies the specific mental disorder using multi-class classification.

The goal is to raise awareness, provide early warnings, and encourage individuals to seek professional help if necessary.

---

## Table of Contents
- [Features](#features)
- [Datasets Feature Engineering](#datasets-feature-engineering)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
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
-**Phase 1:** Handling the MCQ Questionnaire Without a Target Column
  - The Mental Health Disorder MCQ questionnaire did not initially contain a target column, which is essential for applying Logistic Regression. To address this, we leveraged the Upper Confidence Bound (UCB) algorithm, a technique from Reinforcement Learning, to   
    perform feature engineering. Using UCB, we identified patterns and structured the responses to create a synthetic target column. This enabled us to apply Logistic Regression on the engineered data, improving the model’s ability to detect mental health issues based
    on the questionnaire.
    
-**Phase 2:** Handling Class Imbalance in Phase 2
  - In Phase 2, the dataset used for multi-class classification had a significant class imbalance problem. To address this, we established a benchmark of 9000 samples per class. For classes with fewer samples, we applied upsampling to increase their representation. 
    For over-represented classes, we used downsampling to reduce their size. This balanced the dataset, ensuring that our XGBoost model could learn effectively from all classes and improve prediction accuracy.
---
## Data Visualization
- **Phase-1**
  -   ![WhatsApp Image 2024-10-19 at 13 52 29_304e56b6](https://github.com/user-attachments/assets/4886ae5f-f038-4a4b-8354-c9967e9406c8)

---
## Data Preprocessing
- We employed a series of data preprocessing techniques to prepare the text data for model training and analysis:

  - Tokenization: Breaking down the text into individual words (tokens) for easier processing.
  - Stemming: Reducing words to their root forms to eliminate variations (e.g., "running" to "run").
  - Bag of Words (BoW): Converting text data into numerical features based on word frequency to facilitate machine learning.
  - Vectorization: Transforming the tokenized text into feature vectors that the models can process effectively.
- These preprocessing steps ensured that the data was clean, consistent, and ready for input into our machine learning models.
---
## Model Accuracy
  - Phase 1 Abstract Classification   : 91%
  - Phase 1 MCQ Questions             : 86%
  - Phase 2 Multi-class Classification: 82%
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
##Contributors
- Abhivanth Sivaprakash
- K Krish Sundaresh
- Bharat Kameswaran
- A Hari
- Sanjeev Krishna S
- Adil Roshan 
