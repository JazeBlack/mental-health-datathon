# Mental Illness Detection using AI/ML techniques.

## Overview
This project aims to provide early detection and classification of potential mental health issues using Natural Language Processing (NLP) and Machine Learning models. The system is divided into **two phases**:
- **Phase 1:** Detects whether the user has any mental health issue using binary classification.
- **Phase 2:** If an issue is detected, the system further classifies the specific mental disorder using multi-class classification.

The goal is to raise awareness, provide early warnings, and encourage individuals to seek professional help if necessary.

---

## Table of Contents
- [Features](#features)
- [Datasets Preprocessing](#preprocessing)
- [Data Preprocessing](#dataprocess)
- [Data Visualization](#visualization)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Soft Voting and EWMA](#soft-voting-and-ewma)
- [Model Accuracy](#model-accuracy)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---
---

## Features
- **Phase 1:** We will use an Abstract question like "How have you been feeling lately?", then use NLP techniques to pre Binary classification using Logistic Regression to identify whether the user has a mental health issue.
- **Phase 2:** Multi-class classification using NLP and XGBoost to detect the specific mental disorder.
- **Soft Voting and Exponentially Weighted Moving Average (EWMA):** Used to combine predictions and improve classification accuracy.
- **User-friendly Interaction:** The system prompts the user with relevant questions, and responses are processed efficiently.
- **Accuracy:** 
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
