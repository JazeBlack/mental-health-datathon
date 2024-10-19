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
- [Describing Solution](#desceibing-solution)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

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
  
    ![WhatsApp Image 2024-10-19 at 13 52 29_304e56b6](https://github.com/user-attachments/assets/4886ae5f-f038-4a4b-8354-c9967e9406c8)
  
  - The above graph was plotted after using Upper Confidence Bound Algorithm to feature engineer the target, the fixed threshold is 5000.

  - Word Cloud for Abstract Question
 
  - For output == 0 --> No Issue
 
    ![WhatsApp Image 2024-10-19 at 14 09 01_d9742308](https://github.com/user-attachments/assets/8ffa35f2-ed17-4ffa-885b-2987a388bc1d)

  - For output == 1 --> Issue
 
    ![WhatsApp Image 2024-10-19 at 14 09 02_0ffa9bdb](https://github.com/user-attachments/assets/a4531274-7a5e-4cd2-b5c9-c9ead89423cd)

- **Phase-2**
  
- Before using sampling techniques to balance the classes

  ![WhatsApp Image 2024-10-19 at 13 52 28_20cd2dea](https://github.com/user-attachments/assets/d253ec37-9382-43a9-a052-b57fc7011a8d)
  
- After using sampling techniques to balance the classes
  
  ![WhatsApp Image 2024-10-19 at 13 52 28_be09c3ca](https://github.com/user-attachments/assets/f369270d-36db-4308-a1b0-1dcea9649da3)

- Word Cloud for Abstract Question

- For output == 0 --> Anxiety

  ![WhatsApp Image 2024-10-19 at 14 17 35_a6d87344](https://github.com/user-attachments/assets/1ecf9b8b-be87-433f-9cf4-60398edd515c)

- For output == 1 --> Depression

  ![WhatsApp Image 2024-10-19 at 14 17 35_566c124f](https://github.com/user-attachments/assets/9974d76c-287a-4e7e-a1dd-170c4a7916aa)

- For ouput == 2 --> Suicidal

     ![WhatsApp Image 2024-10-19 at 14 17 35_98dc4294](https://github.com/user-attachments/assets/91d1561c-de71-431d-a9bf-fe540dcc5ba6)

- For output == 3 --> Bipolar

   ![WhatsApp Image 2024-10-19 at 14 17 36_fe0d460c](https://github.com/user-attachments/assets/f1a78474-bcff-4a05-86b6-82ac011de493)

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
## Describing Solution
  - To help people identify what the patient is dealing with, we have decided to build an efficient AI/ML system that primarily determines the type of mental disorder the person has if any. This system is driven mostly by yes, no, maybe, or not sure type questions. The model also encompasses a component which requires the user to input a paragraph about how they have been feeling lately to further help determine if their mental health is disturbed. The model will have two crucial datasets, one of which is a corpus to help train the model to predict if they have a mental health issue based on their abstract response. The other is a mental health dataset that will help determine if they have an issue or not. Once the issue has been identified,  Another dataset (this dataset has 5 categorical targets one of them being normal) will narrow it down the issue encountered to one of 4 categories (Anxiety, Depression, Loneliness, and Stress) if the person turns out to have an issue. According to DSM-5 (Diagnostic and Statistical Manual of Mental Disorders), only Anxiety and Depression are classified as neurotic mental health disorders. Ultimately, the model serves as a very simple interface to help mediate an interim recommendation on what the person might be experiencing to then help them make an informed decision on seeking external help.
---
## Future Enhancements
  - 1) Collaboration with Experts
     - Partnership with Mental Health Professionals
          -  Collaborate with psychiatrists, therapists, and mental health organizations to refine the question sets and recommendations.
          -  Use their feedback to improve the reliability and acceptance of the system.
  - 2) Real-time Feedback and Adaptive Questioning
          -  Develop an intelligent, adaptive questionnaire system that adjusts the next question based on previous answers, making the diagnosis process more personalized and reducing irrelevant questions.
  - 3) Multi-Lingual Support
          -  Expand the system’s accessibility by incorporating multi-lingual support using language models that can handle multiple languages, broadening the potential user base globally.
  - 4) Anonymized Data Sharing for Research
          - Introduce a feature where users can opt to anonymously share their data with research institutions, helping to improve the system’s datasets and contributing to mental health research.
  - 5) Mobile Application Development
          - Scale the system into a user-friendly mobile app, ensuring seamless access, push notifications for follow-ups, and data storage for tracking users’ mental health trends over time.
  - 6) Integration of Pretrained NLP Models
          - Incorporate transformer-based models like BERT, RoBERTa, or GPT to better analyze paragraph responses and capture deeper context, improving the text analysis module’s performance and precision.

---
## Contributors
- Abhivanth Sivaprakash
- K Krish Sundaresh
- Bharat Kameswaran
- A Hari
- Sanjeev Krishna S
- Adil Roshan
---
