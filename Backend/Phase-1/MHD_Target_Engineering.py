# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:20:36 2024

@author: asiva
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Mental Health Dataset.csv')

#%% Data Preprocessing

# Extract all columns except the first one (index) and remove missing values
X = df.iloc[:, 1:]
X.dropna(inplace=True)

# Remove the 'Country' column as it's not needed for analysis
X.drop(columns=['Country'], inplace=True)

# Define mappings for categorical columns to convert them into numerical values
mappings = {
    'Gender': {'Female': 2, 'Male': 1},
    'Occupation': {'Corporate': 2, 'Student': 3, 'Business': 2, 'Housewife': 3, 'Others': 1},
    'self_employed': {'Yes': 3, 'No': 1},
    'family_history': {'Yes': 3, 'No': 1},
    'treatment': {'Yes': 3, 'No': 1},
    'Days_Indoors': {'1-14 days': 1, 'Go out Every day': 1, 'More than 2 months': 3, '15-30 days': 2, '31-60 days': 2},
    'Growing_Stress': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Changes_Habits': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Mental_Health_History': {'Yes': 3, 'No': 1, 'Maybe': 2},
    'Mood_Swings': {'High': 3, 'Low': 1, 'Medium': 2},
    'Coping_Struggles': {'Yes': 3, 'No': 1},
    'Work_Interest': {'No': 3, 'Maybe': 2, 'Yes': 1},
    'Social_Weakness': {'Yes': 3, 'Maybe': 2, 'No': 1},
    'mental_health_interview': {'Yes': 3, 'Maybe': 2, 'No': 1},
    'care_options': {'Yes': 1, 'Not sure': 2, 'No': 3}
}

# Apply mappings to the relevant columns
for column, mapping in mappings.items():
    if column in X.columns:
        X[column] = X[column].map(mapping)

#%% Implementing Upper Confidence Bound (UCB) Algorithm

# Define the number of records (N) and features/questions (d)
N = len(X['Gender'])  # Number of records
d = len(X.columns)  # Number of questions/features

# Initialize lists for tracking selected questions, selections count, and rewards
questions_selected = []
numbers_of_selections = [0] * d  # How many times each feature has been selected
sums_of_rewards = [0] * d  # Total rewards for each feature
total_reward = 0

# UCB algorithm loop to select features based on their upper confidence bounds
for n in range(0, N):
    question = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            # Calculate average reward and confidence interval (upper bound)
            avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            # If no selections yet, set the upper bound very high to ensure initial selection
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            question = i
    questions_selected.append(question)
    numbers_of_selections[question] += 1
    reward = X.values[n, question]  # Reward is based on the feature value in the dataset
    sums_of_rewards[question] += reward
    total_reward += reward
    
#%% Visualize the Frequency of Selected Questions

# Plot a histogram showing the frequency of each selected feature/question over time
plt.hist(questions_selected, bins=15, edgecolor='black')  # 15 features, so use 15 bins
plt.xlabel('Questions')
plt.ylabel('Frequency')
plt.title('Question Selections Over Time')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Frequencies extracted from the graph (replace these with actual values)
frequencies = [10, 1500, 5, 10, 5, 150, 1400, 650, 500, 350, 250, 1200, 350, 5, 10]

#%% Apply Condition Based on Frequencies

# Convert DataFrame to NumPy array for faster computation
X_values = X.values

# Create a mask for features that have a selection frequency greater than 200
mask = (np.array(frequencies) > 200)

# Count how many features in each row have values > 1 where their frequency > 200
majority_counter = np.sum((X_values[:, mask] > 1), axis=1)

# Create the binary target based on a threshold: if the count is >= 6, target = 1; otherwise, target = 0
target = (majority_counter >= 6).astype(int)

#%% Save the Processed Data with Target

# Add the target column to the DataFrame
X['Target'] = target

# Save the final DataFrame to a CSV file
X.to_csv('MHD_Final_Feature_Engineered.csv')
