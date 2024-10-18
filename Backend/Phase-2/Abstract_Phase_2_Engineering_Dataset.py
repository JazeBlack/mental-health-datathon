import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from a CSV file and drop the unnecessary 'Unnamed: 0' column
df = pd.read_csv('Combined Data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# Filter out rows where 'status' is 'Normal', 'Stress', or 'Personality disorder'
# Drop any rows with missing values (NaN)
df = df[~df['status'].isin(['Normal', 'Stress', 'Personality disorder'])]
df = df.dropna()

# Reset the index after dropping rows to keep the index sequential
df.reset_index(drop=True, inplace=True)

#%%

# Visualize the distribution of different 'status' categories in the dataset
# Group the data by 'status' and count the number of statements per status
data = df.groupby('status')['statement'].count().sort_values().reset_index()

# Plot a bar chart to show the distribution of 'status' categories
plt.figure(figsize=(10,5))
sns.barplot(data=data, x='status', y='statement')
plt.xlabel('Status')
plt.ylabel('Count')

#%%

# Mapping target values (disorder labels) to numerical values for machine learning models
d_mapping_disorders = {'Anxiety': 0, 'Depression': 1, 'Suicidal': 2, 'Bipolar': 3}
df['status'] = df['status'].map(d_mapping_disorders)

#%%

from sklearn.utils import resample

# Set a benchmark value for upsampling and downsampling based on the inference from the graph at hand
benchmark = 9000

# Separate the dataset into different subsets based on the 'status' (target) values
bipolar_df = df[df['status'] == 3]
anxiety_df = df[df['status'] == 0]
suicidal_df = df[df['status'] == 2]
depression_df = df[df['status'] == 1]

#%%

# Upsample the minority classes (Bipolar and Anxiety) to reach the benchmark sample size
bipolar_upsampled = resample(bipolar_df, replace=True, n_samples=benchmark, random_state=42)
anxiety_upsampled = resample(anxiety_df, replace=True, n_samples=benchmark, random_state=42)

# Downsample the majority classes (Depression and Suicidal) to reduce them to the benchmark size
depression_downsampled = resample(depression_df, replace=False, n_samples=benchmark, random_state=42)
suicidal_downsampled = resample(suicidal_df, replace=False, n_samples=benchmark, random_state=42)

# Combine the upsampled and downsampled dataframes into a new balanced dataset
balanced_df = pd.concat([bipolar_upsampled, anxiety_upsampled, suicidal_downsampled, depression_downsampled])

# Shuffle the combined dataset to mix the rows randomly
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Print the distribution of 'status' in the new balanced dataset
print(balanced_df['status'].value_counts())

#%%

# Visualize the distribution of the balanced dataset after resampling
data = balanced_df.groupby('status')['statement'].count().sort_values().reset_index()

# Plot a bar chart to show the distribution of 'status' categories after resampling
plt.figure(figsize=(10,5))
sns.barplot(data=data, x='status', y='statement')
plt.xlabel('Status')
plt.ylabel('Count')

#%%
# Save the balanced dataset to a CSV file for future use
balanced_df.to_csv('Multi_Class_Main_Final.csv', index=False)
