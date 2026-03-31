
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
# Note: Update the file path as needed for your environment
try:
    df = pd.read_csv('StudentsPerformance.csv')
    print("Dataset loaded successfully!")
    print(f"Shape of the dataset: {df.shape}")
except FileNotFoundError:
    print("Dataset file not found. Please ensure the file is in the correct location.")
    exit()

# 1. Data Exploration and Cleaning
print("\n--- DATA EXPLORATION ---")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# Check for and handle duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# 2. Data Preprocessing
# Ensure consistent column names (lowercase, no spaces)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# For demonstration, let's rename some columns to ensure consistency
# Update these based on your actual column names
if 'math_score' not in df.columns and 'math score' in df.columns:
    df.rename(columns={'math score': 'math_score'}, inplace=True)
if 'reading_score' not in df.columns and 'reading score' in df.columns:
    df.rename(columns={'reading score': 'reading_score'}, inplace=True)
if 'writing_score' not in df.columns and 'writing score' in df.columns:
    df.rename(columns={'writing score': 'writing_score'}, inplace=True)
if 'test_preparation_course' not in df.columns and 'test preparation course' in df.columns:
    df.rename(columns={'test preparation course': 'test_preparation_course'}, inplace=True)
if 'parental_level_of_education' not in df.columns and 'parental level of education' in df.columns:
    df.rename(columns={'parental level of education': 'parental_level_of_education'}, inplace=True)

# Create a total score column
if 'math_score' in df.columns and 'reading_score' in df.columns and 'writing_score' in df.columns:
    df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
    df['average_score'] = df['total_score'] / 3

print("\nUpdated column names:")
print(df.columns.tolist())

# 3. Descriptive Analysis
print("\n--- DESCRIPTIVE ANALYSIS ---")

# Gender-based analysis
print("\nGender Distribution:")
gender_count = df['gender'].value_counts()
print(gender_count)

print("\nPerformance by Gender:")
gender_performance = df.groupby('gender')[['math_score', 'reading_score', 'writing_score', 'average_score']].mean()
print(gender_performance)

# Test preparation impact
print("\nTest Preparation Course Impact:")
test_prep_performance = df.groupby('test_preparation_course')[['math_score', 'reading_score', 'writing_score', 'average_score']].mean()
print(test_prep_performance)

# Lunch type impact
print("\nLunch Type Impact:")
lunch_performance = df.groupby('lunch')[['math_score', 'reading_score', 'writing_score', 'average_score']].mean()
print(lunch_performance)

# Education level impact
print("\nParental Education Impact:")
education_performance = df.groupby('parental_level_of_education')[['math_score', 'reading_score', 'writing_score', 'average_score']].mean().sort_values(by='average_score', ascending=False)
print(education_performance)

# 4. Visualization
print("\n--- CREATING VISUALIZATIONS ---")

# Figure 1: Gender Performance Comparison
plt.figure(figsize=(14, 8))
gender_scores = df.groupby('gender')[['math_score', 'reading_score', 'writing_score']].mean().reset_index()
gender_scores_melted = pd.melt(gender_scores, id_vars='gender', var_name='subject', value_name='score')
sns.barplot(x='gender', y='score', hue='subject', data=gender_scores_melted)
plt.title('Average Scores by Gender and Subject', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Average Score', fontsize=14)
plt.legend(title='Subject')
plt.savefig('gender_performance.png')
plt.close()
print("Gender performance comparison plot saved!")

# Figure 2: Test Preparation Impact
plt.figure(figsize=(14, 8))
test_prep_scores = df.groupby('test_preparation_course')[['math_score', 'reading_score', 'writing_score']].mean().reset_index()
test_prep_scores_melted = pd.melt(test_prep_scores, id_vars='test_preparation_course', var_name='subject', value_name='score')
sns.barplot(x='test_preparation_course', y='score', hue='subject', data=test_prep_scores_melted)
plt.title('Impact of Test Preparation on Scores', fontsize=16)
plt.xlabel('Test Preparation Course', fontsize=14)
plt.ylabel('Average Score', fontsize=14)
plt.legend(title='Subject')
plt.savefig('test_prep_impact.png')
plt.close()
print("Test preparation impact plot saved!")

# Figure 3: Lunch Type Impact
plt.figure(figsize=(14, 8))
lunch_scores = df.groupby('lunch')[['math_score', 'reading_score', 'writing_score']].mean().reset_index()
lunch_scores_melted = pd.melt(lunch_scores, id_vars='lunch', var_name='subject', value_name='score')
sns.barplot(x='lunch', y='score', hue='subject', data=lunch_scores_melted)
plt.title('Impact of Lunch Type on Scores', fontsize=16)
plt.xlabel('Lunch Type', fontsize=14)
plt.ylabel('Average Score', fontsize=14)
plt.legend(title='Subject')
plt.savefig('lunch_impact.png')
plt.close()
print("Lunch type impact plot saved!")

# Figure 4: Parental Education Impact
plt.figure(figsize=(14, 10))
sns.boxplot(x='parental_level_of_education', y='average_score', data=df, order=education_performance.index)
plt.title('Impact of Parental Education on Average Score', fontsize=16)
plt.xlabel('Parental Level of Education', fontsize=14)
plt.ylabel('Average Score', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('education_impact.png')
plt.close()
print("Parental education impact plot saved!")

# Figure 5: Score Distribution
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['math_score'], kde=True)
plt.title('Math Score Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['reading_score'], kde=True)
plt.title('Reading Score Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['writing_score'], kde=True)
plt.title('Writing Score Distribution')

plt.tight_layout()
plt.savefig('score_distributions.png')
plt.close()
print("Score distributions plot saved!")

# Figure 6: Correlation Heatmap
plt.figure(figsize=(10, 8))
# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Scores', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved!")

# 5. Key Findings Summary
print("\n--- KEY FINDINGS ---")

# Gender comparison
gender_diff = gender_performance.iloc[0] - gender_performance.iloc[1]
better_gender = "Female" if gender_performance.loc["female", "average_score"] > gender_performance.loc["male", "average_score"] else "Male"
print(f"1. {better_gender} students perform better on average by {abs(gender_diff['average_score']):.2f} points.")

# Test preparation impact
test_prep_diff = test_prep_performance.loc["completed"] - test_prep_performance.loc["none"]
print(f"2. Students who completed the test preparation course score {test_prep_diff['average_score']:.2f} points higher on average.")

# Lunch type impact
lunch_types = lunch_performance.index.tolist()
better_lunch = lunch_types[0] if lunch_performance.iloc[0]["average_score"] > lunch_performance.iloc[1]["average_score"] else lunch_types[1]
lunch_diff = abs(lunch_performance.iloc[0]["average_score"] - lunch_performance.iloc[1]["average_score"])
print(f"3. Students with '{better_lunch}' lunch perform better by {lunch_diff:.2f} points on average.")

# Education impact
best_education = education_performance.index[0]
worst_education = education_performance.index[-1]
edu_diff = education_performance.iloc[0]["average_score"] - education_performance.iloc[-1]["average_score"]
print(f"4. Students whose parents have '{best_education}' education level perform best, while '{worst_education}' perform worst.")
print(f"   The difference between highest and lowest educational background is {edu_diff:.2f} points.")

# Subject difficulty
subject_means = df[['math_score', 'reading_score', 'writing_score']].mean()
easiest_subject = subject_means.idxmax().replace('_score', '')
hardest_subject = subject_means.idxmin().replace('_score', '')
print(f"5. {easiest_subject.capitalize()} appears to be the easiest subject, while {hardest_subject.capitalize()} appears to be the most challenging.")

print("\nAnalysis complete! Check the saved visualization files for graphical insights.")