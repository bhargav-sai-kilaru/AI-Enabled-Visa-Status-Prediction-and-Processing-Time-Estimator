import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Load Data (assuming the file is in the working directory)
# Note: Given the 154 columns, we select relevant ones to save memory
cols = ['case_status', 'class_of_admission', 'pw_amount_9089', 
        'employer_name', 'job_info_job_title', 'foreign_worker_info_education']
df = pd.read_csv('us_perm_visas.csv', usecols=cols, low_memory=False)

# 2. Data Cleaning
# Focus on Certified vs Denied
df = df[df['case_status'].isin(['Certified', 'Certified-Expired', 'Denied'])]
df['target'] = df['case_status'].apply(lambda x: 1 if 'Certified' in x else 0)

# Clean Wage (remove commas/convert to numeric)
df['pw_amount_9089'] = pd.to_numeric(df['pw_amount_9089'].str.replace(',', ''), errors='coerce')
df = df.dropna(subset=['pw_amount_9089', 'class_of_admission'])

# 3. H-1B Comparison
h1b_stats = df.groupby('class_of_admission')['target'].mean().sort_values(ascending=False)
print("Approval Rate by Visa Type:")
print(h1b_stats.head(10))

# 4. Feature Engineering for Prediction
le = LabelEncoder()
df['employer_enc'] = le.fit_transform(df['employer_name'].astype(str))
df['job_enc'] = le.fit_transform(df['job_info_job_title'].astype(str))
df['edu_enc'] = le.fit_transform(df['foreign_worker_info_education'].astype(str))

X = df[['pw_amount_9089', 'employer_enc', 'job_enc', 'edu_enc']]
y = df['target']

# 5. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Output Results
predictions = model.predict(X_test)
print("\nVisa Decision Prediction Report:")
print(classification_report(y_test, predictions))
