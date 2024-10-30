import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# read the data
df = pd.read_csv('/Users/jadefok/113_1/AIIM/final_project/Diabetes data set/base.csv')

# features
categorical_features = ['SEX', 'edu', 'DM_FAM', 'smoking', 'DRK', 'betel', 'SPORT', 'cardio_b']
continuous_features = ['AGE', 'SBP', 'DBP', 'HR', 'Weight', 'Height', 'BMI', 'WHR', 'T_CHO', 'TG', 'HDL', 'LDL']

# relabel the diabetes
df['target'] = np.where((df['HBA1C_b'] >= 6.5) | (df['FBG_B'] >= 126)| (df['DIABETES_SELF_b'] == 1), 1, 0)

# drop label related columns
df.drop('HBA1C_b', axis=1, inplace=True)
df.drop('FBG_B', axis=1, inplace=True)
df.drop('DIABETES_SELF_b', axis=1, inplace=True)

# remove non related columns (follow up)
df.drop('DIABETES_SELF', axis=1, inplace=True)
df.drop('basedate', axis=1, inplace=True)
df.drop('followdate', axis=1, inplace=True)
df.drop('HBA1C', axis=1, inplace=True)
df.drop('FBG', axis=1, inplace=True)
df.drop('cardio_f', axis=1, inplace=True)

# remove ID
df.drop('CaseNo', axis=1, inplace=True)

# train test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# find outlier

# find outlier : use z-score
for feature in continuous_features:
    z_scores = np.abs((train_df[feature] - train_df[feature].mean()) / train_df[feature].std())
    z_score_filtered = train_df[z_scores < 3]

# find outlier : IQR
for feature in continuous_features:
    Q1, Q3 = np.percentile(train_df[feature] , [25, 75])
    IQR = Q3 - Q1
    IQR_filtered_data = train_df[(train_df[feature]  >= Q1 - 1.5 * IQR) & (train_df[feature]  <= Q3 + 1.5 * IQR)]

# find outlier : MAD
for feature in continuous_features:
    median = np.median(train_df[feature])
    mad = np.median(np.abs(train_df[feature] - median))
    MAD_filtered_data = train_df[np.abs(train_df[feature] - median) / mad < 3]

# imputation : filter outlier
overlapping_indices = z_score_filtered.index.intersection(MAD_filtered_data.index).intersection(IQR_filtered_data.index)
filtered_train_df = z_score_filtered.loc[overlapping_indices]

# imputation : find mean and mode
fillna = {}

train_df_columns = filtered_train_df.columns.to_list()

for i in range(0,len(train_df_columns)):
  if train_df_columns[i] in categorical_features:
    fillna[train_df_columns[i]] = filtered_train_df[train_df_columns[i]].mode().values[0]
  if train_df_columns[i] in continuous_features:
    fillna[train_df_columns[i]] = filtered_train_df[train_df_columns[i]].mean()

# imputation
for index, row in train_df.iterrows():
  for j in train_df:
      if pd.isna(row[j]):
        train_df.loc[index, j] = fillna[j]
for index, row in test_df.iterrows():
  for j in test_df:
      if pd.isna(row[j]):
        test_df.loc[index, j] = fillna[j]

# normalization
scaler = StandardScaler()
train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])
test_df[continuous_features] = scaler.fit_transform(test_df[continuous_features])

# save to csv
train_df.to_csv('train_standard.csv', index=False)
test_df.to_csv('test_standard.csv', index=False)