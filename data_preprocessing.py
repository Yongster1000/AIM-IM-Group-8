import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import csv

def main(
base_file,
methyl_file,
random_state,
test_size,
include_methylation,
number_of_fold
): 
  base_df = pd.read_csv(base_file)
  categorical_features = ['SEX', 'edu', 'DM_FAM', 'smoking', 'DRK', 'betel', 'SPORT', 'cardio_b']
  continuous_features = ['AGE', 'SBP', 'DBP', 'HR', 'Weight', 'Height', 'BMI', 'WHR', 'T_CHO', 'TG', 'HDL', 'LDL']
  
  with open(methyl_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    list = [row for row in spamreader]
    #print(len(list[0][0].split(',')[1:-1]))
    gene_features = list[0][0].split(',')[1:-1]
  
  if include_methylation:
    methyl_df = pd.read_csv(methyl_file, index_col=0)
    df = pd.merge(base_df, methyl_df, on='CaseNo', how='left')
    attr_list = categorical_features + continuous_features + gene_features
  else:
    df = base_df
    attr_list = categorical_features + continuous_features

   ### Add label and remove HBA1C_b, FBG_B, DIABETES_SELF_b
  df['target'] = np.where((df['HBA1C_b'] >= 6.5) | (df['FBG_B'] >= 126)| (df['DIABETES_SELF_b'] == 1), 1, 0)
  df.drop('HBA1C_b', axis=1, inplace=True)
  df.drop('FBG_B', axis=1, inplace=True)
  df.drop('DIABETES_SELF_b', axis=1, inplace=True)
  ### Remove columns for follow up (out of scope)
  df.drop('DIABETES_SELF', axis=1, inplace=True)
  df.drop('basedate', axis=1, inplace=True)
  df.drop('followdate', axis=1, inplace=True)
  df.drop('HBA1C', axis=1, inplace=True)
  df.drop('FBG', axis=1, inplace=True)
  df.drop('cardio_f', axis=1, inplace=True)
  ### Remove ID
  df.drop('CaseNo', axis=1, inplace=True)
  ### Train Test Split
  df.to_csv("revised_clinical_with_new_methylation_data.csv", index= False)
  
  skf = StratifiedKFold(n_splits = number_of_fold, shuffle=True, random_state=random_state)
  
  X = df.loc[:, df.columns != 'target']
  X = X[attr_list]
  y = df['target']
  list_result = []
  print(X.shape, y.shape)
  #print(X['Unnamed: 0'])
  for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.copy().iloc[train_index], X.copy().iloc[test_index]
    y_train, y_test = y.copy().iloc[train_index], y.copy().iloc[test_index]

    #print(X_train.columns)
    #train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    X_train['target'] = y_train
    X_test['target'] = y_test
    train_df = X_train
    test_df = X_test
    #find_outlier
    ##### z_score
    #print('train_df =', train_df)
    #The Z-score calculates how far a data point is from the mean, measured in standard deviations. 
    # If a value’s Z-score is beyond a certain threshold (e.g., 3), it's flagged as an outlier. 
    # This method assumes data is normally distributed, making it suitable for symmetric, bell-shaped distributions.
    if include_methylation:
      continuous_features.extend(gene_features)
    
    for feature in continuous_features:
      z_scores = np.abs((train_df[feature] - train_df[feature].mean()) / train_df[feature].std())
      z_score_filtered = train_df[z_scores < 3]
    #### Interquartile Range (IQR) Method
    
    feature_data = train_df.dropna()
    for feature in continuous_features:
      Q1, Q3 = np.percentile(feature_data[feature] , [25, 75])
      IQR = Q3 - Q1
      IQR_filtered_data = feature_data[(feature_data[feature]  >= Q1 - 1.5 * IQR) & (feature_data[feature]  <= Q3 + 1.5 * IQR)]
    #### Median Absolute Deviation (MAD)
    '''MAD is a robust method based on the median and measures the absolute deviation of each value from the median. 
    Dividing by MAD (or scaled by 1.482 for a normal approximation) helps identify outliers by setting a threshold (e.g., 3 MAD).
    MAD is more robust than Z-score for non-normal distributions and is less influenced by extreme values.
    '''
    feature_data = train_df.dropna()
    for feature in continuous_features:
      median = np.median(feature_data[feature])
      mad = np.median(np.abs(feature_data[feature] - median))
      MAD_filtered_data = feature_data[np.abs(feature_data[feature] - median) / mad < 3]
    
    overlapping_indices = z_score_filtered.index.intersection(MAD_filtered_data.index).intersection(IQR_filtered_data.index)
    filtered_train_df = z_score_filtered.loc[overlapping_indices]

    ### Imputation
    fillna = {}
    train_df_columns = filtered_train_df.columns.to_list()

    for i in range(0,len(train_df_columns)):
      if train_df_columns[i] in categorical_features:
          fillna[train_df_columns[i]] = filtered_train_df[train_df_columns[i]].mode().values[0]
      if train_df_columns[i] in continuous_features:
          fillna[train_df_columns[i]] = filtered_train_df[train_df_columns[i]].mean()
    
    for index, row in train_df.iterrows():
      for j in train_df:
        if pd.isna(row[j]):
          train_df.loc[index, j] = fillna[j]
    for index, row in test_df.iterrows():
      for j in test_df:
        if pd.isna(row[j]):
          test_df.loc[index, j] = fillna[j]
    
    scaler = StandardScaler()
    train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])
    test_df[continuous_features] = scaler.fit_transform(test_df[continuous_features])
    list_result.append((train_df, test_df))
  #print(random_state)

  return list_result

def parse_args():
  """arguments"""
  config = {
    "base_file": "base.csv",
    "methyl_file": "methylation_filtered_final_new.csv",
    "random_state": 42,
    "test_size": 0.2,
    "include_methylation": True,
    "number_of_fold": 5
  }
  return config
if __name__ == "__main__":
    main(**parse_args())
     