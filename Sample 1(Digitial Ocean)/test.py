import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


df = pd.read_csv('salary_data.csv')
model = IsolationForest(n_estimators=50, max_samples='auto',max_features=1)
model.fit(df[['salary']])
df['scores'] = model.decision_function(df[['salary']])
df['anomaly'] = model.predict(df[['salary']])

anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)


outliers_counter = len(df[df['salary'] > 99999])

print('Model Prediction Accuracy: ', 100 * (list(df['anomaly']).count(-1) / outliers_counter))