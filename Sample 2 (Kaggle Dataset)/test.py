import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime


df = pd.read_csv("Employee_Login_Logout_TimeSeries.csv")

# Remove rows that are "Public Holiday/Weekend"
df = df[df['In'] != "Public Holiday/Weekend"]

#Convert HH:MM to minutes from midnight
def time_to_minutes(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M")
    return (time_obj.hour * 60) + time_obj.minute

df['In'] = df['In'].apply(time_to_minutes)
df['Out'] = df['Out'].apply(time_to_minutes)

#  #Convert Date to numerical value (e.g., number of days since 2000-01-01)
# def date_to_days(date_str):
#     base_date = datetime(2000, 1, 1)
#     date_obj = datetime.strptime(date_str, "%d %m %Y")  # Adjust if your format is different
#     return (date_obj - base_date).days

# df['Date'] = df['Date'].apply(date_to_days)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['In','Out']])

model = IsolationForest(n_estimators=100,max_samples="auto")
model.fit(scaled_features)
df['scores'] = model.decision_function(scaled_features)
df['anomaly'] = model.predict(scaled_features)

# model = IsolationForest(n_estimators=100,max_samples='auto')
# model.fit(df[['In','Out']])
# df['anomaly_scores'] = model.decision_function(df[['In','Out']])
# df['anomaly'] = model.predict(df[['In','Out']])


plt.scatter(df['In'], df['Out'], c=df['anomaly'], cmap='coolwarm')
plt.xlabel('In')
plt.ylabel('Out')
plt.title('Anomalies in Attendance Data')
plt.show()


# df['scores'] = model.decision_function(df[['salary']])
# df['anomaly'] = model.predict(df[['salary']])

# anomaly=df.loc[df['anomaly']==-1]
# anomaly_index=list(anomaly.index)



# normalization = (x - min value) / (range(max value - min value))
# x = 10
# min value =  3
# max value = 50
# range = 47

# normalization = (10-3) / 47
#               = 7 / 47
#               = 0.1489
