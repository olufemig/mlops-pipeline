import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# load the dataset
data = pd.read_csv('screentime_analysis.csv')

# check for missing values and duplicates
print(data.isnull().sum())
print(data.duplicated().sum())

# convert Date column to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# encode the categorical 'App' column using one-hot encoding
data = pd.get_dummies(data, columns=['App'], drop_first=True)

# scale numerical features using MinMaxScaler
scaler = MinMaxScaler()
data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

# feature engineering
data['Previous_Day_Usage'] = data['Usage (minutes)'].shift(1)
data['Notifications_x_TimesOpened'] = data['Notifications'] * data['Times Opened']

# save the preprocessed data to a file
data.to_csv('preprocessed_screentime_analysis.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# split data into features and target variable
X = data.drop(columns=['Usage (minutes)', 'Date'])
y = data['Usage (minutes)']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')