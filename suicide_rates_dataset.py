import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

from google.colab import drive
drive.mount('/content/drive/')

GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'ML-Project' 
GOOGLE_DRIVE_PATH = os.path.join('drive', 'MyDrive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))

data = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH,'age_std_suicide_rates_1990-2022.csv'))

features = ['Year', 'GDP', 'CauseSpecificDeathPercentage', 'EmploymentPopulationRatio', 'InflationRate','CountryCode','Sex','Population','StdDeathRate','GNI']
data.dropna(inplace=True)
X = data[features]
y = data['SuicideCount']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(data['CountryCode'])
data['CountryCode'] = encoded_values
encoded_values = label_encoder.fit_transform(data['Sex'])
data['Sex'] = encoded_values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')
accuracy = 100 * (1 - (mae / y_test.mean()))
print(f'Accuracy: {accuracy:.2f}%')
