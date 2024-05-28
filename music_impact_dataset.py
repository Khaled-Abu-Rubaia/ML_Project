import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler

from google.colab import drive
drive.mount('/content/drive/')


GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'ML-Project'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'MyDrive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))

data = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH,'mxmh_survey_results.csv'))

features = ['Age', 'Primary streaming service', 'Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory', 'Foreign languages', 'BPM', 'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]', 'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]', 'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]', 'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]', 'OCD']
data.drop(['Timestamp', 'Music effects'], axis=1, inplace=True)
data.dropna(inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_features = ['Primary streaming service', 'While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory', 'Foreign languages', 'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]', 'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]', 'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]', 'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]']
for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])

# Define the target and features
features_y = ['Anxiety', 'Depression', 'Insomnia']
y = data[features_y]
X = data[features]
# Scale the features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=features)

# Initialize LOOCV
loo = LeaveOneOut()
mae_anxiety_lr, mae_depression_lr, mae_insomnia_lr = [], [], []

# Define the SVR models with default parameters
svr_anxiety = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_depression = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_insomnia = SVR(kernel='rbf', C=1.0, gamma='scale')

# Calculate average MAE for SVR
avg_mae_anxiety_svr = np.mean(mae_anxiety_svr)
avg_mae_depression_svr = np.mean(mae_depression_svr)
avg_mae_insomnia_svr = np.mean(mae_insomnia_svr)

# Calculate accuracy
accuracy_anxiety_svr = 100 * (1 - (avg_mae_anxiety_svr / y['Anxiety'].mean()))
accuracy_depression_svr = 100 * (1 - (avg_mae_depression_svr / y['Depression'].mean()))
accuracy_insomnia_svr = 100 * (1 - (avg_mae_insomnia_svr / y['Insomnia'].mean()))

# Print results
print(f"Mean Absolute Error - Anxiety: {avg_mae_anxiety_svr}")
print(f'Accuracy - Anxiety: {accuracy_anxiety_svr:.2f}%')
print(f"Mean Absolute Error - Depression: {avg_mae_depression_svr}")
print(f'Accuracy - Depression: {accuracy_depression_svr:.2f}%')
print(f"Mean Absolute Error - Insomnia: {avg_mae_insomnia_svr}")
print(f'Accuracy - Insomnia: {accuracy_insomnia_svr:.2f}%')
