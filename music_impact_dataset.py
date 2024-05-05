import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import os

from google.colab import drive
drive.mount('/content/drive/')


GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'ML-Project'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'MyDrive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))

data = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH,'mxmh_survey_results.csv'))

features = ['Age','Primary streaming service','Hours per day','While working','Instrumentalist','Composer','Fav genre','Exploratory','Foreign languages'
,'BPM','Frequency [Classical]','Frequency [Country]','Frequency [EDM]','Frequency [Folk]','Frequency [Gospel]','Frequency [Hip hop]','Frequency [Jazz]',
'Frequency [K pop]','Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]','Frequency [Pop]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]',
'Frequency [Video game music]','OCD']
data.drop('Timestamp', axis=1)
data.drop('Music effects', axis=1)
data.dropna(inplace=True)
features_y = ['Anxiety','Depression','Insomnia']
label_encoder = LabelEncoder()
data['Primary streaming service'] = label_encoder.fit_transform(data['Primary streaming service'])
data['While working'] = label_encoder.fit_transform(data['While working'])
data['Instrumentalist'] = label_encoder.fit_transform(data['Instrumentalist'])
data['Composer'] = label_encoder.fit_transform(data['Composer'])
data['Fav genre'] = label_encoder.fit_transform(data['Fav genre'])
data['Exploratory'] = label_encoder.fit_transform(data['Exploratory'])
data['Foreign languages'] = label_encoder.fit_transform(data['Foreign languages'])
data['Frequency [Classical]'] = label_encoder.fit_transform(data['Frequency [Classical]'])
data['Frequency [Country]'] = label_encoder.fit_transform(data['Frequency [Country]'])
data['Frequency [EDM]'] = label_encoder.fit_transform(data['Frequency [EDM]'])
data['Frequency [Folk]'] = label_encoder.fit_transform(data['Frequency [Folk]'])
data['Frequency [Gospel]'] = label_encoder.fit_transform(data['Frequency [Gospel]'])
data['Frequency [Hip hop]'] = label_encoder.fit_transform(data['Frequency [Hip hop]'])
data['Frequency [Jazz]'] = label_encoder.fit_transform(data['Frequency [Jazz]'])
data['Frequency [K pop]'] = label_encoder.fit_transform(data['Frequency [K pop]'])
data['Frequency [Latin]'] = label_encoder.fit_transform(data['Frequency [Latin]'])
data['Frequency [Lofi]'] = label_encoder.fit_transform(data['Frequency [Lofi]'])
data['Frequency [Metal]'] = label_encoder.fit_transform(data['Frequency [Metal]'])
data['Frequency [Pop]'] = label_encoder.fit_transform(data['Frequency [Pop]'])
data['Frequency [R&B]'] = label_encoder.fit_transform(data['Frequency [R&B]'])
data['Frequency [Rap]'] = label_encoder.fit_transform(data['Frequency [Rap]'])
data['Frequency [Rock]'] = label_encoder.fit_transform(data['Frequency [Rock]'])
data['Frequency [Video game music]'] = label_encoder.fit_transform(data['Frequency [Video game music]'])
y = data[features_y]
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


y_pred_df = pd.DataFrame(y_pred, columns=['Anxiety', 'Depression', 'Insomnia'])
mae_anxiety = mean_absolute_error(y_test['Anxiety'], y_pred_df['Anxiety'])
mae_depression = mean_absolute_error(y_test['Depression'], y_pred_df['Depression'])
mae_insomnia = mean_absolute_error(y_test['Insomnia'], y_pred_df['Insomnia'])
accuracy_Anxiety = 100 * (1 - (mae_anxiety / y_test['Anxiety'].mean()))
accuracy_Depression = 100 * (1 - (mae_depression / y_test['Depression'].mean()))
accuracy_Insomnia = 100 * (1 - (mae_insomnia / y_test['Insomnia'].mean()))
print(f"Mean Squared Error_Anxiety: {mae_anxiety}")
print(f'Accuracy: {accuracy_Anxiety:.2f}%')
print(f"Mean Squared Error_Depression: {mae_depression}")
print(f'Accuracy: {accuracy_Depression:.2f}%')
print(f"Mean Squared Error_Insomnia: {mae_insomnia}")
print(f'Accuracy: {accuracy_Insomnia:.2f}%')