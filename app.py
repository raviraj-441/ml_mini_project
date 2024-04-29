


from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Blood_samples_dataset_balanced_2(f).csv')
dt = pd.read_csv('blood_samples_dataset_test.csv')
df = pd.concat([df, dt], ignore_index=True)

le = LabelEncoder()
df[df.columns[-1]] = le.fit_transform(df[df.columns[-1]])

X = df.drop(columns=['Disease'])
y = df['Disease']

best_rf_model = RandomForestClassifier(max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=100)
best_rf_model.fit(X, y)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[feature]) for feature in X.columns]
    
    input_data = pd.DataFrame([features], columns=X.columns)

    prediction = best_rf_model.predict(input_data)
    
    result = le.inverse_transform(prediction)[0]
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
