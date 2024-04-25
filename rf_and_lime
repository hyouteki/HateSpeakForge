from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import lime
import lime.lime_tabular

dataset = load_dataset("social_bias_frames")
dataframe = pd.DataFrame(dataset['train'])
data = dataframe

data['targetStereotype'] = data['targetStereotype'].apply(lambda x: 'empty' if x == "" or len(x) == 0 else 'non-empty')

X = data.drop(columns=["post","targetMinority","targetCategory","targetStereotype"], axis=1)
y = data["targetStereotype"]

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, n_jobs=30)

rf_classifier.feature_names = X_train.columns.tolist()

rf_classifier.fit(X_train, y_train)

joblib.dump(rf_classifier, 'rf_classifier_model.pkl')

y_pred = rf_classifier.predict(X_test)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), 
                                                   class_names=['empty','non-empty'], mode='classification')

from collections import defaultdict

num_samples = 100

aggregate_features = defaultdict(float)

for i in range(num_samples):
    exp = explainer.explain_instance(X_train.values[i], rf_classifier.predict_proba, num_features=len(X_train.columns))
    
    top_features = exp.as_list()
    
    for feature, importance in top_features:
        aggregate_features[feature] += abs(importance)

total_samples = min(num_samples, len(X_train))
average_features = {feature: importance / total_samples for feature, importance in aggregate_features.items()}

sorted_features = sorted(average_features.items(), key=lambda x: x[1], reverse=True)

print("Top Averaged Features:")
for feature, importance in sorted_features:
    print(feature, ": ", importance)
