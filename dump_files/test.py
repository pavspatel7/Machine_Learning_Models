from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import ast
import pandas as pd

df = pd.read_excel('training_dataSet.xlsx')

df['encoding'] = df['encoding'].apply(ast.literal_eval)
df['multi_classification'] = df['multi_classification'].apply(ast.literal_eval)

# Assuming the column you want to print is named 'Encoding'
column_values_encoding = df['encoding'].tolist()
column_values_multi_classification = df['multi_classification'].tolist()

X = column_values_encoding
y = column_values_multi_classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # 10% data is used for testing

# Use softmax regression by setting multi_class='multinomial' and solver='lbfgs'
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, train_predictions)}")

test_predictions = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, test_predictions)}")
print(classification_report(y_test, test_predictions))

# Print the number of iterations
print(f"Number of iterations: {model.n_iter_}")
