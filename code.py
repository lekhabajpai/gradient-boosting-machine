# --------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

data = pd.read_csv(path)
print(data.head())

# Explore the data 
print(data['sex'].value_counts())
print(data.loc[data['sex'] == 'Female', 'age'].mean())
print(float((data['native-country'] == 'Germany').sum()/ data.shape[0]))

# mean and standard deviation of their age
ages1 = data.loc[data['salary'] == '>50K', 'age'].mean()
ages2 = data.loc[data['salary'] == '<=50K', 'age'].mean()
print(round(ages1.mean(), 1), round(ages1.std(), 1))

# Display the statistics of age for each gender of all the races (race feature).
for (race, sex), sub_df in data.groupby(['race','sex']):
    print("Race : {0}, sex: {1}".format(race, sex))
    print(sub_df['age'].describe())

# encoding the categorical features.
data['salary'] = data['salary'].apply(lambda x:1 if x == '>50K' else 0)
X = pd.get_dummies(data.drop('salary', 1))
y = data[['salary']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Split the data and apply decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train1, y_train1)
y_predict = clf.predict(X_val)
y_pred1 = clf.predict(X_test)
print("Accuracy on validation data", accuracy_score(y_val, y_predict))
print("Accuracy on test data", accuracy_score(y_test, y_pred1))

bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=100, random_state=0)
bagging_clf.fit(X_train1, y_train1)
y_predict = bagging_clf.predict(X_val)
y_pred1 = bagging_clf.predict(X_test)
print("Accuracy on validation data", accuracy_score(y_val, y_predict))
print("Accuracy on test data", accuracy_score(y_test, y_pred1))

pasting_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=100, random_state=0, bootstrap=False)
pasting_clf.fit(X_train1, y_train1)
y_predict = pasting_clf.predict(X_val)
y_pred1 = pasting_clf.predict(X_test)
print("Accuracy on validation data", accuracy_score(y_val, y_predict))
print("Accuracy on test data", accuracy_score(y_test, y_pred1))

random_clf = RandomForestClassifier(n_estimators=100, random_state=0)
random_clf.fit(X_train1, y_train1)
y_predict = random_clf.predict(X_val)
y_pred1 = random_clf.predict(X_test)
print("Accuracy on validation data", accuracy_score(y_val, y_predict))
print("Accuracy on test data", accuracy_score(y_test, y_pred1))

# Perform the boosting task
model_10 = GradientBoostingClassifier(n_estimators=10, max_depth=6, random_state=0).fit(X_train1, y_train1)
model_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6, random_state=0).fit(X_train1, y_train1)
model_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=0).fit(X_train1, y_train1)
#print("Accuracy validation for model_10", model_10.score(X_val, y_val))
#print("Accuracy validation for model_50", model_50.score(X_val, y_val))
#print("Accuracy validation for model_100", model_100.score(X_val, y_val))


#print("Accuracy validation for model_10", model_10.score(X_val, y_val))
#print("Accuracy validation for model_50", model_50.score(X_val, y_val))
#print("Accuracy validation for model_100", model_100.score(X_val, y_val))

#  plot a bar plot of the model's top 10 features with it's feature importance score
feat_imp = pd.DataFrame({'importance':model_100.feature_importances_})
feat_imp['feature'] = X_train1.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp = feat_imp.iloc[:10]
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title="Feature Importance", figsize=(10,5))
plt.xlabel("Feature Importance Score")
plt.show()

#  Plot the training and testing error vs. number of trees
train_err_10 = 1-model_10.score(X_train, y_train)
train_err_50 = 1-model_50.score(X_train, y_train)
train_err_100 = 1-model_100.score(X_train, y_train)

training_errors = [train_err_10, train_err_50, train_err_100]

testing_err_10 = 1-model_10.score(X_test, y_test)
testing_err_50 = 1-model_50.score(X_test, y_test)
testing_err_100 = 1-model_100.score(X_test, y_test)

testing_errors = [testing_err_10, testing_err_50, testing_err_100]

validation_err_10 = 1-model_10.score(X_val,y_val)
validation_err_50 = 1-model_50.score(X_val,y_val)
validation_err_100 = 1-model_100.score(X_val,y_val)

validation_errors = [validation_err_10, validation_err_50, validation_err_100]

plt.figure(figsize=(10,4))
plt.plot([10,50,100], training_errors, linewidth=4.0,label='Training Error')
plt.plot([10,50,100], validation_errors, linewidth=4.0,label='Validation Error')
plt.title('Error Vs no of Trees')
plt.xlabel('no of Trees')
plt.ylabel('Classification error')
plt.legend(['Training error', 'validation error'])
plt.show()


