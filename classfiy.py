# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

breast_data = pd.read_csv('./breast_data/data.csv')
# print(breast_data.head())

X = breast_data.iloc[:,2:-1].values
y = breast_data['diagnosis']
# print(y)

encode_y = LabelEncoder()
y = encode_y.fit_transform(y)
# print(y)

training_set,validation_test,training_labels,validation_labels = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=100)

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(training_set,training_labels)
y_pred = classifier.predict(validation_test)
print("Accuracy :" + str(classifier.score(validation_test,validation_labels)) )

test_sample = np.array([[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]])
y_pred_new = classifier.predict(test_sample)
if y_pred_new == 1:
    print("Malignant")
elif y_pred_new == 0:
    print("Benigne")