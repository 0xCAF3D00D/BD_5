from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
 
source = "vac_classes.csv"
output = "output.csv"
classes_file = "classes.csv"
detailed = "classes-detailed.csv"
 
df = pd.read_csv(source)

encoder = preprocessing.LabelEncoder()
 
classes = pd.read_csv(classes_file)

nmdf = pd.DataFrame(columns=['clazz', '№'])
nmdf['№'] = encoder.fit_transform(classes['clazz'])
nmdf['clazz'] = classes['clazz']
nmdf.to_csv(detailed)
 
# 2.2
df_train = df[df.city != 'Сочи']
df_test = df[df.city == 'Сочи']

df_train = df_train.apply(encoder.fit_transform)
x_values = df_train.iloc[:,:-1].values
y_values = df_train['clazz'] 

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size = 0.30)
 
print('****KNeighborsClassifier****')
k_neighbors = KNeighborsClassifier()    
k_neighbors.fit(x_train, y_train) 
train_predictions = k_neighbors.predict(x_test) 
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))

print('****RandomForestClassifier****')
random_forest = RandomForestClassifier(max_depth = 5)
random_forest.fit(x_train, y_train)
train_predictions = random_forest.predict(x_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))

print('****GradientBoostingClassifier****')
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
train_predictions = gb.predict(x_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))

print('****Test****')
df_test = df_test.apply(encoder.fit_transform)
x_values = df_test.iloc[:,:-1].values
y_values = df_test['clazz'] 
 
test_predictions = gb.predict(x_values)
acc = accuracy_score(test_predictions, y_values)
print("Accuracy: {:.4%}".format(acc))

df_test["actual"] = y_values
df_test["predicted"] = test_predictions
df_test.to_csv(output)