import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./data/adult.data')
data.replace(' ?', np.nan,inplace=True)
data.replace(' <=50K',0,inplace = True)
data.replace(' >50K',1,inplace=True)

labels = data['inc-cat']
data = data.drop(['inc-cat'],axis = 1)

num_cols = ['fnlwgt','edu-num','cap-gain','cap-loss','work-hrs']
cat_cols = ['workclass','edu','mar-stat','occ','rel','race','sex','country']

from sklearn.impute import SimpleImputer

catimp = SimpleImputer(strategy='most_frequent')
catimp.fit(data[cat_cols])

data[cat_cols] = catimp.transform(data[cat_cols])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data[num_cols])
data[num_cols] = scaler.transform(data[num_cols])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
ohe.fit(data[cat_cols])

frame = ohe.transform(data[cat_cols])
frame = pd.DataFrame(frame,
                     index=data.index,

                     columns = ohe.get_feature_names_out(cat_cols))
data = data.drop(cat_cols,axis = 1)
data = pd.concat([data,frame],axis = 1)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(data,labels)
model2 = DecisionTreeClassifier

test = pd.read_csv('./data/adult.test')
test.replace(' ?',np.nan,inplace=True)
test.replace(' <=50K.',0,inplace = True)
test.replace(' >50K.',1,inplace=True)

actual = test['inc-cat']

test.drop('inc-cat',axis=1,inplace=True)

test[num_cols] = scaler.transform(test[num_cols])
test[cat_cols] = catimp.transform(test[cat_cols])

frame2 = pd.DataFrame(ohe.transform(test[cat_cols]),index = test.index,columns=ohe.get_feature_names_out(cat_cols))
test.drop(cat_cols,axis = 1,inplace=True)
test = pd.concat([test,frame2],axis = 1)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model,data,labels,cv = 10)
print(scores)

scores = cross_val_score(model2,data,labels,cv = 10)
print(scores)
output = model.predict(test)
error = 0
for i,j in zip(actual,output):
    error += abs(i-j)
print(error)
print(len(test))
