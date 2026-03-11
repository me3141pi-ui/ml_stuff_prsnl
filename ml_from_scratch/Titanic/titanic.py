import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit


data = pd.read_csv('./Data/Titanic-Dataset.csv')
data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1)

strat_cat = data['Pclass']

split = StratifiedShuffleSplit(n_splits = 1,test_size = 0.2,random_state=42)

for train_id , test_id in split.split(data,strat_cat):

    train = data.loc[train_id];test = data.loc[test_id]

y_train = train['Survived']
x_train = train.drop('Survived',axis = 1)

from sklearn.impute import SimpleImputer


num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy = 'most_frequent')

y_test = test['Survived']
x_test = test.drop('Survived',axis = 1)

num_cols = x_train.select_dtypes(include = [np.number]).columns.tolist()
cat_cols = x_train.select_dtypes(include = ['object']).columns.tolist()
num_imputer.fit(x_train[num_cols])
cat_imputer.fit(x_train[cat_cols])
x_train[num_cols] = num_imputer.transform(x_train[num_cols])
x_test[num_cols] = num_imputer.transform(x_test[num_cols])
x_test[cat_cols] = cat_imputer.transform(x_test[cat_cols])
x_train[cat_cols] = cat_imputer.transform(x_train[cat_cols])



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train[num_cols])
x_train[num_cols] = scaler.transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])


from sklearn.preprocessing import OneHotEncoder

coder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')

x_train_ohe = coder.fit_transform(x_train[cat_cols])
x_test_ohe = coder.transform(x_test[cat_cols])

ohe_col = coder.get_feature_names_out(cat_cols)

x_train_ohe = pd.DataFrame(
    x_train_ohe,
    index = x_train.index,
    columns =ohe_col
)
x_test_ohe = pd.DataFrame(
    x_test_ohe,
    index = x_test.index,
    columns=ohe_col
)

x_train = x_train.drop(cat_cols,axis = 1)
x_test = x_test.drop(cat_cols,axis = 1)
x_train = pd.concat([x_train,x_train_ohe],axis= 1)
x_test = pd.concat([x_test,x_test_ohe],axis=1)

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
model = LogisticRegression()

model.fit(x_train,y_train)

y_out = model.predict(x_test)
y_out = np.array(y_out)
y_test = np.array(y_test)
print(len(y_out))
print(sum((y_test-y_out)**2))