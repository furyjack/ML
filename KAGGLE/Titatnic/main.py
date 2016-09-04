import pandas as pd
import numpy as np
import sklearn.cross_validation as skc
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
df =pd.read_csv('train (1).csv',header=0)

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df['AgeFill'] = df['Age']
median_ages=np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin','Embarked','Age'], axis=1)
df=df.dropna()



train_data=df.values
label=train_data[:,1]

df=df.drop(['Survived'],axis=1)
train_data=df.values

X_train,X_test,Y_train,Y_test=skc.train_test_split(train_data,label)
clf=RFC(n_estimators=60)
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))


df1=pd.read_csv('test.csv',header=0)
df1['Gender'] = df1['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#
df1['AgeFill'] = df1['Age']
median_ages=np.zeros((2,3))
for i in range(0, 2):
     for j in range(0, 3):
         df1.loc[ (df1.Age.isnull()) & (df1.Gender == i) & (df1.Pclass == j+1),\
                 'AgeFill'] = median_ages[i,j]

df1['AgeIsNull'] = pd.isnull(df1.Age).astype(int)
df1['FamilySize'] = df1['SibSp'] + df1['Parch']
df1['Age*Class'] = df1.AgeFill * df1.Pclass
df1 = df1.drop(['Name', 'Sex', 'Ticket', 'Cabin','Embarked','Age'], axis=1)
#
#
#
#
test_data=df1.values
ans=clf.predict(test_data)
for x in ans:
    print(x)









