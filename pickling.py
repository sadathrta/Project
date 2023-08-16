import numpy as np
import pandas as pd
import pickle



df = pd.read_csv("HR dataset Project.csv")
## Replacing missing values in educatuion with mode of age category
# Step 1: Create a function to fill null values in 'education' based on 'age'
def fill_education_by_age(row):
    age = row['age']

    # Get the mode of education for individuals with the same age
    mode_education = df[(df['age'] == age)]['education'].mode().values

    # If there's a mode value, use it; otherwise, keep the original value
    return mode_education[0] if len(mode_education) > 0 else row['education']

#Apply the function to fill null values in the 'education' column
df['education'] = df.apply(lambda row: fill_education_by_age(row) if pd.isnull(row['education']) else row['education'], axis=1)

## Replacing Null values in previous_year_rating  with 0 as lenght of service is 1 year for employees for them i.e they are new employees
df['previous_year_rating'].fillna(0,inplace=True)

##ordinal encoding education column
from sklearn.preprocessing import OrdinalEncoder


Education = ['Below Secondary',"Bachelor's","Master's & above"]

enc = OrdinalEncoder(categories=[Education])

df[['education']] = enc.fit_transform(df[['education']])





##onehotencoding on gender recruitmentchannel and Department
from numpy import int32
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False,dtype=int32,drop = 'first')
cols = df[['gender','recruitment_channel','department']]
onehot_encoded = ohe.fit_transform(cols)
# Get the feature names
feature_names = ohe.get_feature_names_out(['gender', 'recruitment_channel','department'])
# Create a new DataFrame with the encoded data and feature names
df_encoded = pd.DataFrame(onehot_encoded, columns=feature_names)
# Concatenate the original DataFrame and the encoded DataFrame
df = pd.concat([df, df_encoded], axis=1)
# Droping columns 'gender','recruitment_channel','department'
df.drop(['gender','recruitment_channel','department'],axis=1,inplace=True)

## Encoding region with the region number
### extract number from region name

df['region'] = df['region'].str.extract('(\d+)')

## checkinf data type of column
df['region'].dtype
df['region'] = df['region'].astype(int)




##Feature Scaling
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
cols = df[['avg_training_score']]
scaled = scalar.fit_transform(cols)
## Droping unscaled columns
df.drop(['avg_training_score'],axis =1,inplace = True)
# Create a new DataFrame with the encoded data and feature names
scaled_col = pd.DataFrame(scaled, columns=['avg_training_score'])
# Concatenate the original DataFrame and the encoded DataFrame
df = pd.concat([df, scaled_col], axis=1)

#Feature Reduction
df.drop('employee_id',axis = 1 ,inplace = True)

#Training and test Splt

x = df.drop(['is_promoted'],axis = 1) ## independent variable
y = df['is_promoted']

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.25,random_state= 0)


#Balancing
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
x_train_sm,y_train_sm = sm.fit_resample(x_train,y_train)
from xgboost import XGBClassifier

# Create an XGBoost classifier with specified parameters

model = XGBClassifier(booster='gbtree',eval_metric='auc',learning_rate=0.01,max_depth=220,min_child_weight=1,n_estimators=581,objective='binary:logistic',scale_pos_weight=1,tree_method='hist')

# Fit the model to your training data
model.fit(x_train_sm.values, y_train_sm.values)
pickle.dump(model,open('xgmodel.pkl','wb') )
pickle.dump(scalar,open('scale.pkl','wb') )
pickle.dump(ohe,open('ohe.pkl','wb') )

## Creating Pickel import pickle






