import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df=pd.read_csv('Crop_recommendation.csv')
X=df[['N','P','K','temperature','humidity','ph','rainfall']]
y=df['label']

model=RandomForestClassifier()
model.fit(X,y)

pickle.dump(model,open('crop_model.pkl','wb'))
