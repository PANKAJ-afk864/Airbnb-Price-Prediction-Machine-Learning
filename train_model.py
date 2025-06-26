import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("AB_NYC_2019.csv")
df = df.drop(['id', 'name', 'host_name', 'last_review'], axis=1)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df = df[df['price'] < 1000]
df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)

df_sample = df.sample(n=10000, random_state=42)
X = df_sample.drop(['price', 'neighbourhood'], axis=1)
y = df_sample['price']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

joblib.dump(model, 'airbnb_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
