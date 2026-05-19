import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('california_dataset.csv')

df = df.dropna(subset=['median_house_value'])
df = df.fillna(df.median(numeric_only=True))

if 'ocean_proximity' in df.columns:
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)
