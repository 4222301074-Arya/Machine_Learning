# Week 5 — K-NN Classification

Deskripsi singkat:
- Topik: K-Nearest Neighbors untuk masalah klasifikasi.
- Dataset contoh: `titanic.csv`.

Persyaratan:
- Python 3.8+
- pip install pandas scikit-learn

Contoh program singkat (knn_titanic_example.py):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male':0,'female':1})

features = ['Pclass','Sex','Age','Fare']
X = df[features].fillna(0)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
```

Jalankan:
```
pip install pandas scikit-learn
python knn_titanic_example.py
```
