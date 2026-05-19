# Week 8 — Unsupervised Learning

Deskripsi singkat:
- Topik: Clustering dan pengelompokan data (K-Means contoh).
- Dataset contoh: `berat_tinggi.csv`.

Persyaratan:
- Python 3.8+
- pip install pandas scikit-learn matplotlib

Contoh program singkat (kmeans_example.py):

```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('berat_tinggi.csv')
X = df[['Tinggi','Berat']]

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)
df['cluster'] = labels
print(df.groupby('cluster').mean())

plt.scatter(X['Tinggi'], X['Berat'], c=labels)
plt.xlabel('Tinggi')
plt.ylabel('Berat')
plt.title('KMeans Clustering')
plt.show()
```

Jalankan:
```
pip install pandas scikit-learn matplotlib
python kmeans_example.py
```
