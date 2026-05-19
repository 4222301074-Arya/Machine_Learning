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
