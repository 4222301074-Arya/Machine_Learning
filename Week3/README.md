# Week 3 — Feature Engineering

Deskripsi singkat:
- Topik: Feature engineering dan pra-pemrosesan data.
- Dataset contoh: `california_dataset.csv`, `company.csv`.

File penting:
- `Hands On_Week_3_Feature_Engineering.ipynb` — notebook latihan.

Persyaratan:
- Python 3.8+
- pip install pandas scikit-learn

Contoh program singkat (feature_engineering_example.py):

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Baca dataset (letakkan california_dataset.csv di folder Week3)
df = pd.read_csv('california_dataset.csv')

# Contoh pra-pemrosesan: isi nilai kosong, ubah kategori, split
df = df.dropna(subset=['median_house_value'])
df = df.fillna(df.median(numeric_only=True))

# Contoh one-hot untuk kolom kategori jika ada
if 'ocean_proximity' in df.columns:
	df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)
```

Jalankan:
```
pip install pandas scikit-learn
python feature_engineering_example.py
```
