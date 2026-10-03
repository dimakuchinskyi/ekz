import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

np.random.seed(42)
n_samples = 300

data = {
    'вік': np.random.randint(25, 75, n_samples),
    'тиск': np.random.normal(120, 15, n_samples),
    'холестерин': np.random.normal(200, 40, n_samples),
    'цукор_у_крові': np.random.normal(5.5, 1.5, n_samples),
    'ІМТ': np.random.normal(25, 4, n_samples)
}
df = pd.DataFrame(data)

probabilities = 1 / (1 + np.exp(-(-10 + 0.05*df['вік'] + 0.02*df['тиск'] + 0.01*df['холестерин'] + 0.5*df['цукор_у_крові'])))
df['діагноз'] = (probabilities > 0.5).astype(int)

for col in ['тиск', 'холестерин', 'ІМТ']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan


print("кількість пропусків до обробки:\n", df.isnull().sum())

df.fillna(df.median(), inplace=True)

print("\nкількість пропусків після обробки:\n", df.isnull().sum())

plt.figure(figsize=(6, 4))
sns.countplot(x='діагноз', data=df, palette='Set2')
plt.title('розподіл паієнтів (0 - здорові, 1 - хворі)')
plt.show()

X = df.drop('діагноз', axis=1)
y = df['діагноз']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nоцінка якості моделі")
print(f"Accuracy (точність загальна): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision (точність класифікації 1): {precision_score(y_test, y_pred):.2f}")
print(f"Recall (Ппвнота виявлення хворих): {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")

new_patients = pd.DataFrame({
    'вік': [30, 65, 45],
    'тиск': [110.0, 150.0, 130.0],
    'холестерин': [180.0, 260.0, 210.0],
    'цукор_у_крові': [4.5, 7.8, 5.2],
    'ІМТ': [22.0, 31.0, 26.0]
})

new_patients_scaled = scaler.transform(new_patients)
predictions = model.predict(new_patients_scaled)

print("\nрезультати для нових пацієнтів")
for i, pred in enumerate(predictions):
    status = "хворий" if pred == 1 else "здоровий"
    print(f"пацієнт {i+1}: {status}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.transform(X))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.8)
plt.title('візуалізація даних пацієнтів у 2D просторі (PCA)')
plt.xlabel('головна компонента 1 (PC1)')
plt.ylabel('головна компонента 2 (PC2)')
plt.legend(title='діагноз')
plt.show()