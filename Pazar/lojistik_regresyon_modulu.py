import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
print("Lojistik Regresyon İçin Veri Seti Oluşturuluyor...")

Data_classification = {
    "Age": [22,25,47,52,46,56,55,60,62,61],
    "Income": [25000,32000,47000,52000,46000,58000,60000,62000,64000,63000],
    "Purchased": [0,0,1,1,1,1,1,1,1,1]
}

df_classsication = pd.DataFrame(Data_classification)

print("Lojistik Regresyon İçin Veri Eğitim ve test Setine Ayrılıyor...")

x_cls = df_classsication[["Age","Income"]]
y_cls = df_classsication["Purchased"]

x_train_cls, x_test_cls,y_train_cls,y_test_cls = train_test_split(x_cls,y_cls,test_size=0.2,random_state=42)

print(f"Egitim Seti Boyutu: {x_train_cls.shape}")
print(f"Test Seti Boyutu: {x_test_cls.shape}")

print("Lojistik Regresyon Modeli eğitiliyor...")

model_logistic = LogisticRegression()

model_logistic.fit(x_train_cls,y_train_cls)

print("Lojistik Regresyon Modeli Eğitildi!")
print("Katsayılar:")
print(model_logistic.coef_)
print(f"Intercept: {model_logistic.intercept_}")

print("Lojistik Regresyon Modeli Test Verisi ile Tahmin Yapıyor...")

y_pred_logistic = model_logistic.predict(x_test_cls)

for gerçek, tahmin in zip(y_test_cls, y_pred_logistic):
    print(f"Gerçek: {gerçek} -> Tahmin: {tahmin}")

print("Lojistik Regresyon Modeli Performansı Ölçülüyor...")

accuracy_logistic = accuracy_score(y_test_cls,y_pred_logistic)
print(f"Doğruluk Skoru: {accuracy_logistic:.4f}")

print("KNN Modeli Eğitiliyor...")

model_knn = KNeighborsClassifier(n_neighbors=5)

model_knn.fit(x_train_cls,y_train_cls)

print("KNN Modeli Test Verisi ile Tahmin Yapıyor...")

y_pred_knn = model_knn.predict(x_test_cls)

for gerçek, tahmin in zip(y_test_cls, y_pred_knn):
    print(f"Gerçek: {gerçek} -> Tahmin: {tahmin}")
