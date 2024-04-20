# Gerekli kütüphaneleri yükleme
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Örnek veri seti oluşturma
data = {
    'Miktar': [100.0, 200.0, 50.0, 300.0, 150.0, 500.0, 1000.0, 80.0, 120.0, 600.0],
    'Saat': [8, 10, 15, 12, 9, 16, 11, 14, 17, 13],
    'Gün': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    'Sahtekarlık': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # 1: Sahtekarlık, 0: Sahtekarlık yok
}

df = pd.DataFrame(data)

# Bağımsız ve bağımlı değişkenleri tanımlama
X = df.drop("Sahtekarlık", axis=1)
y = df["Sahtekarlık"]

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri önişleme: Standart ölçekleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modeli eğitme
model = LogisticRegression()
model.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))