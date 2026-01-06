"""
PROJENİN AMACI

Bu çalışmanın amacı, üretim hattından çıkan cips ürünlerinin görüntülerini kullanarak:
Sağlam Ürün ve Hatalı Ürün sınıflarını derin öğrenme yöntemleri ile otomatik olarak tespit eden bir kalite kontrol sistemi geliştirmektir.

Sistem, insan gözünden bağımsız olarak, yüksek doğrulukla hatalı ürünleri tespit ederek üretim hatlarında:

- İş gücü maliyetini azaltmayı
- Hata oranını düşürmeyi
- Ürün standartizasyonunu sağlamayı
amaçlamaktadır.

 KULLANILAN TEKNOLOJİLER

- Python	Projenin ana programlama dili
- TensorFlow / Keras	Derin öğrenme modelinin kurulması
- MobileNetV2	Önceden eğitilmiş CNN tabanlı özellik çıkarıcı
- Transfer Learning	Küçük veriyle yüksek doğruluk sağlamak
- ImageDataGenerator	Veri artırma (data augmentation)
- GlobalAveragePooling	Overfitting önleme
- Adam Optimizer	Hızlı ve stabil öğrenme
- EarlyStopping / ReduceLR	Aşırı öğrenmeyi önleme
- Confusion Matrix & F1 Score	Gerçek performans ölçümü

 YAPILAN ADIMLAR

1️- Görüntüler 96×96 boyutuna ölçeklendirildi.
2️- MobileNetV2’nin preprocess_input fonksiyonu ile normalize edildi.
3️- Eğitim seti için döndürme, kaydırma ve zoom işlemleriyle veri artırma uygulandı.
4️- Önceden eğitilmiş MobileNetV2 ağı, özellik çıkarıcı olarak kullanıldı.
5️- Son katmanlar eklenerek ikili sınıflandırma modeli oluşturuldu.
6️- Model Adam optimizasyon algoritması ile eğitildi.
7️- Eğitim süreci EarlyStopping ve ReduceLROnPlateau ile kontrol edildi.
8️-  Model; Accuracy, Precision, Recall ve F1-score metrikleri ile değerlendirildi.
9️- Confusion Matrix ile sınıflandırma başarımı görsel olarak analiz edildi.

🎯 ÇIKTI

Model;
    - %99+ doğruluk
    - Yüksek Recall (hatalı ürünleri kaçırmıyor)
    - Endüstri standartlarında güvenilirlik değerlerine ulaşarak otomatik kalite kontrol sistemi olarak kullanılabilir seviyeye gelmiştir.
"""
# %% GEREKLİ KÜTÜPHANELER
# Uyarıları kapat
import warnings
warnings.filterwarnings('ignore')

# Dosya işlemleri ve sayısal işlemler
import os, random
import numpy as np
import matplotlib.pyplot as plt

# Veri bölme ve performans metrikleri
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Derin öğrenme kütüphaneleri
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
# %% VERİ YOLU
MAIN_DIR = "data"
train_dir = os.path.join(MAIN_DIR, "Train")
test_dir  = os.path.join(MAIN_DIR, "Test")

# %% VERİYİ YÜKLEME VE GÖRÜNTÜLEME 

# Görüntüler 96x96 boyutuna ölçeklenir
train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(96,96), batch_size=32)
test_ds  = tf.keras.utils.image_dataset_from_directory(test_dir,  image_size=(96,96), batch_size=32)

# cips görüntülerine bakalım
plt.figure(figsize=(10,10))
plt.title("Eğitim görüntüleri")
plt.axis("off")

for i in range(9):
    img_path = random.choice(train_ds.file_paths)
    img = tf.keras.preprocessing.image.load_img(img_path)
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.axis("off")
plt.show()

# TensorFlow dataset --> NumPy dönüşümü
def ds_to_numpy(ds):
    X, y = [], []
    for img, lab in ds:
        X.append(img.numpy())
        y.append(lab.numpy())
    return np.concatenate(X), np.concatenate(y)

X_train, y_train = ds_to_numpy(train_ds)
X_test,  y_test  = ds_to_numpy(test_ds)

# modele girecek görüntü
plt.figure(figsize=(10,10))
plt.title("Model Girdisi")
plt.axis("off")

for i in range(9):
    idx = random.randint(0, len(X_train)-1)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(X_train[idx].copy())
    plt.subplot(3,3,i+1)
    plt.imshow((img + 1) / 2)
    plt.axis("off")

plt.show()

# %% EĞİTİM / VALIDATION AYIRMA
# Stratify: sınıf oranlarını korur
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)
# %% DATA AUGMENTATION & NORMALİZASYON
train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)
# %% TRANSFER LEARNING MODELİ
# Önceden eğitilmiş MobileNetV2 backbone
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96,96,3))
base_model.trainable = False  # İlk aşamada dondurulur

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),   # Özellik özetleme
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # Binary classification
])
# %% MODEL DERLEME
model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# %% MODEL EĞİTİMİ
history = model.fit(
    train_gen.flow(X_train, y_train, 32),
    validation_data=val_gen.flow(X_valid, y_valid, 32),
    epochs=10,
    callbacks=[
        EarlyStopping(monitor = "val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=5)
    ]
)
# %% TEST DEĞERLENDİRME
test_loss, test_acc = model.evaluate(val_gen.flow(X_test, y_test, 32))
print("Test Accuracy:", test_acc)
# Test verisini preprocess ederek tahmin al
X_test_pp = tf.keras.applications.mobilenet_v2.preprocess_input(X_test)

y_pred_prob = model.predict(X_test_pp)
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print(classification_report(
    y_test, y_pred,
    target_names=["Sağlam Ürün", "Hatalı Ürün"]
))

# %% EĞİTİM GRAFİKLERİ
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend(); plt.title("Accuracy")
plt.show()

# %% MODEL KAYDETME
model.save("cips_transfer_model.h5")