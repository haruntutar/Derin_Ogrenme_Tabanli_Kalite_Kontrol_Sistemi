# Derin_Ogrenme_Tabanli_Kalite_Kontrol_Sistemi
Bu çalışmanın amacı, üretim hattından çıkan cips ürünlerinin görüntülerini kullanarak: Sağlam Ürün ve Hatalı Ürün sınıflarını derin öğrenme yöntemleri ile otomatik olarak tespit eden bir kalite kontrol sistemi geliştirmektir.
# 🥔 Cips Ürünleri İçin Otomatik Kalite Kontrol Sistemi

## 📌 Projenin Amacı
Bu çalışmanın amacı, üretim hattından çıkan cips ürünlerinin görüntülerini kullanarak:  
**Sağlam Ürün** ve **Hatalı Ürün** sınıflarını derin öğrenme yöntemleri ile otomatik olarak tespit eden bir kalite kontrol sistemi geliştirmektir.  

Sistem, insan gözünden bağımsız olarak yüksek doğrulukla hatalı ürünleri tespit ederek üretim hatlarında:  
- İş gücü maliyetini azaltmayı  
- Hata oranını düşürmeyi  
- Ürün standartizasyonunu sağlamayı  
amaçlamaktadır.  

---

## 🛠 Kullanılan Teknolojiler
- **Python** → Projenin ana programlama dili  
- **TensorFlow / Keras** → Derin öğrenme modelinin kurulması  
- **MobileNetV2** → Önceden eğitilmiş CNN tabanlı özellik çıkarıcı  
- **Transfer Learning** → Küçük veriyle yüksek doğruluk sağlamak  
- **ImageDataGenerator** → Veri artırma (data augmentation)  
- **GlobalAveragePooling** → Overfitting önleme  
- **Adam Optimizer** → Hızlı ve stabil öğrenme  
- **EarlyStopping / ReduceLR** → Aşırı öğrenmeyi önleme  
- **Confusion Matrix & F1 Score** → Gerçek performans ölçümü  

---

## 🔄 Yapılan Adımlar
1. Görüntüler **96×96** boyutuna ölçeklendirildi.  
2. MobileNetV2’nin **preprocess_input** fonksiyonu ile normalize edildi.  
3. Eğitim seti için döndürme, kaydırma ve zoom işlemleriyle **veri artırma** uygulandı.  
4. Önceden eğitilmiş **MobileNetV2** ağı, özellik çıkarıcı olarak kullanıldı.  
5. Son katmanlar eklenerek **ikili sınıflandırma modeli** oluşturuldu.  
6. Model **Adam optimizasyon algoritması** ile eğitildi.  
7. Eğitim süreci **EarlyStopping** ve **ReduceLROnPlateau** ile kontrol edildi.  
8. Model; **Accuracy, Precision, Recall ve F1-score** metrikleri ile değerlendirildi.  
9. **Confusion Matrix** ile sınıflandırma başarımı görsel olarak analiz edildi.  

---

## 🎯 Çıktı
Model:  
- ✅ %99+ doğruluk  
- ✅ Yüksek Recall (hatalı ürünleri kaçırmıyor)  
- ✅ Endüstri standartlarında güvenilirlik değerlerine ulaşarak **otomatik kalite kontrol sistemi** olarak kullanılabilir seviyeye gelmiştir.
-  
## 📊 Dataset
Veri seti:https://www.kaggle.com/datasets/concaption/pepsico-lab-potato-quality-control
---
