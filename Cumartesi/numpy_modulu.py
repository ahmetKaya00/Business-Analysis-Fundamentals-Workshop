import numpy as np
import matplotlib.pyplot as plt

dizi = np.array([10,654,30,123,50])

print("Dizi:", dizi)
print("Toplam: ", np.sum(dizi))
print("Ortalama: ", np.mean(dizi))
print("Maksimum Değer: ", np.max(dizi))
print("Minimum Değer: ", np.min(dizi))

np.random.seed(42)

notlar = np.random.randint(0,101,1000)

print("Öğrencinin Notları (ilk 10): ", notlar[:10])

print("Ortalama Notlar:", np.mean(notlar))
print("En Yüksek Not", np.max(notlar))
print("En Düşük Not", np.min(notlar))

gecenler = notlar[notlar >= 50]
kalanlar = notlar[notlar < 50]

print("Gecen Öğrenci Sayisi: ",len(gecenler))
print("Kalan Öğrenci Sayisi: ",len(kalanlar))

plt.figure(figsize=(10,5))
plt.hist(notlar, bins=10, edgecolor="black", alpha=0.7)
plt.xlabel("Not Ağırlıkları")
plt.ylabel("Öğrenci Sayısı")
plt.title("Öğrenci Not Dağılımı")
plt.grid(True)
plt.show()