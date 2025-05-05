import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

x = data[['Annual Income (k$)','Spending Score (1-100)']]

print(x.head())

wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss,marker='o')
plt.title("Elbow Yontemi")
plt.xlabel("Kume Sayisi")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(x)

data['Cluster'] = clusters

plt.figure(figsize=(8,6))
colors = ['red','green','blue','cyan','purple']

for i in range(5):
    plt.scatter(x[clusters == i]['Annual Income (k$)'],
                x[clusters == i]['Spending Score (1-100)'],
                c=colors[i], label=f"Kume {i}")

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Merkezler', marker='X')

plt.xlabel("Yıllık Gelir")
plt.ylabel("Harcama Skoru")
plt.title("Musteri Segmentasyonu")
plt.legend()
plt.grid(True)
plt.show()