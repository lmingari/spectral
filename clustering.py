import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering,DBSCAN
from sklearn.metrics import pairwise_distances
from metrics import kagan_angle,minArc

fname = "filtered_data.csv"
df = pd.read_csv(fname)

#Define X (n_samples_X, n_features)
X = df[['rr','tt','pp','rt','rp','tp']].to_numpy()

#Define affinity matrix A (n_samples, n_samples)
A = pairwise_distances(X, metric=kagan_angle)
A = np.where(A>45.,0.0,np.cos(np.radians(A)))

#Clustering
model = SpectralClustering(n_clusters=4, affinity='precomputed')
labels = model.fit_predict(A)

#Save data
df['cluster'] = labels
df.sort_values('cluster',inplace=True)
df.set_index('cluster', inplace=True)
df.to_csv('spectral_clustering_original.csv')

ref = df.groupby("cluster").first()[['stk_1','dip_1','rake_1']]

for index, row in df.iterrows():
    x0=ref.loc[index]
    x1=row[['stk_1','dip_1','rake_1']].to_numpy()
    x2=row[['stk_2','dip_2','rake_2']].to_numpy()
    arc1 = minArc(x1,x0)
    arc2 = minArc(x2,x0)
    if arc2<arc1:
        #Swap nodal planes
        row[['stk_1','dip_1','rake_1']]=x2
        row[['stk_2','dip_2','rake_2']]=x1

df.to_csv('spectral_clustering.csv')

print(df)

