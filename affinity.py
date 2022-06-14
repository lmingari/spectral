import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from metrics import kagan_angle
import matplotlib.pyplot as plt

fname = "spectral_clustering.csv"

df = pd.read_csv(fname)

#Define X (n_samples_X, n_features)
X = df[['rr','tt','pp','rt','rp','tp']].to_numpy()

#Define affinity matrix A (n_samples, n_samples)
A = pairwise_distances(X, metric=kagan_angle)
A = np.where(A>45.,0.0,np.cos(np.radians(A)))

fig,ax=plt.subplots()
im = ax.imshow(A)

size_cum=0
for size in df.groupby('cluster').size():
    xmin,xmax = size_cum-0.5,size_cum+size-0.5
    ax.plot([xmin,xmax],[xmin,xmin],'r-')
    ax.plot([xmin,xmax],[xmax,xmax],'r-')
    ax.plot([xmin,xmin],[xmin,xmax],'r-')
    ax.plot([xmax,xmax],[xmin,xmax],'r-')
    size_cum += size

ticks = np.arange(0,20,2)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
cbar = fig.colorbar(im)
cbar.set_label("Affinity")
plt.savefig("affinity.png", dpi=200)
