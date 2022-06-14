import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Inputs ###
fname   = 'spectral_clustering.csv'
nplane  = 1
figname = 'polar_{n}'.format(n=nplane) 
csvname = 'sampling_parameters_{n}.csv'.format(n=nplane)
##############

class MyComplex:
    """A simple class for complex numbers"""
    def __init__(self,angles):
        #Angles in degree
        if isinstance(angles,list):
            self.degree = np.array(angles)
        else:
            self.degree = angles
        self.z  = np.exp(1j*self.degree*np.pi/180.)

    def _cmean(self):
        return self.z.mean()

    def mean_phase(self):
        return np.angle(self._cmean(), deg=True)

    def mean_arc(self):
        minAngle = (self.degree-self.mean_phase())%360
        minAngle = np.where(minAngle>180, 360-minAngle,minAngle)
        return minAngle.mean()

df = pd.read_csv(fname)

df_grouped = df.groupby("cluster")
groups  = df_grouped.groups.keys()
columns = ['stk','dip','rake']

fig, axs = plt.subplots(ncols=len(columns),nrows=len(groups),figsize=(14,20),subplot_kw={'projection': 'polar'})

###References
r1={'stk_1':270, 'dip_1':37,  'rake_1':-95 }
r2={'stk_1':96.1,'dip_1':53.1,'rake_1':-86.2}

i=0
thetas = []
for group_name, df_group in df_grouped:
    theta = {}
    j=0
    for col in columns:
        colDF  = '{col}_{n}'.format(col=col,n=nplane)
        colRef = col+'_ref'
        colPer = col+'_per'
        #
        data = [ x/180.0*np.pi for x in df_group[colDF] ]
        axs[i,j].scatter(data,np.ones_like(data), marker='o',facecolors='r',edgecolors='k')
        #
        x = MyComplex(df_group[colDF])
        theta[colRef] = x.mean_phase()
        theta[colPer] = x.mean_arc()
        #
        axs[i,j].plot([0,np.radians(theta[colRef])],[0,1],'r-')
        axs[i,j].bar(np.radians(theta[colRef]),1,
                     width=2*np.radians(theta[colPer]),
                     bottom=0.0,
                     alpha=0.25)
        axs[i,j].set_title("Cluster ID: {} for {}".format(group_name,colDF))
        #
#        axs[i,j].plot([0,np.radians(r1[col])],[0,1],'g-')
#        axs[i,j].plot([0,np.radians(r2[col])],[0,1],'g-')
        #
        j+=1
    thetas.append(theta)
    i += 1

fig.tight_layout()
fig.savefig(figname)

df = pd.DataFrame(thetas,index=groups)
df.index.name="cluster"
df.to_csv(csvname)
print(df)

