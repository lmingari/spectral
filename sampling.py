import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

### Parameters ###
NENS = 60
##################

#Read data
df1 = pd.read_csv('sampling_parameters_1.csv', index_col=0)
df2 = pd.read_csv('sampling_parameters_2.csv', index_col=0)
df1.index = df1.index.map(lambda i: str(i)+'a')
df2.index = df2.index.map(lambda i: str(i)+'b')
df  = pd.concat([df1,df2])

#Compute volume in the parameter space
df['vol'] = 8*df['stk_per']*df['dip_per']*df['rake_per']
df.sort_values('vol',inplace=True)

total_vol = df['vol'].sum()
print("Total volume: ",total_vol)

#Compute sub-ensemble sizes and sample
sampler = qmc.LatinHypercube(d=3)
df_list = []
NC=[]
for index, row in df.iterrows():
    nc_tmp = max(1,round(NENS*row['vol']/total_vol))
    NENS -= nc_tmp
    total_vol -= row['vol']
    NC.append(nc_tmp)
    #
    sample = sampler.random(n=nc_tmp)
    x0 = row[['stk_ref','dip_ref','rake_ref']].to_numpy()
    dx = row[['stk_per','dip_per','rake_per']].to_numpy()
    df_tmp = pd.DataFrame(qmc.scale(sample, x0-dx, x0+dx),columns=['stk','dip','rake'])
    df_tmp['cluster'] = index
    df_list.append(df_tmp)
df['size'] = NC
df_sample = pd.concat(df_list,ignore_index=True)
df_sample['stk'] =df_sample['stk']%360

#Plot parameter subspaces
fig, ax = plt.subplots()

xcol = 'rake'
ycol = 'stk'
xmin,xmax,xstep = -180,180,45
ymin,ymax,ystep = 0,360,45
labels = {'stk': "Strike [deg]",'dip':"Dip [deg]",'rake': "Rake [deg]"}
ax.set(title  = "Sampling of parameter space",
       xlabel = labels[xcol],
       ylabel = labels[ycol],
       xticks = np.arange(xmin,xmax+xstep,xstep),
       yticks = np.arange(ymin,ymax+ystep,ystep),
       xlim   = (xmin, xmax),
       ylim   = (ymin, ymax),
       )
for index, row in df.iterrows():
    x0 = row[xcol+'_ref']
    dx = row[xcol+'_per']
    y0 = row[ycol+'_ref']%360
    dy = row[ycol+'_per']
    x  = np.array([x0-dx,x0+dx])
    y1 = y0-dy
    y2 = y0+dy
    ax.fill_between(x,y1,y2,label=index,alpha=0.5)
    ax.text(x.mean(),y2,
            r"$n_c={:d}$".format(row['size'].astype(int)),
            fontsize = 6,
            ha='center',
            va='bottom')

ax.scatter(df_sample['rake'],df_sample['stk'],
           marker = '.',
           s      = 6,
           color  = 'k', 
           alpha  = 0.5
           )

ax.legend(loc=4,title="Cluster ID")
fig.savefig('areas.png',dpi=200)
df_sample.to_csv("ensemble_samples.csv",index=False)
print(df)
print(df_sample.agg(['min','max']))
