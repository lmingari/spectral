import numpy as np
import pandas as pd
import obspy
from haversine import haversine

cat=obspy.read_events("../DATA/Mediterranean_SPUD_QUAKEML_bundle_2021-07-30T09.51.41.xml")

NumEventsTotal = len(cat)

# Parameters
maxDistance = 90 #in km

eventData  = []
data = []
for e in cat:
    nodal_planes = e.focal_mechanisms[0]['nodal_planes']
    tensor = e.focal_mechanisms[0]['moment_tensor'].tensor
    #
    item = {'mag':   e.magnitudes[0]['mag'],
            'lat':   e.origins[0]['latitude'],
            'lon':   e.origins[0]['longitude'],
            'depth': e.origins[0]['depth'],
            'time':  e.origins[0]['time'] }
    eventData.append(item)
    #
    item = {'rr': tensor.m_rr,
            'tt': tensor.m_tt,
            'pp': tensor.m_pp,
            'rt': tensor.m_rt,
            'rp': tensor.m_rp,
            'tp': tensor.m_tp,
            'stk_1':  nodal_planes.nodal_plane_1.strike,
            'dip_1':  nodal_planes.nodal_plane_1.dip,
            'rake_1': nodal_planes.nodal_plane_1.rake,
            'stk_2':  nodal_planes.nodal_plane_2.strike,
            'dip_2':  nodal_planes.nodal_plane_2.dip,
            'rake_2': nodal_planes.nodal_plane_2.rake }
    data.append(item)



df = pd.DataFrame(data)
df_aux = pd.DataFrame(eventData)

#Compute distance and filter
epicentre = (37.918, 26.790)
df['dist'] = df_aux.apply(lambda x: haversine((x['lat'],x['lon']),epicentre), axis=1)
df = df[df['dist']<maxDistance]

#Save data
df.to_csv("filtered_data.csv", index=False)
