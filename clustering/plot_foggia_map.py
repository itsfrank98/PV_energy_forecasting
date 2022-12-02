import arff
import gmplot
import pandas as pd

d = {}
d['lat'] = []
d['lon'] = []

src = "../datasets/SS-DT_foggia_prod/train_2019.arff"
seen_keys = []
for row in arff.load(src):
    index = row[0]
    if index not in seen_keys:
        seen_keys.append(index)
        lat = row[1]
        lon = row[2]
        d['lat'].append(lat)
        d['lon'].append(lon)

df = pd.DataFrame(d)
center = (df.lat.min()+(df.lat.max()-df.lat.min())/2, df.lon.min()+(df.lon.max()-df.lon.min())/2)
gmap = gmplot.GoogleMapPlotter(center[0], center[1], 10.5)
gmap.scatter(list(df.lat), list(df.lon), '#FF0000', size = 40, marker = False )
gmap.draw("map.html")
