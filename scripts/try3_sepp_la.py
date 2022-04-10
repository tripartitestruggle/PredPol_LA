# Allow us to load `open_cp` without installing
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join("..","..")))
import csv
import open_cp.data
import geopandas as gpd
import pyproj
import dateutil.parser
import os
import open_cp.scripted as scripted
import datetime
import matplotlib.pyplot as plt
import open_cp.seppexp as sepp
import descartes
import open_cp.evaluation as evaluation
import numpy as np
import matplotlib.colors as colors

grid_size = 2000
block_size = 50

def row_to_datetime(row):
    datetime_string = row[1]
    return dateutil.parser.parse(datetime_string)
    
proj = pyproj.Proj({"init" : "epsg:3311"})

def row_to_coords(row):
    x = float(row[4])
    y = float(row[3])
    return proj(x, y)

def row_to_coords_noisy(row):
    x = float(row[5])
    y = float(row[6])
    return (x, y)

def load_points():
    with open("csv_arrest_10_11_masked_noisy.csv") as file:
        reader = csv.reader(file)
        header = next(reader)
        # Assume the header is correct
        times = []
        xcoords = []
        ycoords = []
        for row in reader:
            times.append(row_to_datetime(row))
            x, y = row_to_coords(row)
            #x, y = row_to_coords_noisy(row)
            xcoords.append(x)
            ycoords.append(y)
      
    xcoords += block_size*(0.5-np.random.rand(len(xcoords),))
    xcoords += block_size*(0.5-np.random.rand(len(ycoords),))
    # Maybe `times` is not sorted.
    times, xcoords, ycoords = open_cp.data.order_by_time(times, xcoords, ycoords)
      
    return open_cp.data.TimedPoints.from_coords(times, xcoords, ycoords)

def load_geometry():
    #frame = gpd.read_file("LA_shape")
    frame = gpd.read_file("LACityBoundary")
    frame = frame.to_crs({"init": "epsg:3311"})
    #frame.crs = {"init": "epsg:4326"}
    #frame = frame.to_crs({"init": "epsg:2790"})
    return frame.geometry[0]

def _add_outline(geo, ax):
    # to add the shapefile to the plot
    p = descartes.PolygonPatch(geo, fc="none", ec="black")
    ax.add_patch(p)
    ax.set_aspect(1)
    
def time_range(first, stop_before, duration):
    st = first
    en = st + duration
    times_array = [st]    
    while en <= stop_before:
        en = st + duration
        np.append(times_array,en)
    return times_array
    
# Some utility methods for plotting etc.    
cdict = {'red':   [(0.0,  1.0, 1.0),
                   (1.0,  1.0, 1.0)],
         'green': [(0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)],
         'blue':  [(0.0,  0.2, 0.2),
                   (1.0,  0.2, 0.2)]}
yellow_to_red = colors.LinearSegmentedColormap("yellow_to_red", cdict)

geo = load_geometry()
data = load_points()

#print("x_min=",geo.bounds[0]," x_max=",geo.bounds[2])
#print("y_min=",geo.bounds[1]," y_max=",geo.bounds[3])
#print("x_range=",geo.bounds[2]-geo.bounds[0])
#print("y_range=",geo.bounds[3]-geo.bounds[1])

x_max = 170000
y_max = -407000
#region = open_cp.RectangularRegion(xmin=0, xmax=500, ymin=0, ymax=500)
#region = open_cp.RectangularRegion(xmin=geo.bounds[0], xmax=geo.bounds[2], ymin=geo.bounds[1], ymax=geo.bounds[3])
region = open_cp.RectangularRegion(xmin=np.min(data.xcoords), xmax=x_max, ymin=np.min(data.ycoords), ymax=y_max)
trainer = sepp.SEPPTrainer(region, grid_size=grid_size)
timed_points = data
trainer.data = timed_points
predictor = trainer.train(iterations=200,cutoff_time=datetime.datetime(2011,12,31),use_corrected=True)
print("Predicted omega={}, theta={}".format(predictor.omega, predictor.theta))

#### make prediction
predictor.data = timed_points
#dates = time_range(datetime.datetime(2016,1,1),datetime.datetime(2016,12,30), datetime.timedelta(days=1))
#dates = time_range(datetime.datetime(2016,12,28),datetime.datetime(2016,12,30), datetime.timedelta(days=1))
start_date = datetime.datetime(2012,1,2)
numdays = 364
dates = [start_date + datetime.timedelta(days=x) for x in range(numdays)]
predictions = [predictor.predict(date) for date in dates]
background_prediction = predictor.background_prediction()

# count the number of times each grid cell shows up in the top x% riskiest predictions of the day
intensity_array_for_cell = []
cell_address = [3,14]
fraction = 0.006
top_slice_count = np.zeros((predictions[0].yextent,predictions[0].xextent))
print("hi")
i = 0
total_predictions = len(predictions)
for pred in predictions :
    covered = evaluation.top_slice(pred.intensity_matrix, fraction)
    covered_int = covered.astype(int)
    #print("############################\n###intensity_matrix")
    #print(pred.intensity_matrix)
    #print("###top_slice")
    #print(covered_int)
    top_slice_count = np.add(covered_int,top_slice_count)
    #also save intensity of one cell for plotting
    intensity_value = pred.grid_risk(*cell_address)
    np.append(intensity_array_for_cell,intensity_value)
    print("### %s out of %s done. Prediction done for date : %s" %(i+1,total_predictions,dates[i]))
    i += 1
    
print("### count of days")
print(top_slice_count)
#counting has finished

#now plot count for all grid cells
fig, ax = plt.subplots(ncols=1, figsize=(100,100))
ax.set(xlim=[region.xmin-1000, region.xmax+1000 ], ylim=[region.ymin-1000, region.ymax+1000])
#ax.set(xlim=[region.xmin-200, region.xmax+200], ylim=[region.ymin-200, region.ymax+200])
#m = ax.pcolormesh(*pred.mesh_data(), top_slice_count, cmap=yellow_to_red, edgecolor="black", linewidth=1)
m = ax.pcolormesh(*pred.mesh_data(), top_slice_count, cmap="Reds", edgecolor="black", linewidth=0)
#add the region shapefile
_add_outline(geo,ax)
# Tedious graph layout...
cax = fig.add_axes([0.9, 0.2, 0.01, 0.7])
cbar = fig.colorbar(m, orientation="vertical", cax=cax)
cbar.set_label("Total crime count in region")
ax.set_title("Count in each grid cell")
fig.savefig('temp.png', transparent=True)
plt.show()
None

