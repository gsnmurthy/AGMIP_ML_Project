import rioxarray
import geopandas
from shapely.geometry import mapping

geodf = geopandas.read_file("/home/samay/Desktop/IIT/HPC-project/shape_files/Lower_48_grid_fixed.shp")

xds = rioxarray.open_rasterio("../ggcmi/AgMerra/pr_agmerra_1980-2010.nc4")

print("[INFO]. Setting the Parameters.")
# xds = xds[['tasmin']].transpose('time', 'lat', 'lon')
# xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
xds.rio.write_crs("EPSG:4326", inplace=True)

print("[INFO]. Clipping")
clipped = xds.rio.clip(geodf.geometry.apply(mapping), geodf.crs, from_disk=True)

print("[INFO]. Saving the clipped file at ")

# DF = clipped.to_dataframe()
# DF.to_csv("./tasmin_clipped.csv")

del xds, clipped
