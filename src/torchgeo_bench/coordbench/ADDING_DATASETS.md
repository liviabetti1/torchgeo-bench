# Adding Spatiotemporal Datasets to CoordBench

## 1. Adding your data to huggingface.
For ease of access across different machines, we store our data in the [Temporal Coordbench](https://huggingface.co/Temporal-Coordbench) 
huggingface repository.  If you make a new dataset, just add a new dataset repository and upload the related files.  For consistency,
files should be in a `.parquet` format.  You should have the following columns:
- lat
- lon
- posix timestamp (should be in seconds from Jan 1, 1970, and in utc)
- end posix timestamp (if your temporal resolution is coarser than a day)
- task column(s)
- optional: spatial aggregation key (if multiple lat/lon points belong to the same label)

## 2. Building your dataloader.
All our dataloaders live in the [datasets.py](datasets.py) file.  You can see dataloaders for existing datasets, 
such as VIIRS night lights, ERA5, etc.Those should serve as a template. You should use `load_config_temporal` to load your
dataset in as a pandas dataframe.  From there, turn it into a `CoordBenchmark` object, specifying lat, lon, labels,
timestamps, temporal resolution, etc.  Again, refer to existing spatiotemporal dataloaders as a guide.  

You can optionally add a `subsample_locations` default variable (set to False) that returns a toy version of the dataset.  
Given that expanding datasets on both a spatial and temporal axis gets big, fast, this is highly recommended for 
development work.

If you are working with a polygon-style dataset (i.e. for admin-level data such as a census), refer to the 
`load_weekly_usa_electric_usage` and `load_usa_census_data` dataloaders.  We default to using the centroid of the polygon
currently.  As such, your dataset should contain the centroid for each polygon, in addition to any polygon details.  If 
you want to add an optional path for spatial aggregation, you should include the `from_polygon` default variable (set to
False) in your dataloader.  Then pull lat/lon from the polygon, either from pre-sampled points or by sampling in-flight.
You'll also need to set a `spatial_aggregation_key` so that our downstream code knows which rows to group together.

## 3. Adding your dataloader to configs
The most straighforward component of this is adding your dataset name and dataloader as an entry to `FAMILY_LOADERS`.  This
is what wires up the dataset's name in the command line entry to actually calling your dataloader.

If you have any default variables (such as `subsample_locations` or `from_polygon`), add that same dataset name to the list
of datasets which have that variable (`CAN_SUBSAMPLE` and `POLYGON_BENCH` respectively).  This is necessary for allowing the
command line configs of `--subsample` or `--from-polygon` to be passed in as arguments to the dataloaders.
