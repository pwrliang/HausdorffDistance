# Introduced two new parameters
- "n_dims", new parameter, you have to explicity specify this parameter
- "input_type", new parameter, default is "wkt"

# Run Geospatial Datasets
```bash
./bin/hd_exec \
    -input1 /local/storage/liang/PPoPPAE/datasets/polygons/dtl_cnty.wkt \
    -input2 /local/storage/liang/PPoPPAE/datasets/polygons/dtl_cnty.wkt \
    -n_dims 2 \ 
    -repeat=1 \
    -limit 1000000 \
    -move_offset 1 \
    -variant rt \
    -execution gpu \
    -v=1
```

# Run MRI Datasets
```bash
./bin/hd_exec \
    -input1 /local/storage/shared/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii \
    -input2 /local/storage/shared/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_seg.nii \
    -input_type image \ 
    -n_dims 3 \ 
    -repeat 1 \
    -variant rt \
    -execution gpu \
    -v=1
```