# trk_to_annotation
This repository provides utilities for converting .trk tractography files into Neuroglancer precomputed annotation format, and serving them locally with HTTP requests.

## Install
```
python -m build
pip install .
```

## Usage
```
python -m trk_to_annotation <trk_file> \
    --annotation_output_dir ./precomputed_annotations \
    --segmentation_output_dir ./precomputed_segmentations \
    --grid_densities 1 2 4 8 16
```

Arguments
| Argument | Required | Default | Description |
| :---:   | :---: | :---: | :---: |
| trk_file | ✅ | — | Path to the input .trk file |
| --annotation_output_dir | ❌ | ./precomputed_annotations | Output directory for precomputed annotations |
| --segmentation_output_dir | ❌ | ./precomputed_annotations/precomputed_segmentations | Output directory for precomputed segmentations |
| --grid_densities | ❌ | [1, 2, 4, 8, 16] |Grid densities (powers of two, ascending order) |

## Running server for neuroglancer

Run the following command in the directory with your outputted folders
```
python -m trk_to_annotation.http_server
```

You should now be able to access the annotation and segmentation layers on neuroglancer via http://localhost:8000/