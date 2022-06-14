# Packages required
Packages installed for this work are as following:

| Package | `python` | `pytorch` | `torchvision` | `numpy` | `scipy` | `matplotlib` | `ffmpeg` | `opencv` | `shapely` | `pillow` |
| --------|----------|-----------|---------------|---------|---------|--------------|----------|----------|-----------|----------|
| Version | 3.8.12   | 1.7.1     | 0.8.2         | 1.19.2  | 1.4.1   | 3.5.0        | 4.2.2    | 3.4.10   | 1.7.1     | 8.4.0    |

To use Mask R-CNN, you need

| Package | `pycocotools` |
| --------| --------------|
| Version | 2.0.4         |

To use Flow Walk, you need

| Package | `spatial-correlation-sampler` |
| --------| ------------------------------|
| Version | 0.4.0                         |

# Generating synthetic GPI data
You can generate a set of synthetic GPI data for training dataset by running the shell script "run_generate_synth_gpi.sh". The synthetic blobs have their sizes, amplitudes, speeds at each moment, as well as trajactories which are randomly yet accordingly assigned to mimic the blobs in the real GPI data. The key fields used are as following:

| Field          | Description                                         | Default    |
| -------------- |-----------------------------------------------------| ----------:|
| `--image_size` | The size of the image (width X height)              | [256, 256] |
| `--n_frame`    | The number of frames in a file                      | 200        |
| `--n_data`     | The number of data files to generate                | 30         |
| `--val_prop`   | The proportion of total frames for validation data  | 0.05       |
| `--test`       | Use when you generate testing dataset               | False      |

The output files will be saved in "GPI-blob-tracking/data/synthetic_gpi". For testing dataset, the save directory is "GPI-blob-tracking/data/synthetic_gpi/testing".

# Training models with synthetic GPI data
You can train a model in "motion" folder by running "run_train_model.sh". The key fields used are as following:

| Field          | Description                                  |
| -------------- |----------------------------------------------|
| `--name`       | The name of the run. (model name)-synblobs   |
| `--num_steps`  | The number of epochs in training             |
| `--lr`         | Learning rate                                |
| `--gamma`      | Exponential weighting                        |
| `--wdecay`     | Weight decay                                 |

The field values used in this work are in "run_train_model.sh" for each model. The output model will be saved in "GPI-blob-tracking/models".

# Evaluating trained models with synthetic GPI data
You can get the evaluation scores of a trained model on synthetic GPI data by running "run_evaluate_model.sh". This will print out the scores on the validation dataset with the metric corresponding to each model.

# Processing real GPI data
You can pre-process your own GPI data. The brightness is upsampled and standardized by running "run_process_real_gpi.sh". The key fields used are as following:

| Field          | Description                                           | Default    |
| -------------- |-------------------------------------------------------| ----------:|
| `--filename`   | The directory to your raw GPI data file               | '../data/real_gpi/65472_0.35_raw.pickle' |
| `--image_size` | The size of the image (width X height) for upsampling | [256, 256] |

Your raw GPI data (.pickle file) must have the following variables:

| Variable          | Description                                  | Dimension |
| ----------------- |----------------------------------------------|-----------|
| `brt_arr` | The brightness video array measured by GPI   | (number of GPI views in R-axis) X (number of GPI views in z-axis) X (number of time points) (e.g. 12 X 10 X 1000)|
| `r_arr`  | The actual R-coordinates of the GPI views    | (number of GPI views in R-axis) X (number of GPI views in z-axis) (e.g. 12 X 10)|
| `z_arr`  | The actual z-coordinates of the GPI views    | (number of GPI views in R-axis) X (number of GPI views in z-axis) (e.g. 12 X 10)|
| `shear_contour_x`  | The x-indices of pixels corresponding to the shear layer (LCFS)    | The dimension may vary. If LCFS covers across the whole height of the image, the dimension is (ny_upsample). (e.g. 256)|
| `shear_contour_y`  | The y-indices of pixels corresponding to the shear layer (LCFS)    | The dimension may vary. If LCFS covers across the whole height of the image, the dimension is (ny_upsample). (e.g. 256)|

As shown in the above table, your "brt_arr" is the video data recorded from each GPI views. This will be upsampled to (nx_upsample) X (ny_upsample) X (number of time points) onto a regular grid from "r_arr" and "z_arr" which are typically irregular grid of GPI views. The indices of your "brt_true" with dead views must have NaN and will be interpolated.

"shear_contour_x" and "shear_contour_y" must have the same dimension, and their indices values must correspond to the upsampled grid (i.e. you should interpolate your LCFS coordinates).

The processed data will be saved in "GPI-blob-tracking/data/real_gpi".

# Tracking blobs in real GPI data
You can run the blob-tracking on real GPI data by running "run_track_blobs.sh". The key fields used are as following:

| Field                   | Description                                                                            |
|-------------------------|----------------------------------------------------------------------------------------|
| `--model`               | The directory to the trained model file                                                |
| `--filename`            | The directory to the processed GPI data file                                           |
| `--make_video`          | Use if you want to make the tracking video.                                            |
| `--hand_labels`         | Use if you want to make the tracking video including hand-labels of the blobs          |
| `--viou_threshold`      | The predicted blobs with VIoU below this threshold will be discarded                   |
| `--amp_threshold`       | The predicted blobs with maximum amplitude below this threshold will be discarded      |
| `--blob_life_threshold` | The predicted blobs last shorter than this threshold will be discarded                 |

The result file will be saved in "GPI-blob-tracking/data/real_gpi" and will contain the followings:

| Field                   | Description                                                                            |
|-------------------------|----------------------------------------------------------------------------------------|
| `output` if the model is for optical flow detection (e.g. RAFT, GMA, Flow Walk)  | img_flo: images and optical flows. Dimension: (num_time) X (2*n_x) X (n_y) X (num_RGB = 3) where the 1st half of the 2nd dimension is the images and the rest is the optical flows. |
| `output` if the model is not for optical flow detection (e.g. Mask R-CNN)       | img: images. Dimension: (num_time) X (n_x) X (n_y) X (num_RGB = 3) |
| `output_tracking` | The dictionary with keys as time index and values as the list of blob informations at that time. The blob information is [Blob ID, VIoU, x of blob center of mass, y of blob center of mass, polygon of the predicted blob, polygon of the brightness contour around this blob]   |

For hand labels, your file must be located in "GPI-blob-tracking/data/real_gpi/" and have dictionary with the keys as time index and values as the list of tuples where each tuple is (x, y) of the center of mass of the blobs labeled by human.
