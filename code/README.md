# Generating synthetic GPI data
You can generate a set of synthetic GPI data for training dataset by running the shell script "run_generate_synth_gpi.sh". The synthetic blobs have their sizes, amplitudes, speeds at each moment, as well as trajactories which are randomly yet accordingly assigned to mimic the blobs in the real GPI data. The key fields used are as following:

| Field          | Description                                         | Default    |
| -------------- |-----------------------------------------------------| ----------:|
| `--image_size` | The size of the image (width X height)              | [256, 256] |
| `--n_frame`    | The number of frames in a file                      | 200        |
| `--n_data`     | The number of data files to generate                | 30         |
| `--val_prop`   | The proportion of total frames for validation data  | 0.05       |

The output files will be saved in "GPI-blob-tracking/data/synthetic_gpi".

# Training models with synthetic GPI data
You can train a model in "motion" folder by running "run_train_model.sh". The key fields used are as following:

| Field          | Description                                  |
| -------------- |----------------------------------------------|
| `--name`       | The name of the run (model name)-synblobs    |
| `--num_steps`  | The number of epochs in training             |
| `--lr`         | Learning rate                                |
| `--gamma`      | Exponential weighting                        |
| `--wdecay`     | Weight decay                                 |

The fields used in this work are in "run_train_model.sh" for each model. The output model will be saved in "GPI-blob-tracking/models".

# Evaluating trained models with synthetic GPI data
You can get the evaluation scores of a trained model on synthetic GPI data by running "run_evaluate_model.sh". This will print out the evaluation scores with the metric corresponding to each model.

# Processing real GPI data


# Implementing your own motion detection model
