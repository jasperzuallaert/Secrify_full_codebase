# Updated codebase for SECRiFY
This section contains the updated codebase to train secretability predictors and use them for predictions and visualizations of important features.
### Installed packages
See requirements.txt for the appropriate modules and versions to install
## Usage
### Training a model
To train a model, you have to specify a <data directory> and a <dataset_name> to train/validate/test on. The script will then look for datasets in `<data_directory>/<dataset_name>_train.csv`, in `<data_directory>/<dataset_name>_valid.csv`, and in `<data_directory>/<dataset_name>_test.csv`. The csv files should be comma-separated with `SEQ` and `LABEL` columns (with label being a 1 or a 0).

`` python train.py <data_directory> <dataset_name> ``

The training script will store the trained model in `saved_models/model_<dataset_name>.ckpt`

### Using the prediction notebook
When you have a trained model, you can make predictions for any sequences. First, create a `.csv` file with `ID` and `SEQ` columns with sequences that you want to see predicted.

Then, in the USER inputs cell, specify the prediction model(s) and the `.csv` file you just created. You can make a list of multiple prediction models, and the average predicted probabilities and saliency maps will then be reported.

Note that different prediction models can operate on different scales. A model trained on the (imbalanced) old_naive dataset will produce values that are mostly in the 0~0.3 range, whereas the new dataset will yield models that predict mostly in the 0.2~0.8 range.

