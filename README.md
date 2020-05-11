# CSCI-1850-Midterm-Project

## How To Use

### Getting The Data

You can download the dataset at https://www.kaggle.com/c/gene-expression-prediction-cs1850-final/data. Extract only the train.npz, eval.npz, and seq_data.csv files into the project directory, then run `python data_prep.py` with the appropriate arguments. We strongly recommend -window_size 100 and -stride 100, and generally also -motif_length 3.

### Training a model

First, run `python train.py -h` to see a list of available command line arguments and their descriptions.

To continue training a saved model, simply put the path to the existing model as the model_path argument and specify the epochs, learning_rate, and batch_size to the desired values.

### Generating a Submission

To generate a Kaggle submission, run `python generate_submission [path to the model] [path to the data used to train the model]`.

## Model

### Current

We are currently using an ensemble of "fully"-convolutional models with 11 layers each. The convolutional blocks are (now quite loosely) based on those used by Codevilla et al (2018).

### History

In order from midterm experiments to current model:

1. Current model, minus sequence inputs
2. [1.] plus 8-layer sequence processing module
3. [1.] plus Bi-Gru sequence processing module
4. [1.] plus auto-encoder learned sequence embeddings
5. Current model


## References

Codevilla, F., Miiller, M., López, A., Koltun, V., & Dosovitskiy, A. (2018, May). End-to-end driving via conditional imitation learning. In 2018 IEEE International Conference on Robotics and Automation (ICRA) (pp. 1-9). IEEE.
