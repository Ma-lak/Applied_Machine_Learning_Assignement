# Applied_Machine_Learning_Assignement

# Description
This GitHub repositiory contains a main.py which benchmarks a Support Machine Vector model and a ResNet18 model on the BreastMNIST dataset, which can be found at https://zenodo.org/records/10519652. It also includes feature extracting using HOG and data augmentation.

The GitHub repository is structured as follows:

There are two folders ModelA and ModelB containing the code files for each model.

ModelA: there is a feature_pipeline file which performs feature extraction on the original data, there is a data_augmentation file containing the function augment_img to augment the image files and there is a model_training file which contains the function model_train_eval to train and evaluate the model based on different training data.

ModelB: there is a augment.py file which contains the augment_resnet_images functions, to augment input images, a basic_block.py file which contains a class defining ResNet's residual block, a resnet18.py file containing a class defining ResNet18's architecture and finally a resnet18_train_evaluate.py file which contains a function train and evaluate the ResNet model.

# How to run the code
To run the main.py file, one needs to install a local python environment through conda using the environment.yml file and the requirements.txt file. This is done by cloning the repository on a local machine and running the following commands on terminal:
conda env create -f environment.yml -n amls  # install env
conda activate amls # activate env
pip install -r requirements.txt # install dependencies
python main.py # spawn the server

# Once code is running
Different graphs will pop up during the program's run:
- First the original images' features distribution on 2D plan.
- Training and testing accuracy plotted against the regularisation parameter. (three graphs, one for each feature)
- The learning curve and training loss for the ResNet18 model. (two graphs, one for original dataset, and one for augmented dataset).
The classification report of each model will be printed out in the terminal, along with the data shape.
