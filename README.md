# Airbus Ship Detection

The goal of the test project is to build a semantic segmentation model. 

dataset - directory with all data from kaggle  
analysis.ipynb - exploratory data analysis  

requirements.txt - required python modules and versions   
dice_coefficient.py - dice metrics and loss for model training  
rle_encode_decode.py - run length strings encoding and decoding  
train_test_dataset.py - datasets for training and inference  
model_build.py - function to create a U-Net model  
model_training.py - function to train model  
model_tuning.py - function to tune model  
model_inference.py - function for model inference  
main.py - the entry point of a program  
model.h5 - saved trained model  
results.csv - encoded inference results written in kaggle required format  

submission_score.jpg - screenshot of kaggle submission  
model_work_example.jpg - matplotlib example figure saved  

To run the project run the main file (uncomment what you need).

If analysis.ipynb doesn't load try:  
https://nbviewer.org/github/lesha13/AirbusShipDetection/blob/main/analysis.ipynb