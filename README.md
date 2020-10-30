## Project Description:
This project illustrates the CI CD Pipeline of Spam Classification model hosted in Azure Kubernetes Services

## Steps to run the api as a docker container

1. Use the docker file present in the project directory to create a docker image.
2. Run the docker image as a container in local system
3. The API can then be accessed using the endpoint - http://localhost:5000/infer

## Model Training

Run the notebook (spam_classification.ipynb) located inside the folder "notebooks" in the project directory.
Note: The model used in this project is a baseline model. The model can be further optimized by tuning the hyperparameters or implementing more sophisticated network architecture such as bidirectional LSTM.

## Azure CI CD Pipeline
Please refer the word doc "Azure_CI_CD_Pipeline.docx" located in the project directory to know more about the deployment setup.
