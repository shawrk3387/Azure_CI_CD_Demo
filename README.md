# ML Engineer Prompt 1

**Core Deliverable**: you will need to provide a *containerized* solution that will allow this notebook and the corresponding model to be run. The code is not reproducible without both the steps and the containers to be built. You must use publicly available images. 

**You must include**:

* **Dockerfile** for the container to be built
* **Jupyter Notebook (.ipynb)** for the model to be built
* **README.md** that outlines the instructions to replicate your results

**About**:

In this exercise you will download the following dataset: 

https://www.kaggle.com/uciml/sms-spam-collection-dataset#spam.csv

1. Create a classifier to tell whether a SMS message is spam or ham, you can use any framework and model you wish. Be sure to document why you chose this framework and model, as well as the measurement you used to determine its "Accuracy". 

You must include a Jupyter notebook that solves for this, and one that is **reproducible**. Include all the steps to reproduce.

2. Create a container that will serve your model. We should be able to query your model as such (example in Python):

```python
import requests
url = "http://localhost:5000/yourmodelapi"
data = { 
    "input": ["spam message", "not spam"]
}
requests.post(url, json=data)
```

Provide documentation where needed, as well as the commands needed to run your application. You may use the language or framework of your choice.

# ML Engineer Prompt 2

Design a diagram illustrating a CI/CD pipeline end-to-end for your machine learning application (step #2 above) to which would be deployed on Kubernetes. In text briefly describe each stage of the pipeline, explaining important aspects of each stage. What considerations did you make? 

As part of this CI/CD, you will also need to deploy this microservice onto Kubernetes. Please include a YAML file that represents the deployment of your microservice onto Kubernetes with your diagram.

**You must include**:

* **application.yaml** that highlights the deployment of the microservice
* A design diagram illustrating the CI/CD pipeline.
* A text document that highlights the CI/CD design with your considerations
