# finalproject-Itesm-mlops
Final project documents
## Analysis of the Dataset and Solutions for Vehicle Insurance Claims Fraud Detection

## Nature of the Dataset
The dataset in question is a CSV file containing information about vehicle insurance claims. Each row in the dataset represents an individual claim and its associated characteristics, such as accident location, driver's age, number of previous claims, etc. The purpose of the dataset is to provide enough information to build a model that can predict whether a claim is fraudulent or not based on these attributes.

Problem Addressed The dataset addresses the problem of fraud detection in vehicle insurance claims. Fraud in insurance claims is a significant problem for insurance companies as it can result in significant financial losses. This dataset provides the necessary information to build a machine learning model that can help identify fraudulent claims.

Developed Solutions On Kaggle, users have developed various solutions or notebooks for this dataset. Some notebooks focus on data exploration and visualization to better understand the data. Others focus on the construction and evaluation of machine learning models using various algorithms such as logistic regression, decision trees, random forests, and more.

## Minimum Solution Required to Train and Save a Model 
The minimum solution required to train and save a model for this dataset would involve:

    Data Loading: The data needs to be loaded into a pandas DataFrame for analysis and manipulation.
    * import pandas as pd

    data = pd.read_csv('fraud_oracle.csv')

    Data Preprocessing: This may include data cleaning, encoding categorical variables, normalizing numeric variables, splitting the data into training and testing sets, etc.

    Model Training: Select a machine learning algorithm, train it with the training data, and evaluate its performance with the testing data.
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    Save the Model: Once the model is trained, it can be saved for future use.
    import joblib
    joblib.dump(model, 'model.pkl')

## Model Scope Definition
The goal of this project is to build a proof-of-concept model for fraud detection in vehicle insurance claims. While the dataset allows for experimentation with a variety of machine learning models, simplicity is key for this project. Therefore, considering the use of a logistic regression model as the baseline due to its simplicity and interpretability would be appropriate.

Furthermore, although the dataset allows for the construction of more complex models such as neural networks or ensemble models, those may be beyond the scope of this project. However, once the baseline is established, more sophisticated models can be explored to improve the fraud detection accuracy.

The final outcome of this project will be a trained and saved machine learning model that can take in a new insurance claim and predict whether it is fraudulent or not. This model could potentially be used by insurance companies to assist in fraud detection. The model will be made available for consumption through the use of Fast API

## Additional Models on Kaggle
In addition to the models published on Kaggle, the random forest technique can also be used.

To train and save a Random Forest model with the vehicle insurance claims fraud detection dataset, you would follow a similar process as described above for logistic regression, with the difference that you would use the RandomForestClassifier class from sklearn.ensemble instead of LogisticRegression. Here are the detailed steps:

    *Data Loading:
    import pandas as pd

    data = pd.read_csv('vehicle_claims.csv')

    Data Preprocessing: This may include data cleaning, encoding categorical variables, normalizing numeric variables, splitting the data into training and testing sets, etc. Let's assume X is the DataFrame containing the input variables and y is the series containing the target variable (fraud or non-fraud).

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Random Forest Model Training:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    In this code, n_estimators=100 indicates that we want our forest to be composed of 100 trees. random_state=42 is used to ensure that the model produces the same results every time it is run.

    Model Evaluation:
    from sklearn.metrics import accuracy_score

    predictions = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')

    Save the Model: Once the model is trained and evaluated, it can be saved for future use.
    import joblib

    joblib.dump(model, 'random_forest_model.pkl')
    This code saves the trained model to a file named 'random_forest_model.pkl'

## Final Project Result
The project will be implemented with FastAPI, which is a modern and high-performance framework for building APIs with Python 3.6+. It is based on open standards for APIs, including OpenAPI (formerly known as Swagger) and JSON Schema.

Below is a step-by-step description of how to implement a machine learning model with FastAPI:

Installing FastAPI: If you haven't installed FastAPI yet, you can do so with pip:

    pip install fastapi
    You will also need an ASGI server, such as uvicorn:

    pip install uvicorn

    Creating the FastAPI application:
    Create a new Python file, for example, main.py, and add the following code:

    from fastapi import FastAPI

    app = FastAPI()

    @app.get('/')
    def read_root():
        return {"Hello": "World"}

    Loading the machine learning model: You can load the machine learning model you previously saved (e.g., random_forest_model.pkl) at the start of your application. This is done so that you don't have to load the model every time you make a prediction, which would be inefficient.

    import joblib

    model = joblib.load('random_forest_model.pkl')

    Creating an endpoint for predictions: Now, you can create a new endpoint for making predictions with your model. For this, you will need to use a POST method instead of a GET since you will be sending data to the server.

    from pydantic import BaseModel

    class Claim(BaseModel):
        feature1: float
        feature2: float
        # Add all the features of your dataset here

    @app.post('/predict')
    def predict_claim(claim: Claim):
        data = claim.dict()
        # Convert the data to the format your model expects, for example a NumPy array or a DataFrame
        prediction = model.predict(data)
        return {'prediction': prediction}

    Running the FastAPI application: To run the FastAPI application with uvicorn:

    uvicorn main:app --reload

Now, your application should be running at http://localhost:8000. You can make a POST request to http://localhost:8000/predict with a JSON body that matches the structure defined by your Pydantic model Claim to get a prediction from your model.

## Docker 
### Create a Dockerfile
This file must be in the root and should contain:
1. Python version
2. Working directory inside the container to /src
3. requirements.txt file from the host
4. Copy src directory from the host to the /src directory inside the container
5. The command to install the Python packages
6. The command to update the package index inside the container and then installs the vim text editor
7. The container port
8. The default command that will be executed when the container starts

### Build the Image
    *docker build -t fraud .
Where fraud is the name of my container
### Inspect the image
    *docker images
### Run the image in a container
    *docker run -d --rm --name fraud-container -p 3000:8000 fraud
### Check the container running 
* docker ps -a
### Debug the Container
* docker exec -ID_CONTAINER /bin/bash
### Make Predictions on the Local Machine
Create the network AIService
* docker network create AIservice
Once the container is running, you can make predictions using the API from your local machine. To do this, send an HTTP request to the following address:
* http://localhost:local_port/api_route
### Copy Container Logs to the Local Machine
* docker logs -f 492138c56145 | Select-String -Pattern "Debug
### Extract logs
* docker cp CONTAINER_ID:/app/server.log .
### Delete all the images
* docker rmi -f $(docker images -aq)
### Delette all containers
* docker rm -f $(docker ps -aq)

## Docker Compose
### Create a Docker Compose file
Create a file named docker-compose.yml in the root folder of your project, the file should contain:
1. version: This line specifies the version of Docker Compose you are using
2. services: This is where you define all the services that make up your application. 
3. build: This line tells Docker to build an image using the Dockerfile in the current directory.
4. command: This line specifies the command that will be run when the container starts.
5. volumes: This section maps directories on the host to directories in the Docker container. In this case, it maps the app directory on the host to the /code/app directory in the Docker container
6. ports: This section maps the service's ports to the host's ports. For example, "8000:8000" means that port 8000 on the host is mapped to port 8000 on the service
7. networks: This section defines the networks that the service is connected to. In this case, both the server and frontend services are connected to the AIservice network
8. aliases: This section defines the network aliases for the service. Network aliases are additional names that a service can be reached at from other services
9. depends_on: This section defines the dependencies between services. In this case, the frontend service depends on the server service, which means Docker Compose will ensure that the server service is started before the frontend service
10. AIservice: This is a network defined at the top level of the Compose file. The external: true option indicates that this network has been created outside of this Compose file

### Run Docker Compose
Be sure you are in the directory where the docker-compose.yml file is located
* docker-compose up --build
Once the containers are running, you can access the frontend at http://localhost:3000 and the API at http://localhost:8000.

### Curl request
For Frontend
* docker exec CONTAINER_ID curl -X GET http://localhost:3000/
* docker exec CONTAINER_ID curl -X GET http://localhost:8000/


