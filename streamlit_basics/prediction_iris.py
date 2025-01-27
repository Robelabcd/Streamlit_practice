import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# '''
# @st.cache_data --> Decorator
# Purpose:
# This decorator is specific to Streamlit, a Python library used to build interactive web apps for data science and machine learning.

# @st.cache_data tells Streamlit to cache the results of the load_data function.

# Caching Benefits:
# --> If the load_data function is called multiple times with the same inputs, 
#     Streamlit will reuse the previously computed results instead of re-executing the function.
    
#     -> This speeds up the app, especially when the function involves expensive operations like loading large datasets.

#     Key Note: Cached data is recomputed only when the code inside the function changes or when the cache is manually cleared.

# '''

@st.cache_data
def load_data():
    # '''
    # A function is defined with no input arguments. 
    # Its purpose is to load and process the Iris dataset and return it in a usable format.
    # '''
    iris = load_iris()  #loading the dataset from scikit-learn
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names


df, target_names=load_data()
# '''
# df: Load the Iris dataset as a pandas DataFrame, including features and the species column (labels).
# target_names: Get the list of target names (e.g., ['setosa', 'versicolor', 'virginica']).
# '''

#Create an instance of the RandomForestClassifier class
model = RandomForestClassifier()
# '''
# a popular machine learning model in the scikit-learn library
# About Random Forest:
# --> It is an ensemble method that uses multiple decision trees to make predictions. 
#     -> Each tree votes, and the majority vote (for classification) or average prediction (for regression) is returned.

#     -> It helps reduce overfitting by combining multiple trees.

# Default Parameters:
# Without arguments, the model uses default parameters, such as:
#     n_estimators=100 (100 trees in the forest).
#     criterion='gini' (uses Gini impurity to measure the quality of a split).
#     Other hyperparameters control tree depth, minimum samples per leaf, etc.
# '''
#Train the model (NB: exclude the last column)
model.fit(df.iloc[:,:-1],df['species'])


st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

##Predict
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

st.write("Prediction")
st.write(f"The predicted species is: {predicted_species}")

# '''
# If prediction[0] = 0, predicted_species = target_names[0] = 'setosa'.
# If prediction[0] = 1, predicted_species = target_names[1] = 'versicolor'
# '''