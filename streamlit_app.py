import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import time

# Load the dataset from GitHub
url = "https://github.com/RMKEC111722203119/Deep-Learning/blob/main/exoplanets.csv"
df = pd.read_csv("../blob/main/exoplanets.csv")

# Data preprocessing
df['ExoplanetCandidate'] = df['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)
df['ExoplanetConfirmed'] = df['ExoplanetArchiveDisposition'].apply(lambda x: 2 if x == 'CONFIRMED' else 1 if x == 'CANDIDATE' else 0)

# Drop unnecessary columns
df.drop(columns=['KeplerName', 'KOIName', 'EquilibriumTemperatureUpperUnc.[K', 'KepID', 
                 'ExoplanetArchiveDisposition', 'DispositionUsingKeplerData', 
                 'NotTransit-LikeFalsePositiveFlag', 'koi_fpflag_ss', 'CentroidOffsetFalsePositiveFlag', 
                 'EphemerisMatchIndicatesContaminationFalsePositiveFlag', 'TCEDeliver', 
                 'EquilibriumTemperatureLowerUnc.[K'], inplace=True)

df.dropna(inplace=True)

# Feature and target selection
features = df.drop(columns=['ExoplanetCandidate', 'ExoplanetConfirmed'])
target = df.ExoplanetCandidate

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42, test_size=.30)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Streamlit interface

def show_3d_exoplanet():
    fig = go.Figure(data=[go.Mesh3d(
        x=[0], y=[0], z=[0],
        opacity=0.5,
        color='yellow'
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-2, 2]),
            yaxis=dict(nticks=4, range=[-2, 2]),
            zaxis=dict(nticks=4, range=[-2, 2]),
        ),
        margin=dict(r=10, l=10, b=10, t=10)
    )
    st.plotly_chart(fig)

# Create a list of exoplanet facts
facts = [
    "Exoplanets are planets that orbit stars outside our solar system.",
    "The Kepler Space Telescope has discovered more than 2,600 exoplanets.",
    "Some exoplanets have been found in the 'habitable zone', where conditions might support life.",
    "Hot Jupiters are gas giants that orbit very close to their stars, much closer than Mercury to the Sun.",
    "There are exoplanets made entirely of diamond, like the exoplanet 55 Cancri e."
]

# Streamlit layout
st.title("Exoplanet Predictor with 3D Visualization and Facts")
st.write("Input the features of a potential exoplanet and predict whether it’s a candidate or not.")

# Create a sidebar for input
st.sidebar.header("Input Features")
feature1 = st.sidebar.number_input("Orbital Period [days]", min_value=0.0)
feature2 = st.sidebar.number_input("Transit Depth [ppm]", min_value=0.0)
feature3 = st.sidebar.number_input("Planetary Radius [Earthradii]", min_value=0.0)
feature4 = st.sidebar.number_input("Equilibrium Temperature [K]", min_value=0.0)
feature5 = st.sidebar.number_input("Stellar Radius [Solarradii]", min_value=0.0)

# Predict button
if st.sidebar.button("Predict"):
    features_input = np.array([[feature1, feature2, feature3, feature4, feature5]])
    features_scaled = scaler.transform(features_input)
    prediction = model.predict(features_scaled)
    if prediction[0] == 1:
        st.write("This exoplanet is likely a candidate!")
    else:
        st.write("This exoplanet is likely not a candidate.")

# Show random facts and 3D visualization
if st.sidebar.checkbox("Show 3D Exoplanet Visualization"):
    show_3d_exoplanet()

if st.sidebar.checkbox("Show Random Exoplanet Facts"):
    st.write(np.random.choice(facts))
