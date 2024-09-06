import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from pythreejs import *
from IPython.display import display

# Load the dataset (replace this path with the actual dataset path if not running on Kaggle)
df = pd.read_csv('/content/exoplanets.csv')

# Data preprocessing
df = df.rename(columns={
    'kepid':'KepID',
    'kepoi_name':'KOIName',
    'kepler_name':'KeplerName',
    'koi_disposition':'ExoplanetArchiveDisposition',
    'koi_pdisposition':'DispositionUsingKeplerData',
    'koi_score':'DispositionScore',
    'koi_fpflag_nt':'NotTransit-LikeFalsePositiveFlag',
    'koi_fpflag_ss':'koi_fpflag_ss',
    'koi_fpflag_co':'CentroidOffsetFalsePositiveFlag',
    'koi_fpflag_ec':'EphemerisMatchIndicatesContaminationFalsePositiveFlag',
    'koi_period':'OrbitalPeriod[days',
    'koi_period_err1':'OrbitalPeriodUpperUnc.[days',
    'koi_period_err2':'OrbitalPeriodLowerUnc.[days',
    'koi_time0bk':'TransitEpoch[BKJD',
    'koi_time0bk_err1':'TransitEpochUpperUnc.[BKJD',
    'koi_time0bk_err2':'TransitEpochLowerUnc.[BKJD',
    'koi_impact':'ImpactParamete',
    'koi_impact_err1':'ImpactParameterUpperUnc',
    'koi_impact_err2':'ImpactParameterLowerUnc',
    'koi_duration':'TransitDuration[hrs',
    'koi_duration_err1':'TransitDurationUpperUnc.[hrs',
    'koi_duration_err2':'TransitDurationLowerUnc.[hrs',
    'koi_depth':'TransitDepth[ppm',
    'koi_depth_err1':'TransitDepthUpperUnc.[ppm',
    'koi_depth_err2':'TransitDepthLowerUnc.[ppm',
    'koi_prad':'PlanetaryRadius[Earthradii',
    'koi_prad_err1':'PlanetaryRadiusUpperUnc.[Earthradii',
    'koi_prad_err2':'PlanetaryRadiusLowerUnc.[Earthradii',
    'koi_teq':'EquilibriumTemperature[K',
    'koi_teq_err1':'EquilibriumTemperatureUpperUnc.[K',
    'koi_teq_err2':'EquilibriumTemperatureLowerUnc.[K',
    'koi_insol':'InsolationFlux[Earthflux',
    'koi_insol_err1':'InsolationFluxUpperUnc.[Earthflux',
    'koi_insol_err2':'InsolationFluxLowerUnc.[Earthflux',
    'koi_model_snr':'TransitSignal-to-Nois',
    'koi_tce_plnt_num':'TCEPlanetNumbe',
    'koi_tce_delivname':'TCEDeliver',
    'koi_steff':'StellarEffectiveTemperature[K',
    'koi_steff_err1':'StellarEffectiveTemperatureUpperUnc.[K',
    'koi_steff_err2':'StellarEffectiveTemperatureLowerUnc.[K',
    'koi_slogg':'StellarSurfaceGravity[log10(cm/s**2)',
    'koi_slogg_err1':'StellarSurfaceGravityUpperUnc.[log10(cm/s**2)',
    'koi_slogg_err2':'StellarSurfaceGravityLowerUnc.[log10(cm/s**2)',
    'koi_srad':'StellarRadius[Solarradii',
    'koi_srad_err1':'StellarRadiusUpperUnc.[Solarradii',
    'koi_srad_err2':'StellarRadiusLowerUnc.[Solarradii',
    'ra':'RA[decimaldegrees',
    'dec':'Dec[decimaldegrees',
    'koi_kepmag':'Kepler-band[mag]'
})
df['ExoplanetCandidate'] = df['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)
df['ExoplanetConfirmed'] = df['ExoplanetArchiveDisposition'].apply(lambda x: 2 if x == 'CONFIRMED' else 1 if x == 'CANDIDATE' else 0)

# Drop unnecessary columns
df.drop(columns=['KeplerName', 'KOIName', 'EquilibriumTemperatureUpperUnc.[K',
                 'KepID', 'ExoplanetArchiveDisposition', 'DispositionUsingKeplerData',
                 'NotTransit-LikeFalsePositiveFlag', 'koi_fpflag_ss', 'CentroidOffsetFalsePositiveFlag',
                 'EphemerisMatchIndicatesContaminationFalsePositiveFlag', 'TCEDeliver',
                 'EquilibriumTemperatureLowerUnc.[K'], inplace=True)

# Drop rows with missing values
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

# Model building
num_features = len(features.columns)
model = Sequential()

# Add input layer with L2 regularization
model.add(Dense(256, activation='relu', input_dim=num_features, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())

# Add more hidden layers with L2 regularization and batch normalization
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())

# Add dropout layer
model.add(Dropout(0.5))

# Add output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))

# Streamlit interface

# Create a 3D Model of an exoplanet using pythreejs
def show_3d_exoplanet():
    planet = Mesh(
        SphereGeometry(radius=1, widthSegments=32, heightSegments=32),
        MeshStandardMaterial(color='yellow', roughness=1, metalness=0),
        position=[0, 0, 0]
    )
    scene = Scene(children=[
        planet, AmbientLight(color='#777777'),
        DirectionalLight(position=[3, 5, 1], intensity=0.6)
    ])
    renderer = Renderer(scene=scene, camera=PerspectiveCamera(
        position=[2, 2, 2], fov=20), controls=[OrbitControls(controlling=planet)])
    display(renderer)

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
    result = prediction[0][0]
    if result > 0.5:
        st.success(f"The exoplanet is likely a candidate with a probability of {result:.2f}")
    else:
        st.error(f"The exoplanet is unlikely a candidate with a probability of {result:.2f}")

# Display 3D model of an exoplanet
st.header("3D Model of an Exoplanet")
show_3d_exoplanet()

# Show rotating interesting facts about exoplanets
fact_display = st.empty()
for i in range(5):  # loop over facts
    fact_display.markdown(f"**Interesting Fact:** {facts[i % len(facts)]}")
    time.sleep(3)  # show each fact for 3 seconds
