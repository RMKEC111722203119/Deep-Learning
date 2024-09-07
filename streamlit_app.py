import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from GitHub
url = "exoplanets.csv"
df = pd.read_csv(url)

df = df.rename(columns={
    'kepid':'KepID', 'kepoi_name':'KOIName', 'kepler_name':'KeplerName',
    'koi_disposition':'ExoplanetArchiveDisposition', 'koi_pdisposition':'DispositionUsingKeplerData',
    'koi_score':'DispositionScore', 'koi_fpflag_nt':'NotTransit-LikeFalsePositiveFlag',
    'koi_fpflag_ss':'koi_fpflag_ss', 'koi_fpflag_co':'CentroidOffsetFalsePositiveFlag',
    'koi_fpflag_ec':'EphemerisMatchIndicatesContaminationFalsePositiveFlag',
    'koi_period':'OrbitalPeriod[days', 'koi_period_err1':'OrbitalPeriodUpperUnc.[days',
    'koi_period_err2':'OrbitalPeriodLowerUnc.[days', 'koi_time0bk':'TransitEpoch[BKJD',
    'koi_time0bk_err1':'TransitEpochUpperUnc.[BKJD', 'koi_time0bk_err2':'TransitEpochLowerUnc.[BKJD',
    'koi_impact':'ImpactParamete', 'koi_impact_err1':'ImpactParameterUpperUnc',
    'koi_impact_err2':'ImpactParameterLowerUnc', 'koi_duration':'TransitDuration[hrs',
    'koi_duration_err1':'TransitDurationUpperUnc.[hrs', 'koi_duration_err2':'TransitDurationLowerUnc.[hrs',
    'koi_depth':'TransitDepth[ppm', 'koi_depth_err1':'TransitDepthUpperUnc.[ppm',
    'koi_depth_err2':'TransitDepthLowerUnc.[ppm', 'koi_prad':'PlanetaryRadius[Earthradii',
    'koi_prad_err1':'PlanetaryRadiusUpperUnc.[Earthradii', 'koi_prad_err2':'PlanetaryRadiusLowerUnc.[Earthradii',
    'koi_teq':'EquilibriumTemperature[K', 'koi_teq_err1':'EquilibriumTemperatureUpperUnc.[K',
    'koi_teq_err2':'EquilibriumTemperatureLowerUnc.[K', 'koi_insol':'InsolationFlux[Earthflux',
    'koi_insol_err1':'InsolationFluxUpperUnc.[Earthflux', 'koi_insol_err2':'InsolationFluxLowerUnc.[Earthflux',
    'koi_model_snr':'TransitSignal-to-Nois', 'koi_tce_plnt_num':'TCEPlanetNumbe',
    'koi_tce_delivname':'TCEDeliver', 'koi_steff':'StellarEffectiveTemperature[K',
    'koi_steff_err1':'StellarEffectiveTemperatureUpperUnc.[K', 'koi_steff_err2':'StellarEffectiveTemperatureLowerUnc.[K',
    'koi_slogg':'StellarSurfaceGravity[log10(cm/s**2)', 'koi_slogg_err1':'StellarSurfaceGravityUpperUnc.[log10(cm/s**2)',
    'koi_slogg_err2':'StellarSurfaceGravityLowerUnc.[log10(cm/s**2)', 'koi_srad':'StellarRadius[Solarradii',
    'koi_srad_err1':'StellarRadiusUpperUnc.[Solarradii', 'koi_srad_err2':'StellarRadiusLowerUnc.[Solarradii',
    'ra':'RA[decimaldegrees', 'dec':'Dec[decimaldegrees', 'koi_kepmag':'Kepler-band[mag]'
})

# Data preprocessing
df['ExoplanetCandidate'] = df['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)
df['ExoplanetConfirmed'] = df['ExoplanetArchiveDisposition'].apply(lambda x: 2 if x == 'CONFIRMED' else 1 if x == 'CANDIDATE' else 0)

# Drop unnecessary columns
df.drop(columns=['KeplerName', 'KOIName', 'EquilibriumTemperatureUpperUnc.[K', 'KepID', 
                 'ExoplanetArchiveDisposition', 'DispositionUsingKeplerData', 
                 'NotTransit-LikeFalsePositiveFlag', 'koi_fpflag_ss', 'CentroidOffsetFalsePositiveFlag', 
                 'EphemerisMatchIndicatesContaminationFalsePositiveFlag', 'TCEDeliver', 
                 'EquilibriumTemperatureLowerUnc.[K'], inplace=True)

# Selecting 10 relevant features
selected_features = ['OrbitalPeriod[days', 'TransitDepth[ppm', 'PlanetaryRadius[Earthradii', 
                      'EquilibriumTemperature[K', 'StellarRadius[Solarradii', 'RA[decimaldegrees', 
                      'Dec[decimaldegrees', 'TransitSignal-to-Nois', 'StellarEffectiveTemperature[K', 
                      'StellarSurfaceGravity[log10(cm/s**2)']
df = df[selected_features + ['ExoplanetCandidate']]

df.dropna(inplace=True)

# Feature and target selection
features = df.drop(columns=['ExoplanetCandidate'])
target = df.ExoplanetCandidate

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42, test_size=0.30)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and evaluate the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit interface
st.title("Exoplanet Predictor")

# Main input area for features
st.header("Input Features")
input_features = {feature: st.number_input(feature, min_value=0.0) for feature in features.columns}

# Predict button
st.write(f"Model Accuracy: {accuracy:.2f}")
if st.button("Predict"):
    features_input = np.array([list(input_features.values())])
    
    # Check if the input feature length matches the expected number of features
    if features_input.shape[1] == X_train.shape[1]:
        features_scaled = scaler.transform(features_input)
        prediction = model.predict(features_scaled)
        if prediction[0] == 1:
            st.write("This exoplanet is likely a candidate!")
        else:
            st.write("This exoplanet is likely not a candidate.")
    else:
        st.error(f"Number of input features must be {X_train.shape[1]}. You provided {features_input.shape[1]}.")



