import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("exoplanets.csv")

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
st.title("Exoplanet Predictor")

# Sidebar input for features
st.sidebar.header("Input Features")
# Ensure the number of input features matches the trained model
feature1 = st.sidebar.number_input("Orbital Period [days]", min_value=0.0)
feature2 = st.sidebar.number_input("Transit Depth [ppm]", min_value=0.0)
feature3 = st.sidebar.number_input("Planetary Radius [Earthradii]", min_value=0.0)
feature4 = st.sidebar.number_input("Equilibrium Temperature [K]", min_value=0.0)
feature5 = st.sidebar.number_input("Stellar Radius [Solarradii]", min_value=0.0)

# Predict button
if st.sidebar.button("Predict"):
    # Input features must match the number of features used during training
    features_input = np.array([[feature1, feature2, feature3, feature4, feature5]])
    
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
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset from GitHub
url = "https://raw.githubusercontent.com/your-username/your-repo/main/exoplanets.csv"
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

df.dropna(inplace=True)

# Feature and target selection
features = df.drop(columns=['ExoplanetCandidate', 'ExoplanetConfirmed'])
target = df.ExoplanetCandidate

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42, test_size=0.30)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Streamlit interface
st.title("Exoplanet Predictor")

# Sidebar input for features
st.sidebar.header("Input Features")
# Ensure the number of input features matches the trained model
input_features = list(features.columns)
user_inputs = {}
for feature in input_features:
    user_inputs[feature] = st.sidebar.number_input(feature, min_value=0.0)

# Predict button
if st.sidebar.button("Predict"):
    features_input = np.array([list(user_inputs.values())])
    
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
