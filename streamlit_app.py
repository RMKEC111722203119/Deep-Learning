import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

# Predictions and performance metrics
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit interface
st.title("Exoplanet Predictor")


# Main input area for features
input_features = {feature: st.number_input(feature, min_value=0.0) for feature in features.columns}

# Predict button
if st.button("Predict"):
    features_input = np.array([list(input_features.values())])
    
    # Check if the input feature length matches the expected number of features
    if features_input.shape[1] == X_train.shape[1]:
        features_scaled = scaler.transform(features_input)
        prediction = model.predict(features_scaled)
        if prediction[0] == 1:
            st.write("This exoplanet is likely a candidate for habitability!")
        else:
            st.write("This exoplanet is likely not a candidate for habitability.")
    
    st.header("Model Performance Metrics")
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")
    st.write("**Confusion Matrix**:")
    st.write(conf_matrix)

    
# Explanations for input features
st.header("Input Features")
st.write("""
The following input features are used to predict whether an exoplanet is a candidate for habitability. Understanding these features can help in determining the likelihood of an exoplanet being within a habitable zone:

1. **Orbital Period [days]**: 
   - **Description**: The time an exoplanet takes to complete one orbit around its star.
   - **Contribution**: A suitable orbital period ensures that the exoplanet is neither too close nor too far from its star, which is crucial for maintaining conditions that might support liquid water.
   - **Optimal Range**: 10 to 200 days. Exoplanets within this range are more likely to be in the habitable zone where liquid water could exist.

2. **Transit Depth [ppm]**: 
   - **Description**: The percentage reduction in the star's brightness due to the exoplanet passing in front of it.
   - **Contribution**: Helps in estimating the size of the exoplanet and its potential atmosphere. A deeper transit indicates a larger or closer exoplanet.
   - **Optimal Range**: 100 to 1000 ppm. Lower values typically indicate smaller planets or those further from the observer.

3. **Planetary Radius [Earthradii]**: 
   - **Description**: The radius of the exoplanet compared to Earth's radius.
   - **Contribution**: Affects the planet's surface conditions and potential to retain an atmosphere. Planets with radii similar to Earth's are more likely to have Earth-like conditions.
   - **Optimal Range**: 0.8 to 2.0 Earth radii. Planets within this range are often considered for habitability assessments.

4. **Equilibrium Temperature [K]**: 
   - **Description**: The temperature of the exoplanet based on its distance from its star and the star's brightness.
   - **Contribution**: Helps in determining if the planet is within a temperature range where liquid water could exist.
   - **Optimal Range**: 200 to 320 K. Temperatures within this range are ideal for habitability.

5. **Stellar Radius [Solarradii]**: 
   - **Description**: The radius of the star around which the exoplanet orbits.
   - **Contribution**: Impacts the habitable zone of the star. Larger stars have more expansive habitable zones.
   - **Optimal Range**: 0.8 to 1.2 Solarradii. Stars in this range are similar to the Sun, which supports habitability studies.

6. **RA [decimaldegrees]**: 
   - **Description**: Right ascension of the star’s position in the sky.
   - **Contribution**: Helps in locating the star but does not directly affect habitability.

7. **Dec [decimaldegrees]**: 
   - **Description**: Declination of the star’s position in the sky.
   - **Contribution**: Helps in locating the star but does not directly affect habitability.

8. **Transit Signal-to-Noise**: 
   - **Description**: The ratio of the transit signal strength to the noise level in the data.
   - **Contribution**: Higher values indicate more reliable transit data, which improves the accuracy of habitability predictions.
   - **Optimal Range**: Values above 10 are preferred for reliable detection.

9. **Stellar Effective Temperature [K]**: 
   - **Description**: The effective temperature of the star.
   - **Contribution**: Influences the location of the habitable zone around the star. Stars with temperatures similar to the Sun are ideal for habitability.
   - **Optimal Range**: 4000 to 7000 K. Temperatures within this range are suitable for supporting habitable conditions.

10. **Stellar Surface Gravity [log10(cm/s**2)]**: 
    - **Description**: The gravity of the star's surface.
    - **Contribution**: Affects the star's evolution and stability. Stable stars are better for predicting habitable zones.
    - **Optimal Range**: 4.0 to 4.5. Values in this range suggest stable stellar conditions.

""")
