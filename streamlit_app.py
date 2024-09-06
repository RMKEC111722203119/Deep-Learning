import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Exoplanet Predictor")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('exoplanets.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please upload the 'exoplanets.csv' file.")
        return None
    return df

df = load_data()

if df is not None:
    # Data preprocessing
    df = df.rename(columns={
        'kepid':'KepID',
        'kepoi_name':'KOIName',
        'kepler_name':'KeplerName',
        'koi_disposition':'ExoplanetArchiveDisposition',
        'koi_pdisposition':'DispositionUsingKeplerData',
        'koi_score':'DispositionScore',
        'koi_period':'OrbitalPeriod[days',
        'koi_depth':'TransitDepth[ppm',
        'koi_prad':'PlanetaryRadius[Earthradii',
        'koi_teq':'EquilibriumTemperature[K',
        'koi_srad':'StellarRadius[Solarradii',
    })

    df['ExoplanetCandidate'] = df['DispositionUsingKeplerData'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)

    # Drop unnecessary columns
    df.drop(columns=['KeplerName', 'KOIName', 'KepID', 'ExoplanetArchiveDisposition',
                     'DispositionUsingKeplerData'], inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Feature and target selection
    features = df[['OrbitalPeriod[days', 'TransitDepth[ppm', 'PlanetaryRadius[Earthradii',
                     'EquilibriumTemperature[K', 'StellarRadius[Solarradii']]
    target = df['ExoplanetCandidate']

    # Split dataset
    try:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    except ValueError as e:
        st.error(f"Error during train_test_split: {e}")
        st.stop()

    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = Sequential([
        Dense(256, activation='relu', input_dim=X_train_scaled.shape[1]),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    with st.spinner('Training the model...'):
        history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_data=(X_test_scaled, y_test))

    st.success("Model training completed.")

    # Sidebar inputs for prediction
    st.sidebar.header("Input Features for Prediction")
    feature1 = st.sidebar.number_input("Orbital Period [days]", min_value=0.0, value=300.0)
    feature2 = st.sidebar.number_input("Transit Depth [ppm]", min_value=0.0, value=100.0)
    feature3 = st.sidebar.number_input("Planetary Radius [Earthradii]", min_value=0.0, value=1.0)
    feature4 = st.sidebar.number_input("Equilibrium Temperature [K]", min_value=0.0, value=500.0)
    feature5 = st.sidebar.number_input("Stellar Radius [Solarradii]", min_value=0.0, value=1.0)

    if st.sidebar.button("Predict"):
        # Prepare input data
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        if prediction_proba > 0.5:
            st.sidebar.success(f"The exoplanet is likely a candidate with a probability of {prediction_proba:.2f}")
        else:
            st.sidebar.error(f"The exoplanet is unlikely a candidate with a probability of {prediction_proba:.2f}")

else:
    st.warning("Dataset could not be loaded. Please check the file.")
