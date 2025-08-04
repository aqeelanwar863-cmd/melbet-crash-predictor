
import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated data
def fetch_data(n=500):
    data = {
        'crash_point': [round(np.random.exponential(scale=2.0), 2) for _ in range(n)]
    }
    return pd.DataFrame(data)

# Preprocess
def preprocess(df):
    df['lag_1'] = df['crash_point'].shift(1)
    df['lag_2'] = df['crash_point'].shift(2)
    df['mean_5'] = df['crash_point'].rolling(5).mean()
    df['std_5'] = df['crash_point'].rolling(5).std()
    df.dropna(inplace=True)
    return df

# Train Model
def train_model(df):
    X = df[['lag_1', 'lag_2', 'mean_5', 'std_5']]
    y = df['crash_point']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)
    return model, scaler, df

# Predict
def predict(model, scaler, df, risk):
    latest = df.iloc[-1]
    features = pd.DataFrame([{
        'lag_1': latest['crash_point'],
        'lag_2': df.iloc[-2]['crash_point'],
        'mean_5': df['crash_point'][-5:].mean(),
        'std_5': df['crash_point'][-5:].std()
    }])
    X_scaled = scaler.transform(features)
    base = model.predict(X_scaled)[0]
    adjusted = base * (0.8 + 0.4 * risk)
    return round(adjusted, 2)

# Streamlit UI
st.title("🎯 Mel-Bet Crash Predictor")

risk = st.slider("🧠 Risk Preference", 0.0, 1.0, 0.5, 0.05)
if st.button("Train & Predict"):
    df = fetch_data()
    df = preprocess(df)
    model, scaler, df = train_model(df)
    prediction = predict(model, scaler, df, risk)
    st.success(f"📊 Next crash may happen around **{prediction}x** (Risk: {risk})")
