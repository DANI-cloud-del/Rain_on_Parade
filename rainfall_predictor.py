import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta
import pickle
import os

class RainfallPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_nasa_data(self, city_name='Mumbai'):
        """
        Download data from NASA - completely automatic!
        """
        # City coordinates
        cities = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639)
        }
        
        lat, lon = cities.get(city_name, cities['Mumbai'])
        
        # NASA POWER API - FREE, NO AUTH NEEDED!
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        params = {
            'parameters': 'PRECTOTCORR,T2M,RH2M,WS2M',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': '20140101',
            'end': '20241231',
            'format': 'JSON'
        }
        
        print(f"📡 Downloading NASA data for {city_name}...")
        print(f"   Coordinates: {lat}, {lon}")
        
        try:
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                df = self._parse_nasa_response(data)
                print(f"✅ Downloaded {len(df)} days of data!")
                return df
            else:
                print(f"❌ Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def _parse_nasa_response(self, data):
        """Convert NASA JSON to DataFrame"""
        params = data['properties']['parameter']
        
        dates = list(params['PRECTOTCORR'].keys())
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates, format='%Y%m%d'),
            'precipitation': list(params['PRECTOTCORR'].values()),
            'temperature': list(params['T2M'].values()),
            'humidity': list(params['RH2M'].values()),
            'wind_speed': list(params['WS2M'].values())
        })
        
        # Clean data
        df = df.replace(-999, np.nan)
        df = df.ffill()
        
        return df
    
    def prepare_features(self, df):
        """Create ML features from raw data"""
        df = df.copy()
        
        # Time features
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Historical patterns (lagged features)
        for days_back in [7, 30, 90]:
            df[f'precip_lag_{days_back}'] = df['precipitation'].shift(days_back)
            df[f'temp_lag_{days_back}'] = df['temperature'].shift(days_back)
        
        # Rolling averages
        df['precip_30d_avg'] = df['precipitation'].rolling(30).mean()
        df['temp_30d_avg'] = df['temperature'].rolling(30).mean()
        
        # Target: precipitation 90 days in future
        df['target'] = df['precipitation'].shift(-90)
        
        # Remove rows with missing data
        df = df.dropna()
        
        return df
    
    def train(self, df):
        """Train the ML model"""
        print("\n🎯 Training model...")
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Select feature columns
        feature_cols = ['month', 'day_of_year', 'temperature', 'humidity', 
                       'wind_speed', 'precip_lag_7', 'precip_lag_30', 
                       'precip_lag_90', 'temp_lag_7', 'temp_lag_30', 
                       'precip_30d_avg', 'temp_30d_avg']
        
        X = df_features[feature_cols]
        y = df_features['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Test accuracy
        score = self.model.score(X_test, y_test)
        print(f"✅ Model trained! Accuracy: {score:.2%}")
        
        return self.model
    
    def predict(self, city='Mumbai', date_str='2025-02-01'):
        """Predict rainfall for a future date"""
        if self.model is None:
            return {'error': 'Model not trained yet'}
        
        # Download latest data
        df = self.fetch_nasa_data(city)
        
        if df is None:
            return {'error': 'Failed to fetch data'}
        
        # Get latest features
        df_features = self.prepare_features(df)
        
        feature_cols = ['month', 'day_of_year', 'temperature', 'humidity', 
                       'wind_speed', 'precip_lag_7', 'precip_lag_30', 
                       'precip_lag_90', 'temp_lag_7', 'temp_lag_30', 
                       'precip_30d_avg', 'temp_30d_avg']
        
        latest_features = df_features[feature_cols].iloc[-1:].values
        
        # Scale and predict
        features_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(features_scaled)[0]
        
        # Convert to rain probability
        rain_prob = min((prediction / 10) * 100, 100)
        
        return {
            'city': city,
            'date': date_str,
            'predicted_rainfall_mm': round(max(0, prediction), 2),
            'rain_probability': round(rain_prob, 1),
            'recommendation': self._get_recommendation(rain_prob)
        }
    
    def _get_recommendation(self, rain_prob):
        """Get event planning recommendation"""
        if rain_prob > 70:
            return '❌ High rain risk - Consider rescheduling'
        elif rain_prob > 40:
            return '⚠️ Moderate risk - Have backup plan'
        else:
            return '✅ Low risk - Good to go!'
    
    def save_model(self, filepath='weather_model.pkl'):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='weather_model.pkl'):
        """Load saved model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
            print(f"📂 Model loaded from {filepath}")
            return True
        return False


# MAIN EXECUTION
if __name__ == "__main__":
    print("🌧️ Rain on Parade - Weather Predictor")
    print("=" * 50)
    
    predictor = RainfallPredictor()
    
    # Check if model already exists
    if not predictor.load_model():
        print("\n🔄 No saved model found. Training new model...")
        
        # Download data for Mumbai
        data = predictor.fetch_nasa_data('Mumbai')
        
        if data is not None:
            # Train model
            predictor.train(data)
            
            # Save for future use
            predictor.save_model()
    
    # Make a prediction
    print("\n🔮 Making prediction...")
    result = predictor.predict(city='Mumbai', date_str='2025-02-15')
    
    print("\n" + "=" * 50)
    print(f"📅 Date: {result['date']}")
    print(f"🌆 City: {result['city']}")
    print(f"🌧️  Predicted Rainfall: {result['predicted_rainfall_mm']} mm")
    print(f"📊 Rain Probability: {result['rain_probability']}%")
    print(f"💡 Recommendation: {result['recommendation']}")
    print("=" * 50)
