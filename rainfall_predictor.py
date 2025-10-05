import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta
import pickle
import os
import json


class RainfallPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.cities_cache = {}  # Cache for city coordinates
        
    def load_global_cities(self):
        """Load comprehensive city database - India + World"""
        # COMPREHENSIVE INDIAN CITIES (All major cities + capitals)
        indian_cities = {
            # Major Metro Cities
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Bengaluru': (12.9716, 77.5946),  # Alternate name
            'Hyderabad': (17.3850, 78.4867),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            
            # State Capitals
            'Jaipur': (26.9124, 75.7873),  # Rajasthan
            'Lucknow': (26.8467, 80.9462),  # Uttar Pradesh
            'Chandigarh': (30.7333, 76.7794),  # Punjab/Haryana
            'Bhopal': (23.2599, 77.4126),  # Madhya Pradesh
            'Patna': (25.5941, 85.1376),  # Bihar
            'Ranchi': (23.3441, 85.3096),  # Jharkhand
            'Raipur': (21.2514, 81.6296),  # Chhattisgarh
            'Bhubaneswar': (20.2961, 85.8245),  # Odisha
            'Gandhinagar': (23.2156, 72.6369),  # Gujarat
            'Shimla': (31.1048, 77.1734),  # Himachal Pradesh
            'Srinagar': (34.0837, 74.7973),  # Jammu & Kashmir
            'Dehradun': (30.3165, 78.0322),  # Uttarakhand
            'Panaji': (15.4909, 73.8278),  # Goa
            'Imphal': (24.8170, 93.9368),  # Manipur
            'Kohima': (25.6747, 94.1105),  # Nagaland
            'Shillong': (25.5788, 91.8933),  # Meghalaya
            'Aizawl': (23.7271, 92.7176),  # Mizoram
            'Agartala': (23.8315, 91.2868),  # Tripura
            'Itanagar': (27.0844, 93.6053),  # Arunachal Pradesh
            'Gangtok': (27.3389, 88.6065),  # Sikkim
            'Dispur': (26.1433, 91.7898),  # Assam
            'Amaravati': (16.5415, 80.5134),  # Andhra Pradesh
            
            # Kerala Cities
            'Thiruvananthapuram': (8.5241, 76.9366),
            'Trivandrum': (8.5241, 76.9366),
            'Kochi': (9.9312, 76.2673),
            'Cochin': (9.9312, 76.2673),
            'Kozhikode': (11.2588, 75.7804),
            'Calicut': (11.2588, 75.7804),
            'Thrissur': (10.5276, 76.2144),
            'Kannur': (11.8745, 75.3704),
            'Kollam': (8.8932, 76.6141),
            'Palakkad': (10.7867, 76.6548),
            'Alappuzha': (9.4981, 76.3388),
            'Malappuram': (11.0510, 76.0711),
            'Kottayam': (9.5916, 76.5222),
            
            # Tamil Nadu Cities
            'Coimbatore': (11.0168, 76.9558),
            'Madurai': (9.9252, 78.1198),
            'Salem': (11.6643, 78.1460),
            'Tiruchirappalli': (10.7905, 78.7047),
            'Trichy': (10.7905, 78.7047),
            'Tirunelveli': (8.7139, 77.7567),
            'Vellore': (12.9165, 79.1325),
            
            # Karnataka Cities
            'Mysore': (12.2958, 76.6394),
            'Mangalore': (12.9141, 74.8560),
            'Hubli': (15.3647, 75.1240),
            'Belgaum': (15.8497, 74.4977),
            
            # Maharashtra Cities
            'Nagpur': (21.1458, 79.0882),
            'Nashik': (19.9975, 73.7898),
            'Aurangabad': (19.8762, 75.3433),
            'Thane': (19.2183, 72.9781),
            'Solapur': (17.6599, 75.9064),
            
            # Andhra Pradesh/Telangana
            'Visakhapatnam': (17.6868, 83.2185),
            'Vijayawada': (16.5062, 80.6480),
            'Guntur': (16.3067, 80.4365),
            'Warangal': (17.9689, 79.5941),
            
            # Gujarat Cities
            'Surat': (21.1702, 72.8311),
            'Vadodara': (22.3072, 73.1812),
            'Rajkot': (22.3039, 70.8022),
            
            # Rajasthan Cities
            'Jodhpur': (26.2389, 73.0243),
            'Udaipur': (24.5854, 73.7125),
            'Kota': (25.2138, 75.8648),
            
            # Uttar Pradesh Cities
            'Kanpur': (26.4499, 80.3319),
            'Varanasi': (25.3176, 82.9739),
            'Agra': (27.1767, 78.0081),
            'Meerut': (28.9845, 77.7064),
            'Allahabad': (25.4358, 81.8463),
            'Prayagraj': (25.4358, 81.8463),
            
            # Other Important Cities
            'Guwahati': (26.1445, 91.7362),  # Assam
            'Indore': (22.7196, 75.8577),  # Madhya Pradesh
            'Bhubaneswar': (20.2961, 85.8245),  # Odisha
            'Coorg': (12.4244, 75.7382),  # Karnataka
            'Puducherry': (11.9416, 79.8083),
            'Pondicherry': (11.9416, 79.8083),
        }
        
        # MAJOR WORLD CITIES
        world_cities = {
            # North America
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'San Francisco': (37.7749, -122.4194),
            'Miami': (25.7617, -80.1918),
            'Toronto': (43.6532, -79.3832),
            'Vancouver': (49.2827, -123.1207),
            'Mexico City': (19.4326, -99.1332),
            
            # Europe
            'London': (51.5074, -0.1278),
            'Paris': (48.8566, 2.3522),
            'Berlin': (52.5200, 13.4050),
            'Madrid': (40.4168, -3.7038),
            'Rome': (41.9028, 12.4964),
            'Amsterdam': (52.3676, 4.9041),
            'Barcelona': (41.3851, 2.1734),
            'Vienna': (48.2082, 16.3738),
            'Prague': (50.0755, 14.4378),
            'Moscow': (55.7558, 37.6173),
            
            # Asia
            'Tokyo': (35.6762, 139.6503),
            'Beijing': (39.9042, 116.4074),
            'Shanghai': (31.2304, 121.4737),
            'Hong Kong': (22.3193, 114.1694),
            'Singapore': (1.3521, 103.8198),
            'Bangkok': (13.7563, 100.5018),
            'Dubai': (25.2048, 55.2708),
            'Seoul': (37.5665, 126.9780),
            'Manila': (14.5995, 120.9842),
            'Jakarta': (6.2088, 106.8456),
            'Kuala Lumpur': (3.1390, 101.6869),
            'Islamabad': (33.6844, 73.0479),
            'Karachi': (24.8607, 67.0011),
            'Lahore': (31.5204, 74.3587),
            'Dhaka': (23.8103, 90.4125),
            'Kathmandu': (27.7172, 85.3240),
            'Colombo': (6.9271, 79.8612),
            
            # Australia & Oceania
            'Sydney': (-33.8688, 151.2093),
            'Melbourne': (-37.8136, 144.9631),
            'Brisbane': (-27.4698, 153.0251),
            'Perth': (-31.9505, 115.8605),
            'Auckland': (-36.8485, 174.7633),
            
            # Africa
            'Cairo': (30.0444, 31.2357),
            'Cape Town': (-33.9249, 18.4241),
            'Johannesburg': (-26.2041, 28.0473),
            'Nairobi': (-1.2864, 36.8172),
            'Lagos': (6.5244, 3.3792),
            
            # South America
            'São Paulo': (-23.5505, -46.6333),
            'Rio de Janeiro': (-22.9068, -43.1729),
            'Buenos Aires': (-34.6037, -58.3816),
            'Lima': (-12.0464, -77.0428),
            'Bogotá': (4.7110, -74.0721),
        }
        
        # Combine all cities
        self.cities_cache = {**indian_cities, **world_cities}
        return self.cities_cache
    
    def get_coordinates(self, city_name):
        """Get coordinates for any city (with fallback to geocoding)"""
        # Load cities if not already loaded
        if not self.cities_cache:
            self.load_global_cities()
        
        # Try exact match
        if city_name in self.cities_cache:
            return self.cities_cache[city_name]
        
        # Try case-insensitive match
        for city, coords in self.cities_cache.items():
            if city.lower() == city_name.lower():
                return coords
        
        # Fallback: Use OpenStreetMap Nominatim for unknown cities
        print(f"🔍 City '{city_name}' not in database, searching...")
        try:
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': city_name,
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'RainOnParade/1.0'}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    print(f"✅ Found: {city_name} at ({lat}, {lon})")
                    # Cache it
                    self.cities_cache[city_name] = (lat, lon)
                    return (lat, lon)
        except Exception as e:
            print(f"⚠️ Geocoding failed: {e}")
        
        # Ultimate fallback: Mumbai
        print(f"⚠️ Using default location (Mumbai)")
        return (19.0760, 72.8777)
    
    def fetch_nasa_data(self, city_name='Mumbai'):
        """
        Download data from NASA POWER API for ANY location globally!
        """
        # Get coordinates
        lat, lon = self.get_coordinates(city_name)
        
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
        
        # Clean data (NASA uses -999 for missing values)
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
                'scaler': self.scaler,
                'cities_cache': self.cities_cache
            }, f)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='weather_model.pkl'):
        """Load saved model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.cities_cache = data.get('cities_cache', {})
            print(f"📂 Model loaded from {filepath}")
            return True
        return False


# MAIN EXECUTION
if __name__ == "__main__":
    print("🌧️ Rain on Parade - Global Weather Predictor")
    print("=" * 50)
    
    predictor = RainfallPredictor()
    
    # Load city database
    predictor.load_global_cities()
    print(f"🌍 Loaded {len(predictor.cities_cache)} cities worldwide!")
    
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
    
    # Test predictions for different cities
    test_cities = ['Mumbai', 'Thiruvananthapuram', 'London', 'New York', 'Tokyo']
    
    for city in test_cities:
        print(f"\n🔮 Prediction for {city}...")
        result = predictor.predict(city=city, date_str='2025-02-15')
        
        if 'error' not in result:
            print(f"   🌧️ Rainfall: {result['predicted_rainfall_mm']} mm")
            print(f"   📊 Probability: {result['rain_probability']}%")
            print(f"   💡 {result['recommendation']}")
