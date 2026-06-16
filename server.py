from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React Frontend

# Configuration
app.config['JWT_SECRET_KEY'] = 'super-secret-key-change-this'  # Change in production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)

jwt = JWTManager(app)
bcrypt = Bcrypt(app)

# Default Hospitals List (used for initialization and as a robust fallback)
DEFAULT_HOSPITALS_LIST = [
    {'name': 'Apollo Hospitals', 'specialization': 'Dermatology & Skin Care', 'address': 'Jubilee Hills, Hyderabad', 'phone': '+91-40-23607777', 'email': 'info@apollohospitals.com', 'rating': 4.8, 'available': True, 'timings': '24/7', 'city': 'Hyderabad', 'lat': 17.4265, 'lon': 78.4120},
    {'name': 'AIIMS Delhi', 'specialization': 'Multi-Specialty Hospital', 'address': 'Ansari Nagar, New Delhi', 'phone': '+91-11-26588500', 'email': 'director@aiims.edu', 'rating': 4.9, 'available': True, 'timings': '24/7', 'city': 'Delhi', 'lat': 28.5672, 'lon': 77.2100},
    {'name': 'Fortis Hospital', 'specialization': 'Dermatology & Cosmetic', 'address': 'Bannerghatta Road, Bangalore', 'phone': '+91-80-66214444', 'email': 'enquiry@fortishealthcare.com', 'rating': 4.7, 'available': True, 'timings': '8:00 AM - 10:00 PM', 'city': 'Bangalore', 'lat': 12.8946, 'lon': 77.5976},
    {'name': 'Max Super Specialty Hospital', 'specialization': 'Skin & Allergy', 'address': 'Saket, New Delhi', 'phone': '+91-11-26515050', 'email': 'info@maxhealthcare.com', 'rating': 4.6, 'available': True, 'timings': '24/7', 'city': 'Delhi', 'lat': 28.5276, 'lon': 77.2107},
    {'name': 'Manipal Hospital', 'specialization': 'Dermatology', 'address': 'HAL Airport Road, Bangalore', 'phone': '+91-80-25024444', 'email': 'info@manipalhospitals.com', 'rating': 4.5, 'available': True, 'timings': '8:00 AM - 9:00 PM', 'city': 'Bangalore', 'lat': 12.9610, 'lon': 77.6482}
]

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "skin_disease_classifier"

# Initialize MongoDB connection
predictions_collection = None
patients_collection = None
hospitals_collection = None
users_collection = None

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    mongo_client.admin.command('ping')
    db = mongo_client[DB_NAME]
    predictions_collection = db['predictions']
    patients_collection = db['patients']
    hospitals_collection = db['hospitals']
    users_collection = db['users']
    
    # Initialize default hospitals if collection is empty or missing lat/lon
    if hospitals_collection.count_documents({}) == 0 or hospitals_collection.count_documents({'lat': {'$exists': False}}) > 0:
        if hospitals_collection.count_documents({'lat': {'$exists': False}}) > 0:
            hospitals_collection.delete_many({})
            print("Re-initializing hospitals with coordinates...")
        
        hospitals_collection.insert_many(DEFAULT_HOSPITALS_LIST)
        print("✓ Default hospitals data initialized with coordinates!")
    
    print("✓ MongoDB connected successfully!")
except ConnectionFailure as e:
    print(f"✗ MongoDB connection failed: {e}")
    db = None

# ================= MODEL LOADING =================
# (Keeping original logic to ensure .pth compatibility)
print("Loading model...")
try:
    checkpoint = torch.load('skin_disease_model.pth', map_location=torch.device('cpu'))
    
    if 'classifier.1.weight' in checkpoint:
        in_features_from_checkpoint = checkpoint['classifier.1.weight'].shape[1]
    else:
        in_features_from_checkpoint = 1536

    model = None
    # Try different variants
    for _, variant_func in [
        ('EfficientNet-B3', models.efficientnet_b3),
        ('EfficientNet-B2', models.efficientnet_b2),
        ('EfficientNet-B4', models.efficientnet_b4),
        ('EfficientNet-B1', models.efficientnet_b1),
        ('EfficientNet-B0', models.efficientnet_b0),
    ]:
        try:
            test_model = variant_func(weights=None)
            if test_model.classifier[1].in_features == in_features_from_checkpoint:
                model = test_model
                break
        except: pass

    if model is None:
        model = models.efficientnet_b3(weights=None)

    # Modify classifier for 4 classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 4)
    )

    # Load state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    print("✓ Model loaded successfully!")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

CLASS_LABELS = ['NORMAL_SKIN', 'PSORIASIS', 'Ringworm', 'acne']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= TIPS HELPER =================
def get_tips(condition):
    tips_dict = {
        'NORMAL_SKIN': [
            '✓ Your skin appears healthy!', '✓ Maintain good hygiene', '✓ Use sunscreen', '✓ Stay hydrated'
        ],
        'PSORIASIS': [
            '⚠ Psoriasis detected', '• Keep skin moisturized', '• Avoid stress', '• Consult dermatologist'
        ],
        'Ringworm': [
            '⚠ Ringworm detected', '• Use antifungal creams', '• Keep clean and dry', '• Wash hands frequently'
        ],
        'acne': [
            '⚠ Acne detected', '• Cleanse face twice daily', '• Use non-comedogenic moisturizers', '• Don\'t pick at acne'
        ]
    }
    return tips_dict.get(condition, ['Consult a healthcare professional'])

# ================= AUTH ROUNDS =================

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if users_collection.find_one({'username': username}):
        return jsonify({'error': 'User already exists'}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    users_collection.insert_one({'username': username, 'password': hashed_password})
    
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = users_collection.find_one({'username': username})
    if user and bcrypt.check_password_hash(user['password'], password):
        access_token = create_access_token(identity=username)
        return jsonify({'token': access_token, 'username': username}), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def profile():
    current_user = get_jwt_identity()
    return jsonify({'username': current_user}), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'success',
        'message': 'DermaAI Backend API is running successfully!',
        'documentation': 'Use the Frontend at http://localhost:5173'
    }), 200

# ================= API ROUTES =================

@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    try:
        total = predictions_collection.count_documents({})
        stats = {
            'total': total,
            'normal': predictions_collection.count_documents({'prediction': 'NORMAL_SKIN'}),
            'psoriasis': predictions_collection.count_documents({'prediction': 'PSORIASIS'}),
            'ringworm': predictions_collection.count_documents({'prediction': 'Ringworm'}),
            'acne': predictions_collection.count_documents({'prediction': 'acne'}),
            'patients_count': patients_collection.count_documents({}),
            'hospitals_count': hospitals_collection.count_documents({})
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/recent', methods=['GET'])
def dashboard_recent():
    limit = request.args.get('limit', 10, type=int)
    predictions = list(predictions_collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(limit))
    for pred in predictions:
        if 'timestamp' in pred:
            pred['timestamp'] = pred['timestamp'].isoformat()
    return jsonify(predictions)

@app.route('/api/predict', methods=['POST'])
# @jwt_required()  # Optional: Protect prediction
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read and preprocess
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_LABELS[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Tweak prediction threshold: if the model predicts a disease with low confidence (< 55%),
        # default to NORMAL_SKIN to avoid false alarms on healthy skin.
        normal_prob = float(probabilities[0][0].item() * 100)
        if predicted_class != 'NORMAL_SKIN' and confidence_score < 55.0:
            print(f"Calibrating prediction: {predicted_class} (conf {confidence_score:.1f}%) is below disease threshold. Defaulting to NORMAL_SKIN.")
            predicted_class = 'NORMAL_SKIN'
            confidence_score = normal_prob
            
        all_probs = []
        for i, label in enumerate(CLASS_LABELS):
            prob = float(probabilities[0][i].item() * 100)
            # Adjust the returned probabilities to reflect the calibration
            if predicted_class == 'NORMAL_SKIN' and label == 'NORMAL_SKIN':
                prob = max(prob, confidence_score)
            all_probs.append({'label': label, 'probability': prob})
        all_probs.sort(key=lambda x: x['probability'], reverse=True)
        
        # Save to DB
        # Look for patient info in form data (Multipart)
        patient_info = {
            'name': request.form.get('patientName', ''),
            'age': request.form.get('patientAge', ''),
            'phone': request.form.get('patientPhone', '')
        }
        
        # Logic to save to DB (Simplified)
        doc = {
            'prediction': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': all_probs,
            'timestamp': datetime.now(),
            'patient': patient_info
        }
        predictions_collection.insert_one(doc)
        
        # Upsert to patients collection to ensure patients count is updated
        if patients_collection is not None:
            p_name = patient_info.get('name') or 'Self Check'
            p_phone = patient_info.get('phone') or ''
            p_age = patient_info.get('age') or ''
            
            # Use phone as identifier if available, otherwise use name
            query = {'phone': p_phone} if p_phone else {'name': p_name}
            
            patient_record = {
                'name': p_name,
                'age': p_age,
                'phone': p_phone,
                'last_visit': datetime.now(),
                'updated_at': datetime.now().isoformat()
            }
            try:
                patients_collection.update_one(
                    query,
                    {'$set': patient_record, '$setOnInsert': {'created_at': datetime.now().isoformat()}},
                    upsert=True
                )
            except Exception as db_err:
                print(f"Error saving patient: {db_err}")
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence_score,
            'tips': get_tips(predicted_class),
            'probabilities': all_probs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0 # Radius of the earth in km
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return 99999.0

@app.route('/api/hospitals/nearby', methods=['GET'])
def start_nearby_hospitals_search():
    try:
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        city = request.args.get('city', '')  # Optional manual city input

        if not lat or not lon:
            return jsonify({'error': 'Latitude and Longitude are required'}), 400

        import requests
        
        f_lat = float(lat)
        f_lon = float(lon)
        radius = 10000  # 10km radius
        
        # Use Overpass API for accurate radius-based search
        # Search for: hospitals, clinics, doctors with dermatology/skin specialization
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Overpass QL query - search for medical facilities within radius
        overpass_query = f"""
        [out:json][timeout:25];
        (
          // Hospitals
          node["amenity"="hospital"](around:{radius},{f_lat},{f_lon});
          way["amenity"="hospital"](around:{radius},{f_lat},{f_lon});
          
          // Clinics
          node["amenity"="clinic"](around:{radius},{f_lat},{f_lon});
          way["amenity"="clinic"](around:{radius},{f_lat},{f_lon});
          
          // Doctors/Medical offices
          node["amenity"="doctors"](around:{radius},{f_lat},{f_lon});
          way["amenity"="doctors"](around:{radius},{f_lat},{f_lon});
          
          // Specifically dermatology tagged places
          node["healthcare:speciality"~"dermatology|skin"](around:{radius},{f_lat},{f_lon});
          way["healthcare:speciality"~"dermatology|skin"](around:{radius},{f_lat},{f_lon});
        );
        out center tags;
        """
        
        headers = {'User-Agent': 'DermaAI_Skin_Disease_Classifier/1.0'}
        elements = []
        
        try:
            response = requests.post(
                overpass_url,
                data={'data': overpass_query},
                headers=headers,
                timeout=12
            )
            if response.status_code == 200:
                data = response.json()
                elements = data.get('elements', [])
            else:
                print(f"Overpass API error status {response.status_code}, trying local DB fallback")
        except Exception as api_err:
            print(f"Overpass API exception: {api_err}, trying local DB fallback")
        
        hospitals = []
        
        # If external API failed or returned nothing, use local database
        if not elements:
            print("Using local database fallback for nearby hospitals...")
            db_hospitals = []
            if hospitals_collection is not None:
                try:
                    db_hospitals = list(hospitals_collection.find({}))
                except Exception as db_err:
                    print(f"MongoDB fallback query error: {db_err}")
            
            if not db_hospitals:
                print("MongoDB is empty or unavailable, using in-memory default hospitals list...")
                db_hospitals = DEFAULT_HOSPITALS_LIST
                
            for h in db_hospitals:
                h_lat = h.get('lat')
                h_lon = h.get('lon')
                
                dist = 9999.0
                if h_lat is not None and h_lon is not None:
                    dist = calculate_haversine(f_lat, f_lon, h_lat, h_lon)
                
                spec = h.get('specialization', '')
                name = h.get('name', '')
                is_derma = 'derma' in spec.lower() or 'skin' in spec.lower() or 'derma' in name.lower()
                
                hospitals.append({
                    'name': name,
                    'address': h.get('address', 'Address not available'),
                    'lat': h_lat if h_lat is not None else f_lat,
                    'lon': h_lon if h_lon is not None else f_lon,
                    'type': spec if spec else 'Hospital',
                    'phone': h.get('phone', ''),
                    'website': h.get('website', h.get('email', '')),
                    'is_dermatology': is_derma,
                    'priority': 1 if is_derma else 0,
                    'distance': dist
                })
            # Sort by distance
            hospitals.sort(key=lambda x: x['distance'])
            return jsonify(hospitals[:25])
        else:
            seen_names = set()  # Avoid duplicates
            for item in elements:
                tags = item.get('tags', {})
                name = tags.get('name', '')
                
                if not name or name in seen_names:
                    continue
                    
                seen_names.add(name)
                
                # Get coordinates (for ways, use center)
                if 'center' in item:
                    item_lat = item['center']['lat']
                    item_lon = item['center']['lon']
                else:
                    item_lat = item.get('lat')
                    item_lon = item.get('lon')
                
                # Build address from available tags
                addr_parts = []
                if tags.get('addr:street'):
                    addr_parts.append(tags.get('addr:street'))
                if tags.get('addr:city'):
                    addr_parts.append(tags.get('addr:city'))
                if tags.get('addr:state'):
                    addr_parts.append(tags.get('addr:state'))
                address = ', '.join(addr_parts) if addr_parts else 'Address not available'
                
                # Determine type
                speciality = tags.get('healthcare:speciality', tags.get('speciality', ''))
                amenity = tags.get('amenity', 'medical')
                
                is_derma = 'derma' in speciality.lower() or 'skin' in speciality.lower() or 'derma' in name.lower()
                
                hospitals.append({
                    'name': name,
                    'address': address,
                    'lat': item_lat,
                    'lon': item_lon,
                    'type': speciality if speciality else amenity,
                    'phone': tags.get('phone', tags.get('contact:phone', '')),
                    'website': tags.get('website', tags.get('contact:website', '')),
                    'is_dermatology': is_derma,
                    'priority': 1 if is_derma else 0
                })
            
            # Sort by priority (dermatology first)
            hospitals.sort(key=lambda x: -x['priority'])
            hospitals = hospitals[:25]
        
        print(f"Found {len(hospitals)} medical facilities near {f_lat}, {f_lon}")
        return jsonify(hospitals)

    except Exception as e:
        print(f"Error finding hospitals: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/hospitals/search-city', methods=['GET'])
def search_hospitals_by_city():
    """Search hospitals by city name - handles geocoding server-side"""
    try:
        city = request.args.get('city', '')
        
        if not city:
            return jsonify({'error': 'City name is required'}), 400
        
        import requests
        
        headers = {'User-Agent': 'DermaAI_Skin_Disease_Classifier/1.0'}
        
        # Geocode the city using Nominatim
        geocode_data = []
        try:
            geocode_response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    'q': f"{city}, India",  # Append India for better results
                    'format': 'json',
                    'limit': 1
                },
                headers=headers,
                timeout=8
            )
            if geocode_response.status_code == 200:
                geocode_data = geocode_response.json()
        except Exception as geocode_err:
            print(f"Geocoding exception: {geocode_err}, trying fallback")

        if not geocode_data:
            # Try without ", India" suffix
            try:
                geocode_response = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={'q': city, 'format': 'json', 'limit': 1},
                    headers=headers,
                    timeout=8
                )
                if geocode_response.status_code == 200:
                    geocode_data = geocode_response.json()
            except Exception as geocode_err2:
                print(f"Fallback geocoding exception: {geocode_err2}")
        
        hospitals = []
        lat, lon = None, None
        
        if geocode_data:
            lat = float(geocode_data[0]['lat'])
            lon = float(geocode_data[0]['lon'])
            
            print(f"Geocoded '{city}' to: {lat}, {lon}")
            
            # Now search for hospitals at these coordinates
            radius = 15000  # 15km radius for city search
            overpass_url = "https://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:json][timeout:25];
            (
              node["amenity"="hospital"](around:{radius},{lat},{lon});
              way["amenity"="hospital"](around:{radius},{lat},{lon});
              node["amenity"="clinic"](around:{radius},{lat},{lon});
              way["amenity"="clinic"](around:{radius},{lat},{lon});
              node["amenity"="doctors"](around:{radius},{lat},{lon});
              way["amenity"="doctors"](around:{radius},{lat},{lon});
              node["healthcare:speciality"~"dermatology|skin"](around:{radius},{lat},{lon});
              way["healthcare:speciality"~"dermatology|skin"](around:{radius},{lat},{lon});
            );
            out center tags;
            """
            
            try:
                response = requests.post(
                    overpass_url,
                    data={'data': overpass_query},
                    headers=headers,
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    elements = data.get('elements', [])
                    
                    seen_names = set()
                    for item in elements:
                        tags = item.get('tags', {})
                        name = tags.get('name', '')
                        
                        if not name or name in seen_names:
                            continue
                        
                        seen_names.add(name)
                        
                        if 'center' in item:
                            item_lat = item['center']['lat']
                            item_lon = item['center']['lon']
                        else:
                            item_lat = item.get('lat')
                            item_lon = item.get('lon')
                        
                        addr_parts = []
                        if tags.get('addr:street'):
                            addr_parts.append(tags.get('addr:street'))
                        if tags.get('addr:city'):
                            addr_parts.append(tags.get('addr:city'))
                        address = ', '.join(addr_parts) if addr_parts else city
                        
                        speciality = tags.get('healthcare:speciality', tags.get('speciality', ''))
                        amenity = tags.get('amenity', 'medical')
                        is_derma = 'derma' in speciality.lower() or 'skin' in speciality.lower() or 'derma' in name.lower()
                        
                        hospitals.append({
                            'name': name,
                            'address': address,
                            'lat': item_lat,
                            'lon': item_lon,
                            'type': speciality if speciality else amenity,
                            'phone': tags.get('phone', tags.get('contact:phone', '')),
                            'website': tags.get('website', ''),
                            'is_dermatology': is_derma,
                            'priority': 1 if is_derma else 0
                        })
                else:
                    print(f"Overpass API search error status {response.status_code}")
            except Exception as overpass_err:
                print(f"Overpass API search exception: {overpass_err}")
        
        # Fallback to local DB if no hospitals found or geocoding failed
        if not hospitals:
            print("Using local database fallback for city search...")
            db_hospitals = []
            if hospitals_collection is not None:
                try:
                    db_hospitals = list(hospitals_collection.find({'city': {'$regex': f'^{city}$', '$options': 'i'}}))
                    if not db_hospitals:
                        db_hospitals = list(hospitals_collection.find({}))
                except Exception as db_err:
                    print(f"MongoDB fallback query error: {db_err}")
            
            if not db_hospitals:
                # Filter by city in memory
                db_hospitals = [h for h in DEFAULT_HOSPITALS_LIST if h['city'].lower() == city.lower()]
                if not db_hospitals:
                    db_hospitals = DEFAULT_HOSPITALS_LIST
                
            for h in db_hospitals:
                h_lat = h.get('lat')
                h_lon = h.get('lon')
                
                if lat is None and h_lat is not None:
                    lat = float(h_lat)
                if lon is None and h_lon is not None:
                    lon = float(h_lon)
                    
                spec = h.get('specialization', '')
                name = h.get('name', '')
                is_derma = 'derma' in spec.lower() or 'skin' in spec.lower() or 'derma' in name.lower()
                
                hospitals.append({
                    'name': name,
                    'address': h.get('address', 'Address not available'),
                    'lat': h_lat if h_lat is not None else (lat if lat else 0.0),
                    'lon': h_lon if h_lon is not None else (lon if lon else 0.0),
                    'type': spec if spec else 'Hospital',
                    'phone': h.get('phone', ''),
                    'website': h.get('website', h.get('email', '')),
                    'is_dermatology': is_derma,
                    'priority': 1 if is_derma else 0
                })
        
        hospitals.sort(key=lambda x: -x['priority'])
        hospitals = hospitals[:25]
        
        print(f"Found {len(hospitals)} facilities near {city}")
        
        return jsonify({
            'lat': lat if lat is not None else 0.0,
            'lon': lon if lon is not None else 0.0,
            'city': city,
            'hospitals': hospitals
        })
        
    except Exception as e:
        print(f"Error in city search: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
