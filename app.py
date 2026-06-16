from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

app = Flask(__name__)

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "skin_disease_classifier"

# Default Hospitals List (used for initialization and as a robust fallback)
DEFAULT_HOSPITALS_LIST = [
    {'name': 'Apollo Hospitals', 'specialization': 'Dermatology & Skin Care', 'address': 'Jubilee Hills, Hyderabad', 'phone': '+91-40-23607777', 'email': 'info@apollohospitals.com', 'rating': 4.8, 'available': True, 'timings': '24/7', 'city': 'Hyderabad', 'lat': 17.4265, 'lon': 78.4120},
    {'name': 'AIIMS Delhi', 'specialization': 'Multi-Specialty Hospital', 'address': 'Ansari Nagar, New Delhi', 'phone': '+91-11-26588500', 'email': 'director@aiims.edu', 'rating': 4.9, 'available': True, 'timings': '24/7', 'city': 'Delhi', 'lat': 28.5672, 'lon': 77.2100},
    {'name': 'Fortis Hospital', 'specialization': 'Dermatology & Cosmetic', 'address': 'Bannerghatta Road, Bangalore', 'phone': '+91-80-66214444', 'email': 'enquiry@fortishealthcare.com', 'rating': 4.7, 'available': True, 'timings': '8:00 AM - 10:00 PM', 'city': 'Bangalore', 'lat': 12.8946, 'lon': 77.5976},
    {'name': 'Max Super Specialty Hospital', 'specialization': 'Skin & Allergy', 'address': 'Saket, New Delhi', 'phone': '+91-11-26515050', 'email': 'info@maxhealthcare.com', 'rating': 4.6, 'available': True, 'timings': '24/7', 'city': 'Delhi', 'lat': 28.5276, 'lon': 77.2107},
    {'name': 'Manipal Hospital', 'specialization': 'Dermatology', 'address': 'HAL Airport Road, Bangalore', 'phone': '+91-80-25024444', 'email': 'info@manipalhospitals.com', 'rating': 4.5, 'available': True, 'timings': '8:00 AM - 9:00 PM', 'city': 'Bangalore', 'lat': 12.9610, 'lon': 77.6482}
]

# Initialize MongoDB connection
predictions_collection = None
patients_collection = None
hospitals_collection = None

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    mongo_client.admin.command('ping')
    db = mongo_client[DB_NAME]
    predictions_collection = db['predictions']
    patients_collection = db['patients']
    hospitals_collection = db['hospitals']
    
    # Initialize default hospitals if collection is empty or missing lat/lon
    if hospitals_collection.count_documents({}) == 0 or hospitals_collection.count_documents({'lat': {'$exists': False}}) > 0:
        if hospitals_collection.count_documents({'lat': {'$exists': False}}) > 0:
            hospitals_collection.delete_many({})
            print("Re-initializing hospitals with coordinates...")
        hospitals_collection.insert_many(DEFAULT_HOSPITALS_LIST)
        print("✓ Default hospitals data initialized with coordinates!")
    
    print("✓ MongoDB connected successfully!")
    print(f"  - Predictions: {predictions_collection.count_documents({})} records")
    print(f"  - Patients: {patients_collection.count_documents({})} records")
    print(f"  - Hospitals: {hospitals_collection.count_documents({})} records")
except ConnectionFailure as e:
    print(f"✗ MongoDB connection failed: {e}")
    print("  Dashboard will work with limited functionality")
    db = None
    predictions_collection = None
    patients_collection = None
    hospitals_collection = None

# Load the model - Direct approach
print("Loading model...")
checkpoint = torch.load('skin_disease_model.pth', map_location=torch.device('cpu'))

# Check the classifier input features from checkpoint
if 'classifier.1.weight' in checkpoint:
    in_features_from_checkpoint = checkpoint['classifier.1.weight'].shape[1]
    print(f"Detected {in_features_from_checkpoint} input features in checkpoint")
else:
    in_features_from_checkpoint = 1536  # Default

# Try to find matching EfficientNet variant
model = None
for variant_name, variant_func in [
    ('EfficientNet-B3', models.efficientnet_b3),
    ('EfficientNet-B2', models.efficientnet_b2),
    ('EfficientNet-B4', models.efficientnet_b4),
    ('EfficientNet-B1', models.efficientnet_b1),
    ('EfficientNet-B0', models.efficientnet_b0),
]:
    try:
        test_model = variant_func(weights=None)
        model_in_features = test_model.classifier[1].in_features
        print(f"Trying {variant_name}: {model_in_features} features", end=" ")
        
        if model_in_features == in_features_from_checkpoint:
            print("✓ MATCH!")
            model = test_model
            break
        else:
            print(f"✗ (expected {in_features_from_checkpoint})")
    except:
        pass

if model is None:
    print(f"Warning: No exact match found. Using B3 (closest)")
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
        # Direct state_dict - load it
        model.load_state_dict(checkpoint)
else:
    # If it's a full model
    model = checkpoint

model.eval()
print("Model loaded successfully!")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Class labels
CLASS_LABELS = ['NORMAL_SKIN', 'PSORIASIS', 'Ringworm', 'acne']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def save_prediction_to_db(prediction, confidence, all_probabilities, patient_info=None):
    """Save prediction result to MongoDB with patient information"""
    if predictions_collection is None:
        return None
    
    try:
        prediction_doc = {
            'prediction': prediction,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'timestamp': datetime.now(),
            'created_at': datetime.now().isoformat()
        }
        
        # Add patient information if provided
        if patient_info:
            prediction_doc['patient'] = {
                'name': patient_info.get('name', 'Anonymous'),
                'age': patient_info.get('age', ''),
                'gender': patient_info.get('gender', ''),
                'phone': patient_info.get('phone', ''),
                'email': patient_info.get('email', ''),
                'address': patient_info.get('address', ''),
                'city': patient_info.get('city', '')
            }
            
            # Also save/update patient in patients collection
            if patients_collection is not None:
                p_name = patient_info.get('name') or 'Self Check'
                p_phone = patient_info.get('phone') or ''
                
                query = {'phone': p_phone} if p_phone else {'name': p_name}
                
                patient_record = {
                    'name': p_name,
                    'age': patient_info.get('age', ''),
                    'gender': patient_info.get('gender', ''),
                    'phone': p_phone,
                    'email': patient_info.get('email', ''),
                    'address': patient_info.get('address', ''),
                    'city': patient_info.get('city', ''),
                    'last_visit': datetime.now(),
                    'updated_at': datetime.now().isoformat()
                }
                # Upsert patient record (update if exists, insert if new)
                patients_collection.update_one(
                    query,
                    {'$set': patient_record, '$setOnInsert': {'created_at': datetime.now().isoformat()}},
                    upsert=True
                )
        
        result = predictions_collection.insert_one(prediction_doc)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return None


def get_all_patients(limit=50):
    """Get all registered patients"""
    if patients_collection is None:
        return []
    
    try:
        patients = list(
            patients_collection.find({}, {'_id': 0})
            .sort('last_visit', -1)
            .limit(limit)
        )
        for patient in patients:
            if 'last_visit' in patient:
                patient['last_visit'] = patient['last_visit'].strftime('%Y-%m-%d %H:%M:%S')
        return patients
    except Exception as e:
        print(f"Error getting patients: {e}")
        return []


def get_patient_by_phone(phone):
    """Get patient by phone number"""
    if patients_collection is None:
        return None
    
    try:
        patient = patients_collection.find_one({'phone': phone}, {'_id': 0})
        if patient and 'last_visit' in patient:
            patient['last_visit'] = patient['last_visit'].strftime('%Y-%m-%d %H:%M:%S')
        return patient
    except Exception as e:
        print(f"Error getting patient: {e}")
        return None


def get_patient_history(phone):
    """Get prediction history for a specific patient"""
    if predictions_collection is None:
        return []
    
    try:
        predictions = list(
            predictions_collection.find({'patient.phone': phone}, {'_id': 0})
            .sort('timestamp', -1)
        )
        for pred in predictions:
            if 'timestamp' in pred:
                pred['timestamp'] = pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        return predictions
    except Exception as e:
        print(f"Error getting patient history: {e}")
        return []


def get_all_hospitals():
    """Get all hospitals from database"""
    if hospitals_collection is None:
        return []
    
    try:
        hospitals = list(hospitals_collection.find({}, {'_id': 0}))
        return hospitals
    except Exception as e:
        print(f"Error getting hospitals: {e}")
        return []


def add_hospital(hospital_data):
    """Add a new hospital to the database"""
    if hospitals_collection is None:
        return None
    
    try:
        result = hospitals_collection.insert_one(hospital_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error adding hospital: {e}")
        return None


def get_dashboard_stats():
    """Get statistics for the dashboard"""
    if predictions_collection is None:
        return {
            'total': 0,
            'normal': 0,
            'psoriasis': 0,
            'ringworm': 0,
            'acne': 0
        }
    
    try:
        total = predictions_collection.count_documents({})
        normal = predictions_collection.count_documents({'prediction': 'NORMAL_SKIN'})
        psoriasis = predictions_collection.count_documents({'prediction': 'PSORIASIS'})
        ringworm = predictions_collection.count_documents({'prediction': 'Ringworm'})
        acne = predictions_collection.count_documents({'prediction': 'acne'})
        
        return {
            'total': total,
            'normal': normal,
            'psoriasis': psoriasis,
            'ringworm': ringworm,
            'acne': acne
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            'total': 0,
            'normal': 0,
            'psoriasis': 0,
            'ringworm': 0,
            'acne': 0
        }


def get_recent_predictions(limit=20):
    """Get recent predictions for the dashboard"""
    if predictions_collection is None:
        return []
    
    try:
        predictions = list(
            predictions_collection.find({}, {'_id': 0})
            .sort('timestamp', -1)
            .limit(limit)
        )
        # Format timestamps for display
        for pred in predictions:
            if 'timestamp' in pred:
                pred['timestamp'] = pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        return predictions
    except Exception as e:
        print(f"Error getting recent predictions: {e}")
        return []


def get_timeline_data():
    """Get predictions timeline for the last 7 days"""
    if predictions_collection is None:
        return {'labels': [], 'data': []}
    
    try:
        labels = []
        data = []
        
        for i in range(6, -1, -1):
            date = datetime.now() - timedelta(days=i)
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            count = predictions_collection.count_documents({
                'timestamp': {
                    '$gte': start_of_day,
                    '$lte': end_of_day
                }
            })
            
            labels.append(date.strftime('%b %d'))
            data.append(count)
        
        return {'labels': labels, 'data': data}
    except Exception as e:
        print(f"Error getting timeline data: {e}")
        return {'labels': [], 'data': []}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Render the admin dashboard"""
    stats = get_dashboard_stats()
    predictions = get_recent_predictions(50)
    timeline_data = get_timeline_data()
    patients = get_all_patients(50)
    hospitals = get_all_hospitals()
    
    # Add patient and hospital counts to stats
    stats['patients_count'] = len(patients)
    stats['hospitals_count'] = len(hospitals)
    
    return render_template(
        'dashboard.html',
        stats=stats,
        predictions=predictions,
        timeline_data=timeline_data,
        patients=patients,
        hospitals=hospitals
    )


@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    stats = get_dashboard_stats()
    return jsonify(stats)


@app.route('/api/dashboard/predictions')
def api_dashboard_predictions():
    """API endpoint for recent predictions"""
    limit = request.args.get('limit', 20, type=int)
    predictions = get_recent_predictions(limit)
    return jsonify(predictions)


@app.route('/api/dashboard/timeline')
def api_dashboard_timeline():
    """API endpoint for timeline data"""
    timeline_data = get_timeline_data()
    return jsonify(timeline_data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get patient information from form data
        patient_info = {
            'name': request.form.get('patient_name', ''),
            'age': request.form.get('patient_age', ''),
            'gender': request.form.get('patient_gender', ''),
            'phone': request.form.get('patient_phone', ''),
            'email': request.form.get('patient_email', ''),
            'address': request.form.get('patient_address', ''),
            'city': request.form.get('patient_city', '')
        }
        
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Transform image
        img_tensor = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Prepare results
        predicted_idx = predicted.item()
        predicted_class_name = CLASS_LABELS[predicted_idx]
        confidence_score = confidence.item() * 100
        
        # Tweak prediction threshold: if the model predicts a disease with low confidence (< 55%),
        # default to NORMAL_SKIN to avoid false alarms.
        normal_prob = float(probabilities[0][0].item() * 100)
        if predicted_class_name != 'NORMAL_SKIN' and confidence_score < 55.0:
            print(f"Calibrating prediction: {predicted_class_name} (conf {confidence_score:.1f}%) is below disease threshold. Defaulting to NORMAL_SKIN.")
            predicted_class_name = 'NORMAL_SKIN'
            predicted_idx = 0
            confidence_score = normal_prob
            
        # Get all class probabilities
        all_probabilities = []
        for i, label in enumerate(CLASS_LABELS):
            prob = float(probabilities[0][i].item() * 100)
            if predicted_class_name == 'NORMAL_SKIN' and label == 'NORMAL_SKIN':
                prob = max(prob, confidence_score)
            all_probabilities.append({
                'label': label,
                'probability': prob
            })
        
        # Sort by probability
        all_probabilities.sort(key=lambda x: x['probability'], reverse=True)
        
        # Save prediction to MongoDB with patient info
        save_prediction_to_db(
            predicted_class_name,
            float(confidence_score),
            all_probabilities,
            patient_info if patient_info.get('name') or patient_info.get('phone') else None
        )
        
        # Get tips based on prediction
        tips = get_tips(predicted_class_name)
        
        return jsonify({
            'success': True,
            'prediction': predicted_class_name,
            'confidence': float(confidence_score),
            'all_probabilities': all_probabilities,
            'tips': tips
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============== PATIENT API ENDPOINTS ===============

@app.route('/api/patients')
def api_get_patients():
    """Get all registered patients"""
    limit = request.args.get('limit', 50, type=int)
    patients = get_all_patients(limit)
    return jsonify(patients)


@app.route('/api/patients/<phone>')
def api_get_patient(phone):
    """Get patient details by phone number"""
    patient = get_patient_by_phone(phone)
    if patient:
        return jsonify(patient)
    return jsonify({'error': 'Patient not found'}), 404


@app.route('/api/patients/<phone>/history')
def api_get_patient_history(phone):
    """Get patient prediction history"""
    history = get_patient_history(phone)
    return jsonify(history)


# =============== HOSPITAL API ENDPOINTS ===============

@app.route('/api/hospitals')
def api_get_hospitals():
    """Get all hospitals"""
    hospitals = get_all_hospitals()
    return jsonify(hospitals)


@app.route('/api/hospitals', methods=['POST'])
def api_add_hospital():
    """Add a new hospital"""
    try:
        data = request.get_json()
        hospital_data = {
            'name': data.get('name', ''),
            'specialization': data.get('specialization', ''),
            'address': data.get('address', ''),
            'phone': data.get('phone', ''),
            'email': data.get('email', ''),
            'rating': data.get('rating', 0),
            'available': data.get('available', True),
            'timings': data.get('timings', ''),
            'city': data.get('city', '')
        }
        result = add_hospital(hospital_data)
        if result:
            return jsonify({'success': True, 'id': result})
        return jsonify({'error': 'Failed to add hospital'}), 500
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
        city = request.args.get('city', '')

        if not lat or not lon:
            return jsonify({'error': 'Latitude and Longitude are required'}), 400

        import requests
        
        f_lat = float(lat)
        f_lon = float(lon)
        radius = 10000  # 10km radius
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="hospital"](around:{radius},{f_lat},{f_lon});
          way["amenity"="hospital"](around:{radius},{f_lat},{f_lon});
          node["amenity"="clinic"](around:{radius},{f_lat},{f_lon});
          way["amenity"="clinic"](around:{radius},{f_lat},{f_lon});
          node["amenity"="doctors"](around:{radius},{f_lat},{f_lon});
          way["amenity"="doctors"](around:{radius},{f_lat},{f_lon});
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
                if tags.get('addr:state'):
                    addr_parts.append(tags.get('addr:state'))
                address = ', '.join(addr_parts) if addr_parts else 'Address not available'
                
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
        
        geocode_data = []
        try:
            geocode_response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    'q': f"{city}, India",
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
            
            radius = 15000
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


def get_tips(condition):
    """Return helpful tips based on the diagnosed condition"""
    tips_dict = {
        'NORMAL_SKIN': [
            '✓ Your skin appears healthy!',
            '✓ Maintain good hygiene and moisturize regularly',
            '✓ Use sunscreen to protect from UV damage',
            '✓ Stay hydrated and eat a balanced diet',
            '✓ Continue your current skincare routine'
        ],
        'PSORIASIS': [
            '⚠ Psoriasis detected - consult a dermatologist',
            '• Keep skin moisturized to reduce scaling',
            '• Avoid triggers like stress, smoking, and alcohol',
            '• Use prescribed topical treatments regularly',
            '• Consider phototherapy if recommended',
            '• Maintain a healthy diet rich in omega-3'
        ],
        'Ringworm': [
            '⚠ Ringworm infection detected - seek medical treatment',
            '• Use antifungal creams as prescribed',
            '• Keep the affected area clean and dry',
            '• Avoid sharing personal items (towels, clothes)',
            '• Wash hands frequently',
            '• Complete the full treatment course'
        ],
        'acne': [
            '⚠ Acne detected - follow proper skincare routine',
            '• Cleanse face twice daily with gentle cleanser',
            '• Use non-comedogenic moisturizers',
            '• Avoid touching or picking at acne',
            '• Consider salicylic acid or benzoyl peroxide products',
            '• Consult dermatologist for severe cases',
            '• Maintain a healthy diet and reduce stress'
        ]
    }
    return tips_dict.get(condition, ['Consult a healthcare professional'])


@app.route('/info')
def info():
    return jsonify({
        'model': 'EfficientNet-B0',
        'classes': CLASS_LABELS,
        'input_size': '224x224',
        'accuracy_note': 'This is an AI prediction tool. Always consult a medical professional for proper diagnosis.'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
