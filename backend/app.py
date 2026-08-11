import math
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image, UnidentifiedImageError
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import Config
from .database import Database
from .model import CLASS_LABELS, TRANSFORM, load_model

Config.validate()

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / "dist"

app = Flask(__name__)
app.config.update(
    JWT_SECRET_KEY=Config.JWT_SECRET_KEY,
    JWT_ACCESS_TOKEN_EXPIRES=Config.JWT_ACCESS_TOKEN_EXPIRES,
    MAX_CONTENT_LENGTH=Config.MAX_UPLOAD_MB * 1024 * 1024,
    RATELIMIT_STORAGE_URI=Config.RATELIMIT_STORAGE_URI,
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

if Config.CORS_ORIGINS:
    CORS(app, origins=Config.CORS_ORIGINS, allow_headers=["Content-Type", "Authorization"])
else:
    CORS(app, allow_headers=["Content-Type", "Authorization"])

jwt = JWTManager(app)
bcrypt = Bcrypt(app)
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["120 per minute"])

database = Database(Config.MONGO_URI, Config.DB_NAME)
model = load_model(Config.MODEL_PATH, Config.MODEL_URL, Config.MODEL_SHA256)

DEFAULT_HOSPITALS = [
    {"name": "Apollo Hospitals", "specialization": "Dermatology & Skin Care", "address": "Jubilee Hills, Hyderabad", "phone": "+91-40-23607777", "email": "info@apollohospitals.com", "rating": 4.8, "available": True, "timings": "24/7", "city": "Hyderabad", "lat": 17.4265, "lon": 78.4120},
    {"name": "AIIMS Delhi", "specialization": "Multi-Specialty Hospital", "address": "Ansari Nagar, New Delhi", "phone": "+91-11-26588500", "email": "director@aiims.edu", "rating": 4.9, "available": True, "timings": "24/7", "city": "Delhi", "lat": 28.5672, "lon": 77.2100},
    {"name": "Fortis Hospital", "specialization": "Dermatology & Cosmetic", "address": "Bannerghatta Road, Bangalore", "phone": "+91-80-66214444", "email": "enquiry@fortishealthcare.com", "rating": 4.7, "available": True, "timings": "8:00 AM - 10:00 PM", "city": "Bangalore", "lat": 12.8946, "lon": 77.5976},
    {"name": "Max Super Specialty Hospital", "specialization": "Skin & Allergy", "address": "Saket, New Delhi", "phone": "+91-11-26515050", "email": "info@maxhealthcare.com", "rating": 4.6, "available": True, "timings": "24/7", "city": "Delhi", "lat": 28.5276, "lon": 77.2107},
    {"name": "Manipal Hospital", "specialization": "Dermatology", "address": "HAL Airport Road, Bangalore", "phone": "+91-80-25024444", "email": "info@manipalhospitals.com", "rating": 4.5, "available": True, "timings": "8:00 AM - 9:00 PM", "city": "Bangalore", "lat": 12.9610, "lon": 77.6482},
]

TIPS = {
    "NORMAL_SKIN": ["Maintain good hygiene", "Use sunscreen", "Stay hydrated"],
    "PSORIASIS": ["Keep skin moisturized", "Avoid known triggers", "Consult a dermatologist"],
    "Ringworm": ["Keep the area clean and dry", "Avoid sharing towels", "Consult a healthcare professional"],
    "acne": ["Cleanse gently", "Use non-comedogenic products", "Avoid picking at lesions"],
}

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")


def error(message: str, status: int):
    return jsonify({"error": message}), status


def parse_limit(default: int = 10, maximum: int = 100) -> int:
    value = request.args.get("limit", default, type=int)
    return max(1, min(value, maximum))


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def serialize_prediction(doc):
    doc.pop("_id", None)
    if isinstance(doc.get("timestamp"), datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()
    return doc


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(self)"
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.errorhandler(413)
def payload_too_large(_exc):
    return error(f"Uploaded payload exceeds the {Config.MAX_UPLOAD_MB} MB limit.", 413)


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    app.logger.exception("Unhandled server error")
    return error("An internal server error occurred.", 500)


@app.get("/health")
def health():
    database.client.admin.command("ping")
    return jsonify({"status": "ok", "service": "dermaai", "model_loaded": model is not None}), 200


@app.get("/api")
def api_info():
    return jsonify({"name": "DermaAI API", "version": "1.0.0", "status": "ok"})


@app.post("/api/auth/register")
@limiter.limit("5 per hour")
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = data.get("password", "")
    if not USERNAME_RE.fullmatch(username):
        return error("Username must be 3-64 characters and contain only letters, numbers, _, ., or -.", 400)
    if not isinstance(password, str) or len(password) < 12 or len(password) > 128:
        return error("Password must contain 12-128 characters.", 400)
    if database.users.find_one({"username": username}):
        return error("User already exists.", 409)
    database.users.insert_one({
        "username": username,
        "password": bcrypt.generate_password_hash(password).decode("utf-8"),
        "created_at": datetime.now(timezone.utc),
    })
    return jsonify({"message": "User registered successfully."}), 201


@app.post("/api/auth/login")
@limiter.limit("10 per minute")
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = data.get("password", "")
    user = database.users.find_one({"username": username})
    if not user or not isinstance(password, str) or not bcrypt.check_password_hash(user["password"], password):
        return error("Invalid credentials.", 401)
    return jsonify({"token": create_access_token(identity=username), "username": username})


@app.get("/api/user/profile")
@jwt_required()
def profile():
    return jsonify({"username": get_jwt_identity()})


@app.get("/api/dashboard/stats")
@jwt_required()
def dashboard_stats():
    username = get_jwt_identity()
    scope = {"username": username}
    return jsonify({
        "total": database.predictions.count_documents(scope),
        "normal": database.predictions.count_documents({**scope, "prediction": "NORMAL_SKIN"}),
        "psoriasis": database.predictions.count_documents({**scope, "prediction": "PSORIASIS"}),
        "ringworm": database.predictions.count_documents({**scope, "prediction": "Ringworm"}),
        "acne": database.predictions.count_documents({**scope, "prediction": "acne"}),
        "patients_count": database.patients.count_documents({"username": username}),
        "hospitals_count": database.hospitals.count_documents({}),
    })


@app.get("/api/dashboard/recent")
@jwt_required()
def dashboard_recent():
    username = get_jwt_identity()
    predictions = list(
        database.predictions.find({"username": username}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(parse_limit())
    )
    return jsonify([serialize_prediction(p) for p in predictions])


@app.post("/api/predict")
@jwt_required()
@limiter.limit("30 per minute")
def predict():
    if "image" not in request.files:
        return error("No image provided.", 400)
    file = request.files["image"]
    if not file.filename:
        return error("No image selected.", 400)
    try:
        image = Image.open(file.stream).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError):
        return error("The uploaded file is not a valid image.", 400)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = float(probabilities[predicted_index].item())
    uncertain = confidence < Config.DISEASE_CONFIDENCE_THRESHOLD
    all_probabilities = [
        {"label": label, "probability": round(float(probabilities[i].item()) * 100, 2)}
        for i, label in enumerate(CLASS_LABELS)
    ]
    all_probabilities.sort(key=lambda item: item["probability"], reverse=True)

    username = get_jwt_identity()
    patient_name = request.form.get("patientName", "").strip()[:100]
    patient_age = request.form.get("patientAge", "").strip()
    patient_phone = request.form.get("patientPhone", "").strip()[:30]
    if patient_age:
        try:
            age = int(patient_age)
            if not 0 <= age <= 120:
                raise ValueError
        except ValueError:
            return error("Patient age must be a number between 0 and 120.", 400)

    now = datetime.now(timezone.utc)
    patient_info = {"name": patient_name, "age": patient_age, "phone": patient_phone}
    database.predictions.insert_one({
        "username": username,
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "uncertain": uncertain,
        "all_probabilities": all_probabilities,
        "timestamp": now,
        "patient": patient_info,
    })

    if patient_name or patient_phone:
        query = {"username": username}
        query["phone" if patient_phone else "name"] = patient_phone or patient_name
        database.patients.update_one(
            query,
            {"$set": {"username": username, **patient_info, "last_visit": now}, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "uncertain": uncertain,
        "message": "Low-confidence result; consult a qualified clinician." if uncertain else "Prediction generated.",
        "tips": TIPS.get(predicted_class, ["Consult a qualified healthcare professional."]),
        "probabilities": all_probabilities,
    })


@app.get("/api/hospitals/nearby")
@jwt_required()
def nearby_hospitals():
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (KeyError, TypeError, ValueError):
        return error("Valid latitude and longitude are required.", 400)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return error("Latitude or longitude is outside the valid range.", 400)

    hospitals = list(database.hospitals.find({}, {"_id": 0})) or DEFAULT_HOSPITALS
    for hospital in hospitals:
        if hospital.get("lat") is not None and hospital.get("lon") is not None:
            hospital["distance"] = round(haversine(lat, lon, hospital["lat"], hospital["lon"]), 2)
        else:
            hospital["distance"] = None
    hospitals.sort(key=lambda item: item["distance"] if item["distance"] is not None else float("inf"))
    return jsonify(hospitals[:25])


@app.get("/api/hospitals/search-city")
@jwt_required()
def search_hospitals_by_city():
    city = request.args.get("city", "").strip()
    if not city or len(city) > 100 or any(char in city for char in "$[]{}()"):
        return error("A valid city name is required.", 400)
    hospitals = list(database.hospitals.find({"city": {"$regex": f"^{re.escape(city)}$", "$options": "i"}}, {"_id": 0}))
    if not hospitals:
        hospitals = [h for h in DEFAULT_HOSPITALS if h["city"].lower() == city.lower()]
    return jsonify({"city": city, "hospitals": hospitals[:25]})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def frontend(path):
    if path.startswith("api/") or path == "health":
        return error("Not found.", 404)
    if FRONTEND_DIST.is_dir():
        requested = FRONTEND_DIST / path
        if path and requested.is_file():
            return send_from_directory(FRONTEND_DIST, path)
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"message": "DermaAI API is running. Frontend build is not installed."}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
