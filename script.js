// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const changeImageBtn = document.getElementById('changeImageBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const predictionResult = document.getElementById('predictionResult');
const confidenceBadge = document.getElementById('confidenceBadge');
const probabilitiesList = document.getElementById('probabilitiesList');
const tipsList = document.getElementById('tipsList');

// Camera elements
const cameraBox = document.getElementById('cameraBox');
const cameraModal = document.getElementById('cameraModal');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const flashBtn = document.getElementById('flashBtn');

let selectedFile = null;
let cameraStream = null;
let currentFacingMode = 'environment'; // 'user' for front, 'environment' for back
let flashEnabled = false;

// Detect if device is mobile
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

// Upload box click
uploadBox.addEventListener('click', () => {
    imageInput.click();
});

// Drag and drop (desktop only)
if (!isMobile) {
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#764ba2';
        uploadBox.style.background = '#f0f2ff';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = '#667eea';
        uploadBox.style.background = '#f8f9ff';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#667eea';
        uploadBox.style.background = '#f8f9ff';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

// Touch feedback for mobile
if (isTouchDevice) {
    uploadBox.addEventListener('touchstart', () => {
        uploadBox.style.borderColor = '#764ba2';
        uploadBox.style.background = '#f0f2ff';
    });

    uploadBox.addEventListener('touchend', () => {
        uploadBox.style.borderColor = '#667eea';
        uploadBox.style.background = '#f8f9ff';
    });
}

// File input change
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Change image button
changeImageBtn.addEventListener('click', () => {
    imageInput.click();
});

// Analyze button
analyzeBtn.addEventListener('click', analyzeImage);

// Camera event listeners
cameraBox.addEventListener('click', openCamera);
closeCameraBtn.addEventListener('click', closeCamera);
captureBtn.addEventListener('click', capturePhoto);
switchCameraBtn.addEventListener('click', switchCamera);
flashBtn.addEventListener('click', toggleFlash);

// ===== CAMERA FUNCTIONS =====

// Open camera
async function openCamera() {
    try {
        cameraModal.style.display = 'flex';
        await startCamera();
    } catch (error) {
        console.error('Error opening camera:', error);
        closeCamera();
        if (error.name === 'NotAllowedError') {
            showAlert('Camera access denied. Please grant camera permissions in your browser settings.');
        } else if (error.name === 'NotFoundError') {
            showAlert('No camera found on this device.');
        } else {
            showAlert('Unable to access camera: ' + error.message);
        }
    }
}

// Start camera stream
async function startCamera() {
    // Stop existing stream if any
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            facingMode: currentFacingMode,
            width: { ideal: 1920 },
            height: { ideal: 1080 }
        },
        audio: false
    };

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = cameraStream;

        // Check if flash/torch is supported
        const track = cameraStream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();
        if (capabilities.torch) {
            flashBtn.style.opacity = '1';
        } else {
            flashBtn.style.opacity = '0.5';
        }
    } catch (error) {
        throw error;
    }
}

// Close camera
function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraModal.style.display = 'none';
    cameraVideo.srcObject = null;
}

// Switch between front and back camera
async function switchCamera() {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    try {
        await startCamera();
    } catch (error) {
        console.error('Error switching camera:', error);
        showAlert('Unable to switch camera. This device may only have one camera.');
        // Revert to previous mode
        currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    }
}

// Toggle flash/torch
async function toggleFlash() {
    if (!cameraStream) return;

    const track = cameraStream.getVideoTracks()[0];
    const capabilities = track.getCapabilities();

    if (!capabilities.torch) {
        showAlert('Flash is not supported on this device.');
        return;
    }

    try {
        flashEnabled = !flashEnabled;
        await track.applyConstraints({
            advanced: [{ torch: flashEnabled }]
        });
        flashBtn.style.background = flashEnabled
            ? 'rgba(255, 193, 7, 0.3)'
            : 'rgba(255, 255, 255, 0.1)';
    } catch (error) {
        console.error('Error toggling flash:', error);
        showAlert('Unable to toggle flash.');
    }
}

// Capture photo from camera
function capturePhoto() {
    if (!cameraStream) return;

    // Set canvas size to match video
    const videoWidth = cameraVideo.videoWidth;
    const videoHeight = cameraVideo.videoHeight;
    cameraCanvas.width = videoWidth;
    cameraCanvas.height = videoHeight;

    // Draw current video frame to canvas
    const context = cameraCanvas.getContext('2d');
    context.drawImage(cameraVideo, 0, 0, videoWidth, videoHeight);

    // Convert canvas to blob
    cameraCanvas.toBlob((blob) => {
        if (blob) {
            // Create a file from the blob
            const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            handleFileSelect(file);
            closeCamera();
        } else {
            showAlert('Failed to capture photo. Please try again.');
        }
    }, 'image/jpeg', 0.95);
}


// Handle file selection
function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showAlert('Please select an image file (JPG, PNG, etc.)');
        return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showAlert('Image file is too large. Please select an image smaller than 10MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadBox.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
        resultsSection.style.display = 'none';
    };
    reader.onerror = () => {
        showAlert('Error reading file. Please try again.');
    };
    reader.readAsDataURL(file);
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile) return;

    // Show loading state
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');
    btnText.textContent = 'Analyzing...';
    loader.style.display = 'inline-block';
    analyzeBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Add patient information if filled
        const patientName = document.getElementById('patientName');
        const patientAge = document.getElementById('patientAge');
        const patientGender = document.getElementById('patientGender');
        const patientPhone = document.getElementById('patientPhone');
        const patientEmail = document.getElementById('patientEmail');
        const patientCity = document.getElementById('patientCity');

        if (patientName && patientName.value) formData.append('patient_name', patientName.value);
        if (patientAge && patientAge.value) formData.append('patient_age', patientAge.value);
        if (patientGender && patientGender.value) formData.append('patient_gender', patientGender.value);
        if (patientPhone && patientPhone.value) formData.append('patient_phone', patientPhone.value);
        if (patientEmail && patientEmail.value) formData.append('patient_email', patientEmail.value);
        if (patientCity && patientCity.value) formData.append('patient_city', patientCity.value);

        // Send request with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showAlert('Error: ' + (data.error || 'Unknown error occurred'));
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            showAlert('Request timed out. Please try again with a smaller image.');
        } else if (error.message.includes('Failed to fetch')) {
            showAlert('Network error. Please check your connection and try again.');
        } else {
            showAlert('Error analyzing image: ' + error.message);
        }
    } finally {
        // Reset button
        btnText.textContent = 'Analyze Image';
        loader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';

    // Scroll to results with better mobile support
    setTimeout(() => {
        const yOffset = isMobile ? -20 : -50;
        const element = resultsSection;
        const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;

        window.scrollTo({ top: y, behavior: 'smooth' });
    }, 100);

    // Set prediction
    predictionResult.textContent = data.prediction.replace('_', ' ');
    confidenceBadge.textContent = data.confidence.toFixed(2) + '%';

    // Set confidence badge color based on confidence level
    if (data.confidence >= 80) {
        confidenceBadge.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
    } else if (data.confidence >= 60) {
        confidenceBadge.style.background = 'linear-gradient(135deg, #ffc107 0%, #ff9800 100%)';
    } else {
        confidenceBadge.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
    }

    // Display all probabilities
    probabilitiesList.innerHTML = '';
    data.all_probabilities.forEach((item, index) => {
        const probBar = document.createElement('div');
        probBar.className = 'probability-bar';

        const isTopPrediction = index === 0;
        const barColor = isTopPrediction
            ? 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)'
            : 'linear-gradient(90deg, #e0e0e0 0%, #c0c0c0 100%)';

        probBar.innerHTML = `
            <div class="probability-label">
                <span>${item.label.replace('_', ' ')}</span>
                <span>${item.probability.toFixed(2)}%</span>
            </div>
            <div class="probability-progress">
                <div class="probability-fill" style="width: 0%; background: ${barColor};">
                </div>
            </div>
        `;

        probabilitiesList.appendChild(probBar);

        // Animate the bar
        setTimeout(() => {
            const fill = probBar.querySelector('.probability-fill');
            fill.style.width = item.probability + '%';
            if (item.probability > 10) {
                fill.textContent = item.probability.toFixed(1) + '%';
            }
        }, 100 * index);
    });

    // Display tips
    tipsList.innerHTML = '';
    data.tips.forEach(tip => {
        const li = document.createElement('li');
        li.textContent = tip;
        tipsList.appendChild(li);
    });
}

// Custom alert function for better mobile UX
function showAlert(message) {
    // Use native alert for now, but could be replaced with custom modal
    alert(message);
}

// Prevent zoom on double tap for iOS (optional)
let lastTouchEnd = 0;
document.addEventListener('touchend', (event) => {
    const now = Date.now();
    if (now - lastTouchEnd <= 300) {
        event.preventDefault();
    }
    lastTouchEnd = now;
}, { passive: false });

// Handle orientation change
window.addEventListener('orientationchange', () => {
    // Slight delay to allow browser to adjust
    setTimeout(() => {
        window.scrollTo(0, window.scrollY);
    }, 100);
});

// Prevent pull-to-refresh on mobile (optional, can be removed if needed)
document.body.addEventListener('touchmove', (e) => {
    if (e.target === document.body) {
        e.preventDefault();
    }
}, { passive: false });

// Add some helpful tips on page load
window.addEventListener('load', () => {
    console.log('Skin Disease Classifier loaded successfully!');
    console.log('Device type:', isMobile ? 'Mobile' : 'Desktop');
    console.log('Touch support:', isTouchDevice ? 'Yes' : 'No');
    console.log('Upload an image to get started.');

    // Preload image preview for better performance
    const img = new Image();
    img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
});

// Service Worker registration for PWA support (optional)
if ('serviceWorker' in navigator && window.location.protocol === 'https:') {
    window.addEventListener('load', () => {
        // Uncomment to enable service worker
        // navigator.serviceWorker.register('/sw.js').catch(() => {});
    });
}

// Clean up camera stream when page is hidden or closed
document.addEventListener('visibilitychange', () => {
    if (document.hidden && cameraStream) {
        closeCamera();
    }
});

window.addEventListener('beforeunload', () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
});

// Close camera modal when clicking outside
cameraModal.addEventListener('click', (e) => {
    if (e.target === cameraModal) {
        closeCamera();
    }
});

// ===== LOCATION-BASED HEALTHCARE FINDER =====

// Elements
const detectLocationBtn = document.getElementById('detectLocationBtn');
const refreshLocationBtn = document.getElementById('refreshLocationBtn');
const detectedLocation = document.getElementById('detectedLocation');
const locationName = document.getElementById('locationName');
const doctorsSection = document.getElementById('doctorsSection');
const doctorsGrid = document.getElementById('doctorsGrid');
const hospitalsSection = document.getElementById('hospitalsSection');
const hospitalsList = document.getElementById('hospitalsList');
const locationPlaceholder = document.getElementById('locationPlaceholder');

// Store user coordinates
let userCoords = null;

// Event listeners
if (detectLocationBtn) {
    detectLocationBtn.addEventListener('click', detectUserLocation);
}

if (refreshLocationBtn) {
    refreshLocationBtn.addEventListener('click', detectUserLocation);
}

// Main function to detect location and load healthcare data from API
async function detectUserLocation() {
    const btnIcon = detectLocationBtn.querySelector('.btn-icon');
    const btnText = detectLocationBtn.querySelector('.btn-text');
    const loader = detectLocationBtn.querySelector('.location-loader');

    // Show loading state
    btnIcon.style.display = 'none';
    btnText.textContent = 'Detecting...';
    loader.style.display = 'inline-block';
    detectLocationBtn.disabled = true;

    try {
        // Check if geolocation is supported
        if (!navigator.geolocation) {
            throw new Error('Geolocation is not supported by your browser');
        }

        // Get user's current position
        const position = await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject, {
                enableHighAccuracy: true,
                timeout: 15000,
                maximumAge: 300000
            });
        });

        userCoords = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
        };

        console.log('GPS Coordinates:', userCoords.latitude, userCoords.longitude);

        // Get city name using reverse geocoding
        const cityData = await getCityFromCoordinates(userCoords.latitude, userCoords.longitude);

        // Show detected location
        detectedLocation.style.display = 'flex';
        locationName.textContent = cityData.displayName;

        // Fetch healthcare data from Python backend API
        await fetchHealthcareFromAPI(userCoords);

        // Update button
        btnIcon.style.display = 'inline';
        btnIcon.textContent = '✓';
        btnText.textContent = 'Location Detected';

    } catch (error) {
        console.error('Error detecting location:', error);

        let errorMessage = 'Unable to detect location';
        if (error.code === 1) {
            errorMessage = 'Location access denied. Please enable location permissions.';
        } else if (error.code === 2) {
            errorMessage = 'Location unavailable. Please try again.';
        } else if (error.code === 3) {
            errorMessage = 'Location request timed out. Please try again.';
        }

        showAlert(errorMessage);

        // Reset button
        btnIcon.style.display = 'inline';
        btnIcon.textContent = '🔍';
        btnText.textContent = 'Try Again';
    } finally {
        loader.style.display = 'none';
        detectLocationBtn.disabled = false;
    }
}

// Fetch healthcare data from Python backend API
async function fetchHealthcareFromAPI(coords) {
    try {
        // Show loading state in hospitals section
        locationPlaceholder.style.display = 'none';
        hospitalsSection.style.display = 'block';
        hospitalsList.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px;">
                <div class="loader" style="display: inline-block; margin-bottom: 10px;"></div>
                <p style="color: #666;">Searching for nearby healthcare facilities...</p>
            </div>
        `;

        // Call the Python backend API
        const response = await fetch(`/api/hospitals/nearby?lat=${coords.latitude}&lon=${coords.longitude}`);

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const hospitals = await response.json();

        console.log('API Response:', hospitals);

        if (hospitals.error) {
            throw new Error(hospitals.error);
        }

        if (hospitals.length === 0) {
            showNoDataMessage(coords);
            return;
        }

        // Display the hospitals from API
        displayHospitalsFromAPI(hospitals, coords);

        // Hide doctors section as we're using API data now
        doctorsSection.style.display = 'none';

    } catch (error) {
        console.error('Error fetching from API:', error);

        // Show fallback message
        hospitalsList.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px;">
                <div style="font-size: 3rem; margin-bottom: 15px;">⚠️</div>
                <h4 style="color: #dc3545; margin-bottom: 10px;">Unable to fetch nearby hospitals</h4>
                <p style="color: #666; margin-bottom: 20px;">${error.message}</p>
                <a href="https://www.google.com/maps/search/hospital+near+me/@${coords.latitude},${coords.longitude},14z" 
                   target="_blank" 
                   class="btn-hospital-directions" 
                   style="display: inline-flex; padding: 12px 25px; font-size: 1rem;">
                    <span>🗺️</span> Find Hospitals on Google Maps
                </a>
            </div>
        `;
    }
}

// Get city name from coordinates using reverse geocoding
async function getCityFromCoordinates(lat, lng) {
    try {
        // Using OpenStreetMap Nominatim API (free, no API key required)
        const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=10`,
            {
                headers: {
                    'Accept-Language': 'en'
                }
            }
        );

        const data = await response.json();

        // Extract city/town name
        const city = data.address.city ||
            data.address.town ||
            data.address.village ||
            data.address.county ||
            data.address.state_district ||
            'Unknown';

        const state = data.address.state || '';

        return {
            displayName: `${city}, ${state}`,
            rawCity: city
        };

    } catch (error) {
        console.error('Reverse geocoding failed:', error);
        return { displayName: 'Your Location', rawCity: 'Unknown' };
    }
}

// Calculate distance between two coordinates (in km)
function calculateDistance(lat1, lon1, lat2, lon2) {
    if (!lat1 || !lon1 || !lat2 || !lon2) return 0;
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

// Display hospitals from API response
function displayHospitalsFromAPI(hospitals, coords) {
    hospitalsList.innerHTML = '';

    // Calculate distances and sort by distance
    const hospitalsWithDist = hospitals.map(h => {
        const dist = calculateDistance(
            coords.latitude, coords.longitude,
            parseFloat(h.lat), parseFloat(h.lon)
        );
        return { ...h, calculatedDist: dist };
    }).sort((a, b) => {
        // Sort dermatology first, then by distance
        if (a.is_dermatology && !b.is_dermatology) return -1;
        if (!a.is_dermatology && b.is_dermatology) return 1;
        return a.calculatedDist - b.calculatedDist;
    });

    hospitalsWithDist.forEach((hospital, index) => {
        const card = document.createElement('div');
        card.className = 'hospital-card';
        card.style.animationDelay = `${index * 0.05}s`;

        // Create Google Maps URL
        const mapsUrl = `https://www.google.com/maps/search/${encodeURIComponent(hospital.name)}/@${hospital.lat},${hospital.lon},16z`;

        // Determine icon and styling based on type
        const isDerma = hospital.is_dermatology;
        const icon = isDerma ? '🩺' : '🏥';
        const typeLabel = hospital.type || 'Medical Facility';
        const distanceText = hospital.calculatedDist ? `${hospital.calculatedDist.toFixed(1)} km` : '';

        // Phone handling
        const phoneDisplay = hospital.phone || 'N/A';
        const phoneLink = hospital.phone ? `tel:${hospital.phone.replace(/\s/g, '')}` : '#';

        card.innerHTML = `
            <div class="hospital-header">
                <div class="hospital-icon" style="${isDerma ? 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);' : ''}">${icon}</div>
                <div class="hospital-title">
                    <h4>${hospital.name}</h4>
                    <p class="hospital-type">${typeLabel}</p>
                </div>
            </div>
            ${isDerma ? '<div style="background: linear-gradient(90deg, #667eea, #764ba2); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; margin-bottom: 10px;">✨ Dermatology Specialist</div>' : ''}
            <div class="hospital-details">
                <p><span>📍</span> ${hospital.address} ${distanceText ? `• ${distanceText}` : ''}</p>
                <p><span>📞</span> ${phoneDisplay}</p>
                ${hospital.website ? `<p><span>🌐</span> <a href="${hospital.website}" target="_blank" style="color: #667eea;">Website</a></p>` : ''}
            </div>
            <div class="hospital-actions">
                ${hospital.phone ? `
                    <a href="${phoneLink}" class="btn-hospital-call">
                        <span>📞</span> Call
                    </a>
                ` : ''}
                <a href="${mapsUrl}" target="_blank" class="btn-hospital-directions">
                    <span>🗺️</span> Directions
                </a>
            </div>
        `;

        hospitalsList.appendChild(card);
    });

    // Add summary note
    const dermaCount = hospitals.filter(h => h.is_dermatology).length;
    const note = document.createElement('div');
    note.className = 'hospitals-note';
    note.style.cssText = 'grid-column: 1 / -1; text-align: center; padding: 15px; color: #666; font-size: 0.9rem;';
    note.innerHTML = `
        <p>Found <strong>${hospitals.length}</strong> medical facilities nearby ${dermaCount > 0 ? `(${dermaCount} dermatology specialists)` : ''}</p>
        <p style="margin-top: 5px;">💡 Tip: Call ahead to confirm availability and book an appointment</p>
    `;
    hospitalsList.appendChild(note);
}

// Show message when no data is available for the location
function showNoDataMessage(coords) {
    hospitalsSection.style.display = 'block';
    hospitalsList.innerHTML = `
        <div class="placeholder-card" style="grid-column: 1 / -1; text-align: center; padding: 40px;">
            <div class="placeholder-icon" style="font-size: 4rem; margin-bottom: 20px;">🔍</div>
            <h4 style="color: #333; margin-bottom: 10px;">No Medical Facilities Found Nearby</h4>
            <p style="color: #666; margin-bottom: 20px;">We couldn't find hospitals in the immediate area. Try expanding your search:</p>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
                <a href="https://www.google.com/maps/search/dermatologist+near+me/@${coords.latitude},${coords.longitude},14z" 
                   target="_blank" 
                   class="btn-hospital-directions" 
                   style="display: inline-flex; padding: 12px 25px; font-size: 1rem;">
                    <span>🩺</span> Find Dermatologists
                </a>
                <a href="https://www.google.com/maps/search/hospital+near+me/@${coords.latitude},${coords.longitude},14z" 
                   target="_blank" 
                   class="btn-hospital-call" 
                   style="display: inline-flex; padding: 12px 25px; font-size: 1rem;">
                    <span>🏥</span> Find Hospitals
                </a>
            </div>
        </div>
    `;
}

// Log initialization
console.log('Location-based healthcare finder initialized - Using API');

// ===== PATIENT FORM TOGGLE =====
const togglePatientFormBtn = document.getElementById('togglePatientForm');
const patientFormContainer = document.getElementById('patientFormContainer');
const patientInfoToggle = document.getElementById('patientInfoToggle');

if (togglePatientFormBtn && patientFormContainer) {
    // Toggle on button click
    togglePatientFormBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        togglePatientForm();
    });

    // Toggle on section header click
    if (patientInfoToggle) {
        patientInfoToggle.addEventListener('click', togglePatientForm);
    }
}

function togglePatientForm() {
    const isVisible = patientFormContainer.style.display !== 'none';
    patientFormContainer.style.display = isVisible ? 'none' : 'block';

    if (togglePatientFormBtn) {
        togglePatientFormBtn.classList.toggle('open', !isVisible);
    }
}

console.log('Patient form toggle initialized');
