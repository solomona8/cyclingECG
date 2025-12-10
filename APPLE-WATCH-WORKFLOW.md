# Apple Watch ECG Extraction and Analysis Workflow

Complete guide for extracting ECG data from Apple Watch HealthKit, storing in database, and analyzing the recordings.

## System Overview

```
Apple Watch → HealthKit → iOS App → Backend API → Database → Analysis Results
```

## Prerequisites

### Hardware & Software
- ✅ Apple Watch Series 4+ (with ECG capability)
- ✅ iPhone with iOS 16.0+
- ✅ Paired Apple Watch with ECG recordings
- ✅ Xcode 14.0+ (for building iOS app)
- ✅ Mac computer (for iOS development)

### Backend Setup
- ✅ Python 3.11+ environment
- ✅ FastAPI backend running
- ✅ SQLite database initialized
- ✅ All dependencies installed

## Current Status

### ✅ Backend Server
- **Status**: Running on `http://localhost:8000`
- **Database**: `ecg_data.db` (16 KB, initialized)
- **Endpoints**: Health check, ECG analysis, CSV upload
- **Storage**: Automatic database storage with 30-day stats

### ✅ iOS App Configuration
- **Status**: Ready to build and deploy
- **Default Backend**: `https://cyclingecg.onrender.com`
- **Location**: `iOS/ECGHealthKit/`
- **Features**: HealthKit extraction, analysis, export (JSON/CSV/PDF/TXT)

---

## Step-by-Step Workflow

### Step 1: Build and Install iOS App

#### Option A: Using Xcode (Recommended)

1. **Open the project**:
   ```bash
   cd iOS
   open ECGHealthKit.xcodeproj
   ```

2. **Configure code signing**:
   - Select your development team in Xcode
   - Go to: Signing & Capabilities
   - Choose your Apple ID / Team

3. **Connect your iPhone**:
   - Plug in your iPhone via USB
   - Trust the computer if prompted
   - Select your iPhone as the build target

4. **Build and run**:
   - Press `⌘R` or click the Play button
   - Wait for the app to install and launch
   - **Note**: Simulator won't work - HealthKit requires a physical device

#### Option B: Using Command Line (Advanced)

```bash
cd iOS
xcodebuild -scheme ECGHealthKit -destination 'platform=iOS,name=YOUR_IPHONE_NAME' clean build
```

---

### Step 2: Configure Backend Connection

You have two options for the backend:

#### Option A: Use Local Backend (Current Setup)

The backend is running locally at `http://localhost:8000`. To use it from your iPhone:

1. **Find your computer's local IP**:
   ```bash
   # On Mac:
   ifconfig | grep "inet " | grep -v 127.0.0.1

   # You'll get something like: inet 192.168.1.100
   ```

2. **Update iOS app configuration**:

   Edit `iOS/ECGHealthKit/ECGAnalysisService.swift` line 18:
   ```swift
   // Change from:
   init(baseURL: String = "https://cyclingecg.onrender.com", apiKey: String? = nil)

   // To (use your IP):
   init(baseURL: String = "http://192.168.1.100:8000", apiKey: String? = nil)
   ```

3. **Rebuild the app** in Xcode

#### Option B: Use Cloud Backend (Default)

The app is already configured to use `https://cyclingecg.onrender.com`. This works out of the box but:
- First request takes 30-50 seconds (free tier cold start)
- Data is stored on remote server
- Requires internet connection

**To use cloud backend**: No changes needed, skip to Step 3!

---

### Step 3: Extract ECG Data from HealthKit

1. **Launch the app** on your iPhone

2. **Grant HealthKit permissions**:
   - Tap "Grant Access" button
   - In the system dialog, enable "Electrocardiograms (ECG)"
   - Tap "Allow"

3. **View your ECG recordings**:
   - The app automatically loads all ECG recordings
   - You'll see a list with:
     - Date and time
     - Classification (Sinus Rhythm, AFib, etc.)
     - Heart rate
     - Symptoms (if any)

4. **Verify data extraction**:

   Check Xcode console for logs:
   ```
   === EXTRACTING ECG DATA ===
   ECG ID: <UUID>
   Extracted 5120 voltage measurements
   Sampling frequency: 512.0 Hz
   === END EXTRACTION ===
   ```

   **Troubleshooting**:
   - If you see "0 voltage measurements", check HealthKit permissions
   - If "No ECG recordings found", take an ECG on your Apple Watch first
   - Go to Health app on iPhone to verify ECG data is synced

---

### Step 4: Analyze ECG Recordings

1. **Select an ECG recording** from the list

2. **Tap "Analyze Now"** button

3. **Wait for analysis** (5-10 seconds for local, 30-50s for cloud on first request)

4. **View results**:
   - Rhythm classification
   - Heart rate statistics (mean, min, max)
   - Heart rate variability (SDNN, RMSSD)
   - QRS duration
   - QT/QTc intervals
   - Signal quality assessment
   - 30-day comparison stats

---

### Step 5: Verify Database Storage

The backend automatically stores all analyses in the database.

#### Check Database Contents

```bash
# View database file
ls -lh ecg_data.db

# Query database (requires sqlite3)
sqlite3 ecg_data.db "SELECT recording_id, timestamp_utc, rhythm_classification, hr_mean FROM ecg_analyses;"

# Or use Python:
python3 << 'EOF'
from sqlalchemy import create_engine, text
engine = create_engine("sqlite:///ecg_data.db")
with engine.connect() as conn:
    results = conn.execute(text("SELECT * FROM ecg_analyses")).fetchall()
    print(f"Total analyses stored: {len(results)}")
    for row in results:
        print(f"  - {row.recording_id}: {row.rhythm_classification} @ {row.hr_mean} bpm")
EOF
```

#### Database Schema

The database stores:
- Recording ID (unique)
- Timestamp
- Rhythm metrics (classification, confidence)
- Heart rate metrics (mean, min, max)
- RR intervals
- HRV metrics (SDNN, RMSSD)
- Interval measurements (QRS, QT, QTc)
- Signal quality
- Ectopy burden
- Complete JSON analysis

---

### Step 6: Export ECG Data

The iOS app supports multiple export formats:

1. **Open an ECG recording** in detail view

2. **Tap "Export Recording"** button

3. **Choose export format**:

   - **JSON**: Complete structured data with all analysis
     - Best for: Developers, data scientists
     - Contains: All measurements + metadata

   - **CSV**: Voltage measurements with timestamps
     - Best for: Spreadsheet analysis, plotting
     - Format: `Timestamp (seconds),Voltage (microvolts)`
     - Includes metadata as comments

   - **PDF**: Formatted medical report
     - Best for: Healthcare providers
     - Includes: Summary, analysis results, narratives

   - **TXT**: Plain text summary
     - Best for: Easy reading, email
     - Human-readable format

4. **Share or save**:
   - Save to Files app
   - Email
   - AirDrop
   - Any iOS share extension

---

## Complete Workflow Example

### Scenario: Analyze and Export an ECG Recording

```
1. Take ECG on Apple Watch
   - Open ECG app on Watch
   - Follow on-screen instructions
   - 30-second recording
   - Data syncs to iPhone Health app

2. Open iOS App
   - Launch ECGHealthKit app
   - See new recording at top of list
   - Classification: "Sinus Rhythm"
   - Heart rate: 72 bpm

3. Analyze ECG
   - Tap on recording
   - Tap "Analyze Now"
   - Backend processes data
   - Results appear in ~5 seconds

4. Review Analysis
   - Rhythm: Sinus rhythm (95% confidence)
   - Heart Rate: 68-76 bpm (mean: 72)
   - HRV: SDNN 45ms, RMSSD 38ms
   - QRS: 90ms, QTc: 425ms
   - Signal Quality: Good
   - 30-day comparison: HR within normal range

5. Database Storage (Automatic)
   - Analysis saved to ecg_data.db
   - Recording ID: abc123-def456
   - Timestamp: 2024-12-10 19:30:00 UTC
   - Can query later for trends

6. Export Data
   - Tap "Export Recording"
   - Choose CSV format
   - Save to Files → "ECG Data" folder
   - File: ECG_abc123-def456.csv
   - Open in Excel/Numbers for visualization
```

---

## Backend API Reference

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
# Response: {"status":"ok"}
```

#### Analyze ECG
```bash
curl -X POST http://localhost:8000/v1/ecg/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test-123",
    "samples": [0.1, 0.2, ...],  # microvolts
    "sampling_rate_hz": 512,
    "units": "uV",
    "lead": "I",
    "start_timestamp_utc": "2024-12-10T19:30:00Z",
    "device_info": {
      "manufacturer": "Apple",
      "model": "Apple Watch"
    }
  }'
```

#### Generate Test Data
```bash
curl "http://localhost:8000/v1/ecg/generate_test_data?duration_sec=10&sampling_rate_hz=512&heart_rate_bpm=72"
```

#### Interactive Documentation
Open in browser: http://localhost:8000/docs

---

## Troubleshooting

### iOS App Issues

#### "HealthKit is not available"
- ✅ Make sure you're using a physical iPhone (not Simulator)
- ✅ Check HealthKit capability in Xcode signing
- ✅ Verify iOS 16.0+ on device

#### "No ECG Recordings Found"
- ✅ Take an ECG on Apple Watch first
- ✅ Ensure Watch and iPhone are paired and synced
- ✅ Open Health app → Browse → Heart → Electrocardiogram
- ✅ Verify recordings appear in Health app

#### "Authorization failed"
- ✅ Go to Settings → Privacy & Security → Health → ECG HealthKit
- ✅ Enable "Electrocardiograms" for reading
- ✅ Delete and reinstall app if needed

#### "No voltage measurements extracted"
- ✅ Check Xcode console for error messages
- ✅ Verify HealthKit permissions granted
- ✅ Try a different ECG recording
- ✅ Restart app and try again

### Backend Issues

#### "Connection refused"
- ✅ Verify backend is running: `curl http://localhost:8000/health`
- ✅ Check firewall isn't blocking port 8000
- ✅ For iPhone connection, use local IP not localhost

#### "Analysis failed"
- ✅ Check backend logs in terminal
- ✅ Verify ECG has at least 100 samples
- ✅ Check sample values are reasonable (not all zeros)
- ✅ Look for Python errors in console

#### Database not saving
- ✅ Check ecg_data.db file exists
- ✅ Verify write permissions
- ✅ Look for SQLAlchemy errors in logs
- ✅ Check disk space available

---

## Advanced Features

### 30-Day Trend Analysis

The backend automatically tracks ECG metrics over 30 days:

- **Average values**: Mean of last 30 days
- **Standard deviation**: Variability measure
- **Outlier detection**: Flags values >2 SD from mean

**Access in API response**:
```json
{
  "stats_30d": {
    "heart_rate_bpm": {
      "mean": {
        "avg_30d": 72.5,
        "std_30d": 4.2,
        "is_outlier": false
      }
    }
  }
}
```

### Batch Analysis

Analyze multiple ECGs programmatically:

```swift
// In iOS app
let results = await analysisService.analyzeMultipleECGs(recordings)
```

### CSV Upload

Upload ECG CSV files directly to backend:

```bash
curl -X POST http://localhost:8000/v1/ecg/upload_csv \
  -F "file=@ecg_data.csv" \
  -F "sampling_rate_hz=512" \
  -F "units=uV" \
  -F "lead=I"
```

---

## Data Privacy & Security

### Local Storage
- ✅ HealthKit data never leaves device without explicit user action
- ✅ Only analysis requests sent to backend
- ✅ Database stored locally on your machine
- ✅ No automatic cloud backup

### iOS App Permissions
- ✅ Read-only access to ECG data
- ✅ User controls all data sharing
- ✅ No background data collection
- ✅ Export requires explicit user action

### Backend Security
- ✅ Optional API key authentication
- ✅ No data retention beyond database
- ✅ Local-only by default
- ✅ HTTPS recommended for production

---

## Next Steps

### For Development

1. **Add more ECG recordings**: Take more ECGs on Apple Watch
2. **Build trend analysis**: Query database for historical patterns
3. **Export all data**: Batch export for external analysis
4. **Customize analysis**: Modify feature extraction parameters

### For Production

1. **Deploy backend**: Use Render, AWS, or other cloud platform
2. **Enable HTTPS**: Secure API connections
3. **Add authentication**: Implement API key system
4. **Configure notifications**: Alert on abnormal readings
5. **Add data backup**: Automated database backups

---

## File Locations

```
cyclingECG/
├── iOS/ECGHealthKit/              # iOS application
│   ├── HealthKitManager.swift     # ECG extraction from HealthKit
│   ├── ECGAnalysisService.swift   # Backend API client
│   ├── ExportManager.swift        # Export functionality
│   └── Models.swift               # Data models
├── app/
│   ├── main.py                    # FastAPI backend
│   ├── feature_extractor.py       # ECG analysis algorithms
│   └── database.py                # SQLAlchemy database
├── ecg_data.db                    # SQLite database (created at runtime)
└── APPLE-WATCH-WORKFLOW.md        # This guide
```

---

## Support & Resources

- **iOS App README**: `iOS/README.md`
- **Enhanced Analysis**: `ENHANCED_ANALYSIS.md`
- **Backend README**: `README.md`
- **Lovable Integration**: `LOVABLE-INTEGRATION.md`
- **Quick Start**: `QUICK-START.md`

---

## Summary

You now have a complete system for:

✅ **Extracting** ECG data from Apple Watch HealthKit
✅ **Storing** analyses in SQLite database
✅ **Analyzing** ECG recordings with advanced features
✅ **Tracking** 30-day trends and outliers
✅ **Exporting** data in multiple formats

**Current Status**:
- Backend: Running at `http://localhost:8000`
- Database: Initialized at `ecg_data.db`
- iOS App: Ready to build in Xcode

**Next Action**: Build and install the iOS app in Xcode, then start extracting and analyzing your ECG recordings!
