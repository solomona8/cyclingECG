# Build iOS App and Deploy to Render

Quick guide to rebuild the iOS app with the new UI improvements and deploy the backend to Render.

---

## Part 1: Rebuild iOS App in Xcode

### Step 1: Open the Project

On your Mac, navigate to the project and open Xcode:

```bash
cd ~/cyclingECG/iOS
open ECGHealthKit.xcodeproj
```

Or open Xcode and use File > Open > navigate to `cyclingECG/iOS/ECGHealthKit.xcodeproj`

### Step 2: Clean Build Folder (Optional but Recommended)

In Xcode menu:
```
Product > Clean Build Folder
```
Or press: `Shift + Command + K`

### Step 3: Connect Your iPhone

1. Plug your iPhone into your Mac via USB cable
2. Unlock your iPhone
3. If prompted "Trust This Computer?", tap "Trust"
4. In Xcode, select your iPhone from the device dropdown (top left, next to the Play/Stop buttons)

### Step 4: Build and Run

**Option A: GUI Method**
1. Click the Play button (▶️) in the top left
2. Or go to: Product > Run
3. Or press: `Command + R`

**Option B: Command Line Method**

```bash
cd iOS

# Build only (don't install)
xcodebuild -scheme ECGHealthKit -configuration Debug clean build

# Build and install to connected device
xcodebuild -scheme ECGHealthKit \
  -destination 'platform=iOS,name=YOUR_IPHONE_NAME' \
  clean build

# To see connected devices:
xcrun xctrace list devices
```

### Step 5: Verify Installation

1. App should install and launch on your iPhone
2. Grant HealthKit permissions if prompted
3. Navigate to an ECG recording
4. Tap "Analyze Now"
5. Check the new table layout with:
   - Header row: "Metric | Current | 30 Day Avg"
   - Centered values under headers
   - Clean, aligned columns

### What's New in This Build

✅ Header row with column labels
✅ Current values moved left and centered
✅ 30-day averages in dedicated column (same font size)
✅ No more "30d avg" label on each row
✅ Professional table layout
✅ Better readability and alignment

---

## Part 2: Deploy Backend to Render

### Current Changes to Deploy

- ✅ Database initialization (ecg_data.db)
- ✅ 30-day trend tracking
- ✅ Automatic storage of all analyses
- ✅ Outlier detection (>2 SD from mean)

### Option A: Automatic Deployment (If GitHub Connected)

If your Render service is connected to your GitHub repo:

1. **Push changes** (already done):
   ```bash
   git status  # Verify all changes committed
   ```

2. **Render auto-deploys** from your branch:
   - Go to: https://dashboard.render.com
   - Select your `cyclingecg` service
   - Check "Events" tab for deployment status
   - Wait 2-3 minutes for build to complete

### Option B: Manual Deployment

1. **Go to Render Dashboard**:
   ```
   https://dashboard.render.com
   ```

2. **Select your service**: `cyclingecg`

3. **Manual Deploy**:
   - Click "Manual Deploy" button (top right)
   - Select "Deploy latest commit" or your branch name
   - Click "Deploy"

4. **Monitor deployment**:
   - Watch the "Logs" tab for build progress
   - Look for: "Build succeeded" and "Live"
   - Deployment typically takes 2-3 minutes

### Step 6: Verify Deployment

Once deployed, test the endpoints:

```bash
# Health check
curl https://cyclingecg.onrender.com/health
# Expected: {"status":"ok"}

# Test data generation
curl 'https://cyclingecg.onrender.com/v1/ecg/generate_test_data?duration_sec=5&sampling_rate_hz=512&heart_rate_bpm=70'

# Interactive docs
open https://cyclingecg.onrender.com/docs
```

### Step 7: Update iOS App to Use Render (If Needed)

If you want the iOS app to use the cloud backend instead of local:

**Edit**: `iOS/ECGHealthKit/ECGAnalysisService.swift` line 18

The default is already set to Render:
```swift
init(baseURL: String = "https://cyclingecg.onrender.com", apiKey: String? = nil)
```

If you changed it to local (`http://192.168.x.x:8000`), change it back to:
```swift
init(baseURL: String = "https://cyclingecg.onrender.com", apiKey: String? = nil)
```

Then rebuild the app (repeat Part 1).

---

## Part 3: Configure Render Environment (Optional)

### Database Persistence

**Important**: Render's free tier has ephemeral storage. The SQLite database will be reset on each deployment.

**Options to persist data**:

1. **Upgrade to paid tier** with persistent disk
2. **Use PostgreSQL** (Render provides free PostgreSQL)
3. **Export data regularly** before deployments

### PostgreSQL Setup (Recommended for Production)

1. **Create PostgreSQL database** in Render:
   - Dashboard > New > PostgreSQL
   - Name: `cyclingecg-db`
   - Free tier is fine for testing

2. **Get connection string**:
   - Copy "Internal Database URL"
   - Example: `postgresql://user:pass@host/db`

3. **Add to Render service**:
   - Go to your `cyclingecg` service
   - Environment tab
   - Add variable:
     - Key: `DATABASE_URL`
     - Value: `<your-postgresql-url>`

4. **Update app code** (already supports this):
   ```python
   # app/database.py line 58
   DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./ecg_data.db")
   ```

   The app automatically uses PostgreSQL if `DATABASE_URL` is set!

### Environment Variables

Recommended variables for Render:

```bash
# Required
PYTHON_VERSION = 3.11.0

# Optional but recommended
API_KEY = <your-secret-key>  # Enable authentication
DATABASE_URL = <postgresql-url>  # Use PostgreSQL instead of SQLite

# Optional for AI features
OPENAI_API_KEY = <your-openai-key>  # Enable AI narratives
OPENAI_MODEL = gpt-4  # Or gpt-3.5-turbo for cheaper option
```

---

## Part 4: Testing the Complete Workflow

Once both are deployed:

### 1. Test Backend

```bash
# Generate test data
curl -X GET 'https://cyclingecg.onrender.com/v1/ecg/generate_test_data?duration_sec=10&sampling_rate_hz=512&heart_rate_bpm=72'

# Analyze test data
curl -X POST 'https://cyclingecg.onrender.com/v1/ecg/analyze' \
  -H 'Content-Type: application/json' \
  -d '{
    "recording_id": "test-123",
    "samples": [0.1, 0.2, 0.15, ...],
    "sampling_rate_hz": 512,
    "units": "mV",
    "lead": "I",
    "start_timestamp_utc": "2024-12-10T19:30:00Z",
    "device_info": {
      "manufacturer": "Test",
      "model": "Test Device"
    }
  }'
```

### 2. Test iOS App

1. Open ECG HealthKit app on iPhone
2. Select an ECG recording
3. Tap "Analyze Now"
4. Verify:
   - ✅ Analysis completes successfully
   - ✅ Table has header row
   - ✅ Values are properly aligned
   - ✅ 30-day averages appear (if you have previous data)
   - ✅ Outliers marked in red (if applicable)

### 3. Verify Database Storage

After analyzing an ECG:

```bash
# Check backend logs in Render
# Should see:
# [ANALYZE] Saved analysis to database for <recording-id>
# [ANALYZE] Calculated 30-day stats for <recording-id>

# Query database (if using local backend)
sqlite3 ecg_data.db "SELECT recording_id, rhythm_classification, hr_mean FROM ecg_analyses;"
```

---

## Troubleshooting

### Xcode Build Errors

**"No devices connected"**
- Connect iPhone via USB
- Unlock iPhone and trust computer

**"Code signing error"**
- Select your development team in Xcode
- Signing & Capabilities tab

**"HealthKit entitlement error"**
- Ensure you're using a valid Apple ID with HealthKit access
- Check entitlements file exists

### Render Deployment Issues

**"Build failed"**
- Check Render logs for specific error
- Verify requirements.txt is up to date
- Ensure Python 3.11.0 environment variable is set

**"Service is offline"**
- Free tier sleeps after 15 minutes inactivity
- First request takes 30-50 seconds to wake up
- This is normal for free tier

**"Database reset after deployment"**
- Expected on free tier (ephemeral storage)
- Use PostgreSQL for persistence
- Or upgrade to paid tier with persistent disk

### iOS App Connection Issues

**"Connection refused"**
- For local backend: Verify correct IP address
- For Render: Check service is awake (visit health endpoint)
- Ensure iPhone and Mac on same network (for local)

**"Analysis timeout"**
- Render free tier takes 30-50s on first request
- Increase timeout in iOS app if needed
- Or use local backend for faster testing

---

## Summary

**What you're deploying:**

### iOS App:
- ✅ Improved analysis results table layout
- ✅ Header row with clear column labels
- ✅ Better alignment and readability
- ✅ Same functionality, better UI

### Backend to Render:
- ✅ Database storage of all analyses
- ✅ 30-day trend tracking
- ✅ Outlier detection
- ✅ Automatic stats calculation

**Next Steps:**
1. Rebuild iOS app in Xcode (Part 1)
2. Deploy backend to Render (Part 2)
3. Test complete workflow (Part 4)

Good luck! 🚀
