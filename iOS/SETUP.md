# Quick Xcode Setup Guide

## Getting Started

The iOS app now includes a complete Xcode project! No need to manually create one.

### 1. Open the Project

```bash
cd cyclingECG/iOS
open ECGHealthKit.xcodeproj
```

Or in Xcode:
- File → Open
- Navigate to `iOS/ECGHealthKit.xcodeproj`
- Click Open

### 2. Configure Signing

In Xcode:
1. Select the project in the navigator
2. Select the "ECGHealthKit" target
3. Go to "Signing & Capabilities" tab
4. Select your development team from the dropdown

The project already includes:
- ✅ All Swift source files
- ✅ HealthKit capability configured
- ✅ Info.plist with required permissions
- ✅ Entitlements file
- ✅ Asset catalog

## Testing Requirements

- Must use a REAL iPhone (not Simulator)
- iPhone must be paired with Apple Watch
- Take at least one ECG on Apple Watch first
- For backend connection, use local IP address (e.g., 192.168.1.100:8000)

### 3. Build and Run

1. Connect your iPhone via USB
2. Select your iPhone from the device dropdown in Xcode
3. Click the Play button (⌘R) to build and run

## Backend Setup (Optional)

To enable ECG analysis features, start the backend:

```bash
cd cyclingECG
uvicorn app.main:app --reload --host 0.0.0.0
```

Then in the app's Settings (⚙️ icon):
- API URL: http://YOUR_LOCAL_IP:8000
- API Key: (if you set API_KEY env var)

Find your local IP:
```bash
# macOS
ifconfig | grep "inet " | grep -v 127.0.0.1

# Linux
ip addr show | grep "inet " | grep -v 127.0.0.1
```
