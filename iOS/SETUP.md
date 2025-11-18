# Quick Xcode Setup Guide

## Files You Need

All files are in: `/home/user/cyclingECG/iOS/ECGHealthKit/`

### Swift Source Files (Add to Xcode):
- ECGHealthKitApp.swift
- ContentView.swift
- HealthKitManager.swift
- Models.swift
- ECGAnalysisService.swift
- ExportManager.swift
- ECGListView.swift
- ECGDetailView.swift

### Configuration Files:
- Info.plist (merge with yours or use as reference)
- ECGHealthKit.entitlements (add to project)

## Minimum Info.plist Additions

Add these two keys to your Info.plist:

```xml
<key>NSHealthShareUsageDescription</key>
<string>This app needs access to your ECG recordings from Apple Watch to analyze and export them.</string>

<key>UIRequiredDeviceCapabilities</key>
<array>
    <string>healthkit</string>
</array>
```

## Required Capability

In Xcode:
1. Select project → Target → Signing & Capabilities
2. Click "+ Capability"
3. Add "HealthKit"

## Testing Requirements

- Must use a REAL iPhone (not Simulator)
- iPhone must be paired with Apple Watch
- Take at least one ECG on Apple Watch first
- For backend connection, use local IP address (e.g., 192.168.1.100:8000)

## Backend Setup

Start your backend:
```bash
cd /home/user/cyclingECG
uvicorn app.main:app --reload --host 0.0.0.0
```

In app Settings:
- API URL: http://YOUR_LOCAL_IP:8000
- API Key: (if you set API_KEY env var)
