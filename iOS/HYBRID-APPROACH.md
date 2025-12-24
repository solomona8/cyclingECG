# Hybrid Approach for Apple Watch ECG Data Extraction

This document describes the hybrid approach implemented in the ECG Analyzer iOS app, providing flexibility for both backend connectivity and data export.

## Overview

The hybrid approach provides two key capabilities:

1. **Flexible Backend Configuration**: Seamlessly switch between local and cloud backends with automatic fallback
2. **Robust Export System**: Export ECG data in multiple formats (JSON, CSV, PDF, TXT) with proper error handling

## 1. Backend Flexibility

### Supported Backend Options

The app supports three backend configurations:

#### Cloud Backend (Default)
- **URL**: `https://cyclingecg.onrender.com`
- **Use Case**: Remote access, no local setup required
- **Pros**: Works from anywhere, no configuration needed
- **Cons**: Cold start delay (30-50 seconds on first request)
- **Best For**: Users who want quick setup without technical configuration

#### Local Network Backend
- **URL**: `http://192.168.1.100:8000` (configurable)
- **Use Case**: Fast local analysis when on same Wi-Fi network
- **Pros**: Faster response times, no cold start, data stays local
- **Cons**: Requires backend setup, must be on same network
- **Best For**: Development, testing, users with technical knowledge

#### Custom Backend
- **URL**: User-defined
- **Use Case**: Custom deployments, alternative hosting
- **Pros**: Complete control over backend location
- **Cons**: Requires manual URL configuration
- **Best For**: Advanced users, custom deployments

### Easy Backend Switching

Users can switch backends through the Settings screen:

1. Tap the gear icon ⚙️ in the app
2. Select backend type: Cloud, Local Network, or Custom
3. For Local Network: adjust IP address if needed
4. Tap "Save Settings" to apply

The app shows real-time backend status:
- 🟢 **Online**: Backend is reachable and healthy
- 🔴 **Offline**: Backend is not responding

### Automatic Fallback

The app implements intelligent fallback logic:

- **Local → Cloud**: If local backend fails, automatically tries cloud
- **Transparent**: Users see a note if fallback is used
- **Resilient**: Ensures analysis succeeds even if primary backend is down

**Example Flow**:
1. User selects Local Network backend
2. App tries local backend at `http://192.168.1.100:8000`
3. If unreachable (WiFi changed, backend stopped, etc.):
   - App automatically tries cloud backend
   - Shows message: "Note: Used fallback backend at https://cyclingecg.onrender.com"
4. Analysis completes successfully

### Setup Instructions for Local Backend

Detailed instructions are available in the app:
1. Go to Settings → Select "Local Network"
2. Tap "View Setup Instructions"
3. Follow step-by-step guide to:
   - Find computer's IP address
   - Start backend server
   - Configure firewall if needed

## 2. Export System

### Supported Export Formats

The app supports four export formats, each optimized for different use cases:

#### JSON Export
- **File**: `ECG_[recording-id].json`
- **Contents**:
  - Complete ECG recording data
  - All voltage measurements
  - Analysis results (if analyzed)
  - Metadata (device info, timestamps, etc.)
  - Export timestamp
- **Use Cases**:
  - Developer integration
  - Data science analysis
  - Programmatic processing
  - Complete data backup
- **Structure**:
  ```json
  {
    "recording": {
      "id": "...",
      "voltageMeasurements": [...],
      "samplingFrequency": 512.0,
      ...
    },
    "analysis": {
      "features": {...},
      "narrative": {...}
    },
    "exportDate": "2024-12-21T..."
  }
  ```

#### CSV Export
- **File**: `ECG_[recording-id].csv`
- **Contents**:
  - Time-series voltage data
  - Format: `Timestamp (seconds), Voltage (microvolts)`
  - Metadata in comment headers
  - Analysis summary (if available)
- **Use Cases**:
  - Spreadsheet analysis (Excel, Numbers, Google Sheets)
  - Plotting and visualization
  - Statistical analysis
  - MATLAB/Python processing
- **Example**:
  ```csv
  # ECG Recording Metadata
  # Recording ID: abc-123
  # Sampling Frequency: 512.00 Hz
  # Classification: Sinus Rhythm
  # Average Heart Rate: 72.0 bpm
  #
  Timestamp (seconds),Voltage (microvolts)
  0.000000,125.45
  0.001953,130.22
  ...
  ```

#### PDF Export
- **File**: `ECG_Report_[recording-id].pdf`
- **Contents**:
  - Professional formatted report
  - Recording information
  - Analysis results
  - Patient summary (if available)
  - Clinician notes (if available)
- **Use Cases**:
  - Sharing with healthcare providers
  - Medical records
  - Patient documentation
  - Professional presentation
- **Layout**:
  - Header with title
  - Recording Information section
  - Analysis Results section
  - Summary/Narrative sections
  - Footer with generation date

#### Text Export
- **File**: `ECG_Report_[recording-id].txt`
- **Contents**:
  - Plain text report
  - All recording details
  - Complete analysis results
  - Human-readable format
- **Use Cases**:
  - Email sharing
  - Quick review
  - Text processing
  - Universal compatibility
- **Example**:
  ```
  ECG RECORDING REPORT
  ====================

  RECORDING INFORMATION
  ---------------------
  Recording ID: abc-123
  Start Date: Dec 21, 2024 at 2:30 PM
  Duration: 30.00 seconds
  Sampling Frequency: 512.00 Hz
  Classification: Sinus Rhythm
  Average Heart Rate: 72.0 bpm

  ANALYSIS RESULTS
  ----------------
  Rhythm: sinus
  Rhythm Confidence: 95.0%
  ...
  ```

### Export Features

#### Validation
- Checks for voltage measurements before export
- Prevents export of corrupted/incomplete data
- Provides clear error messages

#### Error Handling
- Graceful failure with informative messages
- Console logging for debugging
- User-friendly error display

#### User Feedback
- Progress indicator during export
- Success confirmation with checkmark
- Error display with retry option
- Warning if recording not analyzed (for formats that include analysis)

#### Share Integration
- Seamless iOS share sheet integration
- Save to Files app
- Email, AirDrop, cloud storage
- Any installed share extensions

### Export Workflow

1. Open ECG recording in detail view
2. Tap "Export Recording" button
3. Select export format:
   - Read format descriptions
   - Choose based on use case
4. Tap "Export [Format]" button
5. Wait for processing (with progress indicator)
6. See success message
7. Use iOS share sheet to:
   - Save to Files
   - Share via AirDrop
   - Email to recipient
   - Upload to cloud storage

## Implementation Details

### Backend Configuration Classes

#### BackendPreset Enum
```swift
enum BackendPreset {
    case cloud      // Cloud backend
    case local      // Local network backend
    case custom     // User-defined backend
}
```

#### ECGAnalysisService Enhancements
- `updateConfiguration()`: Apply new backend settings
- `setupFallbackURLs()`: Configure automatic fallback
- `analyzeECG()`: Main analysis method with fallback logic
- `analyzeECGWithURL()`: Individual backend attempt
- `checkServerHealth()`: Health check endpoint

### Export Manager

#### ExportManager Class
- Static methods for each export format
- `exportECGRecording()`: Main export dispatcher
- `exportAsJSON()`: JSON format export
- `exportAsCSV()`: CSV format export
- `exportAsPDF()`: PDF format export
- `exportAsText()`: Text format export
- `saveToTemporaryFile()`: File system handling

### UI Components

#### SettingsView
- Backend preset selector with radio buttons
- Real-time server status indicator
- Help text and setup instructions
- Save button with configuration update

#### LocalBackendInfoView
- Step-by-step setup guide
- IP address instructions
- Server startup commands
- Troubleshooting tips

#### ExportOptionsView
- Format selector with descriptions
- Export progress indicator
- Success/error feedback
- Analysis status warning

## Benefits of the Hybrid Approach

### For Users
1. **Flexibility**: Choose backend based on needs (speed vs convenience)
2. **Reliability**: Automatic fallback ensures analysis works
3. **Data Control**: Local backend keeps data on local network
4. **Export Options**: Multiple formats for different workflows
5. **Ease of Use**: Clear UI with helpful instructions

### For Developers
1. **Modularity**: Clean separation of concerns
2. **Extensibility**: Easy to add new backends or formats
3. **Testability**: Can test with local backend
4. **Debugging**: Console logging throughout
5. **Error Handling**: Robust error recovery

### For Healthcare Providers
1. **Professional Reports**: PDF format for medical records
2. **Data Export**: CSV for further analysis
3. **Flexibility**: Works in various network environments
4. **Reliability**: Fallback ensures critical data access

## Best Practices

### Backend Selection
- **Use Cloud** when:
  - You want zero configuration
  - You're not on the same network as backend
  - You don't have technical expertise
- **Use Local** when:
  - You want fastest analysis
  - You're on the same Wi-Fi network
  - You want to keep data local
  - You're developing/testing
- **Use Custom** when:
  - You have a custom deployment
  - You need specific backend configuration
  - You're running alternative implementation

### Export Selection
- **Use JSON** when:
  - You need complete data
  - You're a developer
  - You want programmatic access
- **Use CSV** when:
  - You need time-series data
  - You're doing statistical analysis
  - You want to plot in Excel/MATLAB
- **Use PDF** when:
  - Sharing with healthcare providers
  - Creating medical records
  - Presenting professionally
- **Use Text** when:
  - You want simple readability
  - Emailing results
  - Quick review

## Troubleshooting

### Backend Issues

**"Offline" status in settings**:
- Check network connectivity
- Verify backend is running
- For local: check IP address is correct
- For cloud: first request may show offline (cold start)

**Analysis fails with local backend**:
- Verify computer and iPhone on same WiFi
- Check backend is running: `curl http://[ip]:8000/health`
- Check firewall allows connections on port 8000
- App will automatically try fallback to cloud

**Fallback message appears**:
- Normal if local backend unavailable
- Analysis still succeeds using cloud
- To avoid: ensure local backend is running and accessible

### Export Issues

**Export fails**:
- Check device has sufficient storage
- Verify app has file system permissions
- Try different export format
- Check console for detailed error messages

**PDF export shows blank**:
- Ensure recording has valid data
- Check voltage measurements exist
- Try CSV or JSON format first to verify data

**Share sheet doesn't appear**:
- Check iOS permissions
- Try restarting the app
- Ensure Files app is not restricted

## Future Enhancements

Potential improvements to the hybrid approach:

1. **Backend Management**
   - Multiple saved backend configurations
   - Quick-switch between backends
   - Bandwidth usage tracking
   - Response time metrics

2. **Export Enhancements**
   - Batch export multiple recordings
   - Custom export templates
   - DICOM format support
   - Waveform visualization in exports

3. **Sync Features**
   - Cloud backup of analyses
   - Multi-device sync
   - Automatic export on analysis completion
   - Export scheduling

4. **Advanced Fallback**
   - Multiple fallback backends
   - Load balancing
   - Automatic backend selection based on speed
   - Offline analysis capability

## Summary

The hybrid approach provides:

✅ **Three backend options**: Cloud, Local, Custom
✅ **Automatic fallback**: Ensures analysis succeeds
✅ **Four export formats**: JSON, CSV, PDF, Text
✅ **Robust error handling**: Clear feedback on failures
✅ **Easy configuration**: Intuitive UI with helpful guides
✅ **Professional quality**: Ready for both casual and clinical use

This implementation ensures the ECG Analyzer app works reliably in various scenarios while providing users with maximum flexibility and control over their data.
