# ECG Analyzer iOS App

A native iOS application that extracts Apple Watch ECG recordings from HealthKit, analyzes them using the cyclingECG backend, and provides comprehensive export capabilities.

## Features

### HealthKit Integration
- **Secure Access**: Request and manage HealthKit permissions
- **ECG Extraction**: Retrieve all ECG recordings from Apple Watch
- **Complete Data**: Extract voltage measurements, timestamps, classifications, and symptoms
- **Real-time Sync**: Refresh to get the latest recordings

### ECG Analysis
- **Backend Integration**: Send ECG data to the cyclingECG analysis backend
- **Advanced Features**: Get rhythm classification, heart rate variability, signal quality
- **AI Narratives**: Optional patient summaries and clinical notes via OpenAI
- **Offline Support**: View and export recordings even without analysis

### Export Capabilities
- **Multiple Formats**:
  - **JSON**: Complete structured data with analysis results
  - **CSV**: Voltage measurements for use in analysis tools
  - **PDF**: Formatted report suitable for sharing with healthcare providers
  - **TXT**: Plain text summary for easy reading
- **Share Integration**: Export to Files, email, AirDrop, or any sharing extension
- **Metadata Included**: All exports include recording details and timestamps

### User Interface
- **Modern SwiftUI**: Clean, native iOS design
- **Classification Badges**: Visual indicators for sinus rhythm, AFib, etc.
- **Heart Rate Display**: Prominent display of average heart rate
- **Symptom Indicators**: Clear marking of recordings with symptoms
- **Detailed Views**: Comprehensive analysis results with color-coded metrics

## Requirements

- **iOS**: 16.0 or later
- **Device**: iPhone with paired Apple Watch
- **Apple Watch**: Series 4 or later (with ECG capability)
- **Xcode**: 14.0 or later for building
- **Backend**: cyclingECG FastAPI backend (optional for analysis)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cyclingECG.git
cd cyclingECG/iOS
```

### 2. Open in Xcode

1. Open Xcode
2. Select **File > Open**
3. Navigate to `iOS/ECGHealthKit.xcodeproj`
4. Double-click to open the project

### 3. Configure Code Signing

1. Select your development team in **Signing & Capabilities**
2. Ensure your Apple ID has HealthKit entitlements
3. Note: HealthKit apps require a real device for testing (won't work in Simulator)

The project is already configured with:
- ✅ HealthKit capability enabled
- ✅ All Swift source files added
- ✅ Info.plist with required HealthKit permissions
- ✅ Entitlements file configured

### 4. Backend Setup (Optional)

If you want to use the analysis features:

1. **Start the Backend**:
   ```bash
   cd ../  # Go to project root
   uvicorn app.main:app --reload
   ```

2. **Configure in App**:
   - Tap the gear icon (⚙️) in the app
   - Set **API URL** to your backend (e.g., `http://localhost:8000` for local)
   - Add **API Key** if your backend requires authentication
   - Note: Use your computer's local IP (e.g., `http://192.168.1.100:8000`) when testing on a physical device

## Usage

### First Launch

1. **Grant Permissions**:
   - On first launch, the app will request HealthKit access
   - Tap "Grant Access"
   - In the system dialog, enable read access for "Electrocardiograms (ECG)"

2. **Load Recordings**:
   - The app automatically loads all ECG recordings from HealthKit
   - Recordings are sorted by date (newest first)

### Viewing ECG Recordings

- **List View**: Shows all recordings with:
  - Classification badge (Sinus, AFib, Inconclusive, etc.)
  - Date and time
  - Heart rate
  - Symptom indicators

- **Detail View**: Tap any recording to see:
  - Complete recording information
  - Analysis results (if analyzed)
  - Export options

### Analyzing ECGs

1. Open a recording in detail view
2. Tap **"Analyze Now"** button
3. The app sends the ECG data to the backend
4. Results appear automatically when analysis completes

Analysis provides:
- Rhythm classification (sinus, irregular, etc.)
- Heart rate statistics (mean, min, max)
- Heart rate variability (SDNN, RMSSD)
- Signal quality assessment
- Optional AI-generated narratives

### Exporting Recordings

1. Open a recording in detail view
2. Tap **"Export Recording"**
3. Choose your preferred format:
   - **JSON**: For developers and data scientists
   - **CSV**: For spreadsheets and analysis software
   - **PDF**: For healthcare providers
   - **TXT**: For easy reading
4. Use the share sheet to save or send the file

### Settings

Access settings via the gear icon (⚙️):
- **API URL**: Set your backend server address
- **API Key**: Add authentication if required
- **Server Status**: Check if backend is online

## Architecture

### File Structure

```
iOS/ECGHealthKit/
├── ECGHealthKitApp.swift          # App entry point
├── ContentView.swift              # Main view with navigation
├── HealthKitManager.swift         # HealthKit integration
├── Models.swift                   # Data models
├── ECGAnalysisService.swift       # Backend API client
├── ExportManager.swift            # Export functionality
├── ECGListView.swift              # List of recordings
├── ECGDetailView.swift            # Recording detail view
├── Info.plist                     # App configuration
└── ECGHealthKit.entitlements      # HealthKit capabilities
```

### Key Components

#### HealthKitManager
- Handles all HealthKit interactions
- Requests permissions
- Fetches ECG recordings
- Extracts voltage measurements
- Published properties for SwiftUI reactivity

#### ECGAnalysisService
- Communicates with the backend API
- Sends ECG data for analysis
- Receives and stores results
- Health check functionality

#### ExportManager
- Generates exports in multiple formats
- Creates JSON, CSV, PDF, and TXT files
- Includes metadata and analysis results
- Uses temporary files for sharing

#### Models
- `ECGRecording`: Complete ECG data from HealthKit
- `ECGAnalysisRequest`: API request format
- `ECGAnalysisResponse`: API response format
- Codable support for JSON serialization

## Data Flow

1. **Extraction**: HealthKit → HealthKitManager → ECGRecording model
2. **Analysis**: ECGRecording → ECGAnalysisService → Backend API → ECGAnalysisResponse
3. **Display**: Models → SwiftUI Views (reactive updates)
4. **Export**: ECGRecording + Analysis → ExportManager → File → Share Sheet

## Backend API Integration

The app communicates with the cyclingECG backend using these endpoints:

- `GET /health`: Check server status
- `POST /v1/ecg/analyze`: Analyze ECG recording

### Request Format

```json
{
  "recording_id": "UUID",
  "samples": [voltage array in microvolts],
  "sampling_rate_hz": 512,
  "units": "uV",
  "lead": "I",
  "start_timestamp_utc": "2024-01-01T12:00:00Z",
  "device_info": {
    "manufacturer": "Apple",
    "model": "Apple Watch"
  },
  "context": {
    "symptoms": ["User reported symptoms"]
  }
}
```

### Response Format

See `Models.swift` for complete response structure including:
- Rhythm classification
- Heart rate statistics
- HRV metrics
- Signal quality
- Optional AI narratives

## Privacy & Security

- **HealthKit Data**: Never leaves the device without explicit user action (analyze or export)
- **Permissions**: App requests only read access to ECG data
- **Secure Communication**: Use HTTPS for production backends
- **API Keys**: Stored securely in UserDefaults (consider Keychain for production)
- **No Cloud Storage**: App does not store data in cloud services
- **User Control**: All data access and sharing controlled by the user

## Troubleshooting

### "HealthKit is not available"
- Ensure you're running on a real iPhone (not Simulator)
- Check that HealthKit capability is properly configured
- Verify your device supports HealthKit

### "No ECG Recordings Found"
- Take an ECG on your Apple Watch first
- Ensure iPhone and Watch are paired and synced
- Check that Health app on iPhone shows ECG recordings

### "Analysis failed"
- Verify backend server is running
- Check API URL in settings (use local IP, not localhost, for device testing)
- Ensure network connectivity
- Check backend logs for errors

### Authorization Issues
- Go to iPhone Settings > Privacy & Security > Health > ECG Analyzer
- Ensure "Electrocardiograms" is enabled for reading
- Try deleting and reinstalling the app

### Export Not Working
- Ensure device has sufficient storage
- Check app permissions for Files access
- Try a different export format

## Development

### Building for Testing

```bash
# Select your iPhone in Xcode
# Product > Run (⌘R)
```

### Running on Simulator

Note: HealthKit is not available on Simulator. You must use a physical device.

### Debugging

- Use Xcode's debugger and breakpoints
- Check Console for HealthKit authorization status
- Monitor network requests in backend logs
- Use `print()` statements in async/await code

### Testing with Sample Data

The backend includes test data. You can use the `/docs` endpoint to test the API independently:

```bash
# Open in browser
http://localhost:8000/docs
```

## Roadmap

Potential future enhancements:

- [ ] Offline ECG analysis (on-device ML)
- [ ] ECG waveform visualization
- [ ] Export to DICOM format
- [ ] Cloud sync and backup
- [ ] Trend analysis over time
- [ ] Comparison between recordings
- [ ] Apple Health integration (write calculated metrics back)
- [ ] watchOS companion app
- [ ] Widget support
- [ ] Siri shortcuts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the same license as the cyclingECG backend. See the LICENSE file in the root directory.

## Acknowledgments

- Apple HealthKit for ECG data access
- FastAPI backend for ECG analysis
- OpenAI for narrative generation (optional)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to Apple's HealthKit documentation

## Disclaimer

**This app is for informational and educational purposes only. It is not intended to diagnose, treat, cure, or prevent any disease. Always consult with a qualified healthcare provider for medical advice.**

The ECG analysis provided by this app should not replace professional medical evaluation. If you experience symptoms or have concerns about your heart health, seek immediate medical attention.
