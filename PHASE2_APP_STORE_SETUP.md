# Phase 2: App Store Connect Setup Guide

**App Name:** ECG Insights
**Bundle ID:** com.cyclingecg.ECGHealthKit
**Version:** 1.0.0
**Build:** 1

---

## ✅ 2.1 Apple Developer Account (COMPLETED)
- [x] You already have an active Apple Developer Account

---

## 📝 2.2 App Store Connect Configuration

### Step 1: Create New App Record

1. **Go to App Store Connect**
   - Visit: https://appstoreconnect.apple.com
   - Sign in with your Apple Developer credentials

2. **Create New App**
   - Click "My Apps"
   - Click the "+" button
   - Select "New App"

3. **Fill in Basic Information**
   - **Platform:** iOS
   - **Name:** ECG Insights
   - **Primary Language:** English (US)
   - **Bundle ID:** Select `com.cyclingecg.ECGHealthKit` from dropdown
     - ⚠️ If not in dropdown, you need to register it in Apple Developer Portal first
   - **SKU:** ecganalyzer-001 (internal tracking ID, can be anything unique)
   - **User Access:** Full Access

4. **Click "Create"**

### Step 2: Register Bundle ID (if needed)

If Bundle ID doesn't appear in dropdown:

1. Go to https://developer.apple.com/account/resources/identifiers/list
2. Click "+" to create new identifier
3. Select "App IDs" → Continue
4. Type: App
5. Description: ECG Insights
6. Bundle ID: `com.cyclingecg.ECGHealthKit` (Explicit)
7. **Capabilities to enable:**
   - [x] HealthKit
   - [x] Data Protection (complete protection)
8. Click "Continue" → "Register"
9. Return to App Store Connect and refresh

---

## 📱 2.3 App Metadata & Screenshots

### App Information

**App Name:** ECG Insights
**Subtitle (30 chars max):** Apple Watch ECG Analysis

**Category:**
- **Primary:** Health & Fitness
- **Secondary:** Medical

**Age Rating:** 17+
- Reason: Medical/Treatment Information (Frequent/Intense)

**Copyright:** © 2024 [Your Name/Company]

**Privacy Policy URL:**
- You need to create: https://[yourwebsite].com/ecg-analyzer-privacy-policy
- OR use GitHub Pages: https://[yourusername].github.io/ecg-analyzer/privacy.html

**Support URL:**
- Create: https://[yourwebsite].com/ecg-analyzer-support
- OR GitHub: https://github.com/[yourusername]/cyclingECG/issues

---

### App Description (Max 4000 characters)

```
ECG Insights helps you understand your Apple Watch ECG recordings with advanced metrics and trend analysis.

FEATURES

Heart Analysis:
• View all your Apple Watch ECG recordings in one place
• Advanced rhythm classification with confidence scores
• Detailed waveform analysis (P wave, QRS complex, QT intervals)
• Heart rate variability (HRV) metrics (SDNN, RMSSD)

Trends & History:
• 30-day trend tracking for key metrics
• Visual charts showing heart rate and HRV patterns
• Historical comparison of ECG recordings
• 90-day local history with automatic cleanup

Export Options:
• JSON format for developers and data scientists
• CSV format for spreadsheet analysis
• PDF reports for healthcare providers
• Text summaries for easy sharing

Privacy & Security:
• Your data never leaves your device without your explicit action
• End-to-end encryption for all transmitted data
• API keys stored in secure iOS Keychain
• Local data encrypted with AES-256-GCM
• No advertising, no tracking, no data mining
• GDPR and CCPA compliant

Backend Flexibility:
• Cloud backend for easy setup (no configuration needed)
• Local network backend for faster analysis
• Custom backend support for advanced users
• Automatic fallback with user consent

REQUIREMENTS

• Apple Watch Series 4 or later (for ECG recording)
• iPhone with iOS 16.0 or later
• HealthKit permission for ECG data access

MEDICAL DISCLAIMER

⚠️ IMPORTANT: This app is for informational and educational purposes ONLY.

• NOT a medical device
• NOT intended for diagnosis or treatment
• NOT a substitute for professional medical advice
• Results are algorithmically generated, not reviewed by physicians

If you experience chest pain, shortness of breath, or other concerning symptoms, seek immediate medical attention. Always consult with a qualified healthcare provider before making medical decisions based on app results.

PRIVACY COMMITMENT

We take your health data privacy seriously:
• Read-only HealthKit access (we never write data)
• Local analysis history encrypted on your device
• Backend data retention: maximum 30 days
• No sale or sharing of health data to third parties
• Full GDPR data deletion rights

ABOUT THE ANALYSIS

ECG Insights uses signal processing algorithms to:
• Detect R-peaks using Pan-Tompkins algorithm
• Calculate heart rate and rhythm metrics
• Analyze waveform intervals and morphology
• Generate statistical summaries

Analysis accuracy may vary based on recording quality. Results should always be reviewed by healthcare professionals for clinical decisions.

SUPPORT

For questions, feedback, or privacy requests:
• Email: [your-support-email]
• Documentation: [your-support-url]
• Privacy requests: We respond within 30 days per GDPR

VERSION HISTORY

1.0.0 - Initial Release
• Apple Watch ECG extraction and analysis
• Multi-format export (JSON, CSV, PDF, TXT)
• Secure encrypted local storage
• Cloud and local backend support
• 30-day trend tracking
• Privacy-first design
```

**Character count:** ~2,450 (well under 4000 limit)

---

### Keywords (Max 100 characters, comma-separated)

```
ECG,electrocardiogram,heart,HRV,Apple Watch,cardiology,arrhythmia,AFib,health,analysis
```

**Character count:** 96 ✅

---

### Promotional Text (Max 170 characters)

```
Understand your Apple Watch ECG recordings with detailed rhythm analysis, HRV metrics, trend tracking, and secure multi-format export.
```

**Character count:** 147 ✅

---

### App Previews & Screenshots

#### Required Sizes

**iPhone 6.7" Display** (iPhone 14 Pro Max, 15 Pro Max)
- Resolution: 1290 x 2796 pixels
- Format: PNG or JPEG
- Required: 3-10 screenshots

**iPhone 6.5" Display** (iPhone 11 Pro Max, XS Max)
- Resolution: 1242 x 2688 pixels
- Required for backwards compatibility

**iPhone 5.5" Display** (iPhone 8 Plus)
- Resolution: 1242 x 2208 pixels
- May be required depending on deployment target

#### Suggested Screenshots (in order)

1. **ECG List View**
   - Title: "All Your ECG Recordings"
   - Overlay text: "View complete history from Apple Watch"

2. **ECG Detail with Waveform**
   - Title: "Detailed Waveform Analysis"
   - Overlay text: "See voltage patterns and rhythm classification"

3. **Analysis Results**
   - Title: "Comprehensive Metrics"
   - Overlay text: "Heart rate, HRV, intervals, and more"

4. **30-Day Trends**
   - Title: "Track Your Heart Health"
   - Overlay text: "Visualize trends over time"

5. **Export Options**
   - Title: "Export in Any Format"
   - Overlay text: "JSON, CSV, PDF, or Text"

6. **Settings & Privacy**
   - Title: "Privacy-First Design"
   - Overlay text: "Encrypted storage, GDPR compliant"

**Screenshot Tips:**
- Use light mode for consistency
- Show real (but anonymized) data
- Add subtle overlays explaining features
- Keep text minimal and readable
- Use iPhone with notch for modern appearance

---

## 🔒 2.4 Privacy Nutrition Labels

### Data Collection Configuration

Go to App Store Connect → Your App → App Privacy

#### Data Collected: YES

**Health and Fitness**
- [x] Health
  - Data Type: Health
  - Data Use: App Functionality, Analytics
  - Linked to User: NO
  - Used for Tracking: NO
  - Reason: "ECG recordings, heart rate, and rhythm metrics from Apple Watch for analysis and trend visualization"

#### Data NOT Collected

- [ ] Contact Info
- [ ] Location
- [ ] User Content
- [ ] Browsing History
- [ ] Search History
- [ ] Identifiers (if not using analytics)
- [ ] Purchases
- [ ] Financial Info
- [ ] Sensitive Info

#### Privacy Policy URL
```
https://[yourwebsite].com/ecg-analyzer-privacy-policy
```

#### Optional: OpenAI Integration Notice

If backend uses OpenAI for narrative generation:
- Add note: "If AI narrative feature is enabled by backend administrator, anonymized health metrics (not raw ECG data) may be sent to OpenAI API for summary generation. This feature is opt-in and disclosed in Privacy Policy."

---

## 🎂 2.5 Age Rating Questionnaire

### Answers for App Review

**Made for Kids:** NO

**Age Rating Questions:**

1. **Cartoon or Fantasy Violence:** None
2. **Realistic Violence:** None
3. **Sexual Content or Nudity:** None
4. **Profanity or Crude Humor:** None
5. **Alcohol, Tobacco, or Drug Use References:** None
6. **Mature/Suggestive Themes:** None
7. **Horror/Fear Themes:** None
8. **Gambling:** None
9. **Contests:** None
10. **Unrestricted Web Access:** NO
11. **Medical/Treatment Information:** ✅ **Frequent/Intense**
    - Reason: App analyzes ECG data and provides heart rhythm metrics

**Result:** 17+ (Medical)

**Why 17+ not 12+:**
- Medical information is frequent and central to app purpose
- ECG analysis could be misinterpreted by younger users
- Apple Watch ECG feature requires 22+ in most regions anyway

---

## 🔐 2.6 Export Compliance

### Encryption Questions

**Does your app use encryption?**
- Answer: **YES**

**Which encryption does your app use?**
- [x] App connects over HTTPS
- [x] App uses iOS Keychain for secure storage
- [x] App uses standard iOS encryption (AES-GCM via CryptoKit)

**Encryption Type:**
- Standard encryption (AES-256-GCM)
- TLS 1.3 for network connections
- iOS Keychain (Apple-provided)

**Is your app exempt from export compliance?**
- Answer: **YES**
- Reason: Uses only standard iOS encryption APIs (Keychain, CryptoKit, URLSession HTTPS)
- No custom cryptographic implementations
- Qualifies for exemption under Category 5, Part 2

**Documentation needed:** NONE
- Apps using only Apple-provided encryption APIs are exempt
- No ERN (Encryption Registration Number) required

**Reference:**
https://developer.apple.com/documentation/security/complying_with_encryption_export_regulations

---

## 📋 2.7 App Review Information

### Demo Account Information

**Demo Account Required:** NO (app uses user's own HealthKit data)

**Notes for Reviewer:**

```
TESTING INSTRUCTIONS

1. HEALTHKIT PERMISSION
   - On first launch, grant HealthKit permission when prompted
   - If no Apple Watch ECG recordings available, app will show empty list

2. CREATING TEST ECG (if needed)
   - Reviewer will need Apple Watch Series 4+ to create ECG
   - Open Apple Watch ECG app
   - Create 1-2 test ECG recordings
   - Recordings will appear in iOS app

3. ANALYSIS FEATURE
   - Tap any ECG recording to view details
   - Tap "Analyze" button
   - Backend URL: https://cyclingecg.onrender.com (pre-configured)
   - Note: First request may take 30-50 seconds (free tier cold start)
   - No API key required for testing

4. EXPORT FEATURE
   - From ECG detail view, tap "Export Recording"
   - Choose format: JSON, CSV, PDF, or Text
   - Use iOS Share Sheet to save or share

5. PRIVACY POLICY
   - Shown automatically on first launch
   - Must accept to use app
   - Also accessible from Settings → Legal → Privacy Policy

HEALTH DATA USAGE
- App requests READ-ONLY access to ECG data
- No write access requested
- Data transmitted only when user taps "Analyze"
- All data encrypted in transit (HTTPS)
- Local storage encrypted (AES-256-GCM)

BACKEND
- Default: https://cyclingecg.onrender.com
- First request may have cold start delay
- Subsequent requests are faster
- No authentication required for testing

CONTACT
- For review questions: [your-email]
- Response time: 24-48 hours
```

### Contact Information

**First Name:** [Your First Name]
**Last Name:** [Your Last Name]
**Phone Number:** [Your Phone with country code]
**Email:** [Your Email]

---

## 📤 Next Steps After Metadata Entry

After completing all metadata in App Store Connect:

### 1. Prepare Build
- [ ] Open Xcode project
- [ ] Select "Any iOS Device (arm64)" as build target
- [ ] Product → Clean Build Folder (⇧⌘K)
- [ ] Product → Archive
- [ ] Wait for archive to complete

### 2. Upload to App Store Connect
- [ ] Window → Organizer
- [ ] Select your archive
- [ ] Click "Distribute App"
- [ ] Select "App Store Connect"
- [ ] Upload symbols: YES
- [ ] Manage version: Automatically
- [ ] Click "Upload"
- [ ] Wait for processing (10-60 minutes)

### 3. TestFlight (Recommended)
- [ ] Once build is processed, add to TestFlight
- [ ] Add internal testers (your team)
- [ ] Test on real devices
- [ ] Verify all features work
- [ ] Check for crashes or bugs

### 4. Submit for Review
- [ ] Select processed build in App Store Connect
- [ ] Review all metadata
- [ ] Add screenshots
- [ ] Answer all questions
- [ ] Click "Submit for Review"

### 5. Wait for Review
- Average time: 24-48 hours
- Be ready to respond to questions
- Check email and App Store Connect daily

---

## ⚠️ Common Rejection Reasons

Based on APP_STORE_SUBMISSION_CHECKLIST.md:

1. **Privacy Policy Missing** → ✅ FIXED (PrivacyPolicyView shown on launch)
2. **Health Data for Marketing** → ✅ SAFE (no analytics SDKs)
3. **Inaccurate Medical Claims** → ✅ SAFE (disclaimers throughout)
4. **Crash on Launch** → ⚠️ TEST THOROUGHLY
5. **Unencrypted Health Data** → ✅ FIXED (Keychain + AES-256-GCM)
6. **Missing Export Compliance** → ⚠️ ANSWER QUESTIONS ABOVE
7. **Incomplete Metadata** → ⚠️ USE THIS GUIDE

---

## 📞 Support Resources

**Apple Resources:**
- App Store Connect: https://appstoreconnect.apple.com
- Developer Portal: https://developer.apple.com/account
- Review Guidelines: https://developer.apple.com/app-store/review/guidelines/
- App Store Connect Help: https://help.apple.com/app-store-connect/

**Your Resources:**
- Security Analysis: SECURITY_AND_PRIVACY_ANALYSIS.md
- Full Checklist: APP_STORE_SUBMISSION_CHECKLIST.md
- Hybrid Approach: iOS/HYBRID-APPROACH.md

---

**Last Updated:** December 24, 2024
**Status:** Ready for App Store Connect setup
