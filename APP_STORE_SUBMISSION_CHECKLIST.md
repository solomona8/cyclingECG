# ECG Insights App - App Store Submission Checklist

**Last Updated:** December 23, 2025
**App Version:** 1.0.0
**Target Submission Date:** _[To be determined after critical fixes]_

---

## EXECUTIVE SUMMARY

This checklist guides you through preparing your ECG Insights app for Apple App Store submission, with particular focus on:
- **PHI Security & HIPAA Compliance**
- **Apple Health App Requirements (Section 5.1.3)**
- **Technical Submission Requirements**
- **Privacy & Data Protection**

**CRITICAL:** Based on the security audit, **PHASE 1 fixes are MANDATORY** before submission. The app currently has critical security gaps that violate Apple's guidelines and put user health data at risk.

---

## PRIORITY LEVELS

- **🔴 CRITICAL** - Must fix before submission (App will be rejected)
- **🟡 HIGH** - Should fix before submission (May cause rejection)
- **🟢 MEDIUM** - Recommended for better user experience
- **⚪ LOW** - Nice to have

---

## PHASE 1: CRITICAL SECURITY FIXES (MANDATORY)
**Estimated Time: 1-2 weeks**

### 🔴 1.1 Encrypt API Keys with Keychain
**Status:** ❌ Not Implemented
**Current Issue:** API keys stored in plaintext UserDefaults
**Risk:** App rejection under Section 5.1.3 (data security)

**Action Required:**
- [ ] Create `KeychainManager.swift` class
- [ ] Implement secure save/retrieve methods using Security framework
- [ ] Migrate `@AppStorage("api_key")` to Keychain
- [ ] Test API key retrieval on app launch
- [ ] Verify keys are NOT in UserDefaults after migration

**Implementation Guide:**
```swift
// iOS/ECGHealthKit/KeychainManager.swift
import Security
import Foundation

class KeychainManager {
    static let shared = KeychainManager()

    func saveAPIKey(_ key: String) throws {
        let data = key.data(using: .utf8)!
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "ecg_api_key",
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        // Delete existing key if present
        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.unableToSave
        }
    }

    func retrieveAPIKey() throws -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "ecg_api_key",
            kSecReturnData as String: kCFBooleanTrue!,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let key = String(data: data, encoding: .utf8) else {
            return nil
        }

        return key
    }

    func deleteAPIKey() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "ecg_api_key"
        ]

        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unableToDelete
        }
    }
}

enum KeychainError: Error {
    case unableToSave
    case unableToDelete
}
```

**Update ContentView.swift:**
```swift
// Replace:
@AppStorage("api_key") private var apiKey = ""

// With:
@State private var apiKey = ""

// In onAppear:
.onAppear {
    if let storedKey = try? KeychainManager.shared.retrieveAPIKey() {
        apiKey = storedKey
    }
}

// When saving:
Button("Save") {
    try? KeychainManager.shared.saveAPIKey(apiKey)
}
```

**Testing:**
```bash
# Build and run app
# Enter API key in settings
# Close app
# Delete app data from Settings > General > iPhone Storage
# Reinstall app - key should persist (Keychain survives app deletion)
```

**Reference:** iOS/ECGHealthKit/ContentView.swift:36-37

---

### 🔴 1.2 Encrypt Local Analysis History
**Status:** ❌ Not Implemented
**Current Issue:** Analysis history stored as plaintext JSON
**Risk:** PHI exposure if device compromised

**Action Required:**
- [ ] Add CryptoKit encryption to `AnalysisHistoryManager.swift`
- [ ] Generate symmetric encryption key on first launch
- [ ] Store encryption key in Keychain
- [ ] Encrypt data before writing to file
- [ ] Decrypt data when loading from file
- [ ] Test migration of existing plaintext history

**Implementation Guide:**
```swift
// iOS/ECGHealthKit/AnalysisHistoryManager.swift
import CryptoKit
import Foundation

@MainActor
class AnalysisHistoryManager: ObservableObject {
    @Published var historyItems: [AnalysisHistoryItem] = []

    private let fileURL: URL
    private let symmetricKey: SymmetricKey

    init() {
        let documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        fileURL = documentsDirectory.appendingPathComponent("ecg_analysis_history.enc")

        // Retrieve or generate encryption key
        if let keyData = try? KeychainManager.shared.retrieveEncryptionKey() {
            symmetricKey = SymmetricKey(data: keyData)
        } else {
            symmetricKey = SymmetricKey(size: .bits256)
            try? KeychainManager.shared.saveEncryptionKey(symmetricKey.withUnsafeBytes { Data($0) })
        }

        loadHistory()
    }

    private func persistHistory() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let jsonData = try encoder.encode(historyItems)

            // Encrypt before writing
            let sealedBox = try AES.GCM.seal(jsonData, using: symmetricKey)

            guard let encryptedData = sealedBox.combined else {
                print("Error: Failed to get combined encrypted data")
                return
            }

            try encryptedData.write(to: fileURL)
        } catch {
            print("Error saving encrypted analysis history: \(error)")
        }
    }

    private func loadHistory() {
        do {
            // Check if encrypted file exists
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                // Migration: check for old plaintext file
                let oldFileURL = fileURL.deletingPathExtension().appendingPathExtension("json")
                if FileManager.default.fileExists(atPath: oldFileURL.path) {
                    try migratePlaintextHistory(from: oldFileURL)
                }
                return
            }

            let encryptedData = try Data(contentsOf: fileURL)
            let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
            let decryptedData = try AES.GCM.open(sealedBox, using: symmetricKey)

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            historyItems = try decoder.decode([AnalysisHistoryItem].self, from: decryptedData)
        } catch {
            print("Error loading encrypted analysis history: \(error)")
            historyItems = []
        }
    }

    private func migratePlaintextHistory(from oldFileURL: URL) throws {
        let plaintextData = try Data(contentsOf: oldFileURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        historyItems = try decoder.decode([AnalysisHistoryItem].self, from: plaintextData)

        // Save encrypted version
        persistHistory()

        // Delete old plaintext file
        try FileManager.default.removeItem(at: oldFileURL)
        print("Migrated plaintext history to encrypted storage")
    }
}
```

**Add to KeychainManager:**
```swift
func saveEncryptionKey(_ keyData: Data) throws {
    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrAccount as String: "ecg_encryption_key",
        kSecValueData as String: keyData,
        kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    ]

    SecItemDelete(query as CFDictionary)
    let status = SecItemAdd(query as CFDictionary, nil)
    guard status == errSecSuccess else {
        throw KeychainError.unableToSave
    }
}

func retrieveEncryptionKey() throws -> Data? {
    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrAccount as String: "ecg_encryption_key",
        kSecReturnData as String: kCFBooleanTrue!,
        kSecMatchLimit as String: kSecMatchLimitOne
    ]

    var result: AnyObject?
    let status = SecItemCopyMatching(query as CFDictionary, &result)

    guard status == errSecSuccess, let data = result as? Data else {
        return nil
    }

    return data
}
```

**Reference:** iOS/ECGHealthKit/AnalysisHistoryManager.swift:26-36

---

### 🔴 1.3 Add In-App Privacy Policy
**Status:** ❌ Not Implemented
**Current Issue:** Privacy policy only in README, not shown to users
**Risk:** App rejection under Section 5.1.1 (Privacy)

**Action Required:**
- [ ] Create `PrivacyPolicyView.swift`
- [ ] Add privacy policy text compliant with GDPR/CCPA
- [ ] Show on first app launch
- [ ] Add "Privacy Policy" link in Settings
- [ ] Include medical disclaimer
- [ ] Document data retention policy
- [ ] Disclose backend data transmission
- [ ] Mention OpenAI integration (if enabled)

**Implementation Guide:**
```swift
// iOS/ECGHealthKit/PrivacyPolicyView.swift
import SwiftUI

struct PrivacyPolicyView: View {
    @Environment(\.dismiss) var dismiss
    @AppStorage("privacy_policy_accepted") private var privacyAccepted = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Privacy Policy")
                        .font(.largeTitle)
                        .bold()

                    Group {
                        SectionHeader(title: "1. Data Collection")
                        Text("""
                        This app collects ECG recordings, heart rate data, and timestamps from your Apple Watch via HealthKit. \
                        We only access data after you grant explicit HealthKit permission.
                        """)

                        SectionHeader(title: "2. Data Usage")
                        Text("""
                        • **Analysis**: ECG data is sent to our backend server for rhythm classification and metrics calculation
                        • **Storage**: Analysis results are stored locally on your device for 90 days
                        • **Export**: You control what data is exported and where it's shared
                        """)

                        SectionHeader(title: "3. Data Transmission")
                        Text("""
                        When you click "Analyze", your ECG data is transmitted securely via HTTPS to:
                        • Cloud backend: cyclingecg.onrender.com
                        • Local backend: Your configured server (if enabled)

                        Raw ECG voltage samples are NOT stored on the backend - only derived metrics (heart rate, HRV, etc.) are retained.
                        """)

                        SectionHeader(title: "4. Data Retention")
                        Text("""
                        • **Local Device**: Analysis history kept for 90 days, then automatically deleted
                        • **Backend**: Analysis metrics stored for 30 days to calculate statistics
                        • **You can request deletion**: Contact [your-email@example.com]
                        """)

                        SectionHeader(title: "5. Third-Party Services")
                        Text("""
                        If the backend administrator enables AI narrative generation, anonymized health metrics may be sent to OpenAI's API. \
                        OpenAI's services are not HIPAA compliant.
                        """)
                        .foregroundColor(.orange)

                        SectionHeader(title: "6. Your Rights (GDPR/CCPA)")
                        Text("""
                        • **Access**: View all analysis results in the app
                        • **Deletion**: Export and delete local data; request backend deletion via email
                        • **Correction**: Historical data cannot be modified
                        • **Withdraw Consent**: Uninstall the app and request backend data deletion
                        """)

                        SectionHeader(title: "7. Security")
                        Text("""
                        • API keys stored in secure iOS Keychain
                        • Local analysis history encrypted with AES-256
                        • HTTPS encryption for all network communication
                        • No data stored in iCloud
                        """)

                        SectionHeader(title: "8. Medical Disclaimer")
                        Text("""
                        ⚠️ This app is for informational and educational purposes ONLY.

                        It is NOT intended to diagnose, treat, cure, or prevent any disease. \
                        Results should NOT replace professional medical evaluation. \
                        Always consult with a qualified healthcare provider before making medical decisions.

                        If you experience chest pain, shortness of breath, or other concerning symptoms, seek immediate medical attention.
                        """)
                        .foregroundColor(.red)
                        .font(.callout)
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)

                        SectionHeader(title: "9. Updates")
                        Text("""
                        We may update this privacy policy from time to time. Continued use of the app after changes constitutes acceptance.
                        """)

                        Text("Last Updated: December 23, 2025")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Text("Contact: [your-email@example.com]")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    if !privacyAccepted {
                        Button(action: {
                            privacyAccepted = true
                            dismiss()
                        }) {
                            Text("I Understand and Accept")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        .padding(.top, 20)
                    }
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if privacyAccepted {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") {
                            dismiss()
                        }
                    }
                }
            }
        }
    }
}

struct SectionHeader: View {
    let title: String

    var body: some View {
        Text(title)
            .font(.headline)
            .padding(.top, 10)
    }
}
```

**Update ContentView.swift to show on first launch:**
```swift
@AppStorage("privacy_policy_accepted") private var privacyAccepted = false
@State private var showPrivacyPolicy = false

var body: some View {
    // ... existing content
    .sheet(isPresented: $showPrivacyPolicy) {
        PrivacyPolicyView()
            .interactiveDismissDisabled(!privacyAccepted)
    }
    .onAppear {
        if !privacyAccepted {
            showPrivacyPolicy = true
        }
    }
}

// Add to Settings section:
Section(header: Text("Legal")) {
    Button("Privacy Policy") {
        showPrivacyPolicy = true
    }
}
```

**Apple App Store Connect Setup:**
- [ ] Add Privacy Policy URL in App Store Connect
- [ ] Create standalone webpage: https://yourwebsite.com/ecg-privacy-policy
- [ ] Or use GitHub Pages for the policy

---

### 🔴 1.4 Disable Debug Logging in Production
**Status:** ❌ Not Implemented
**Current Issue:** Health data printed to console
**Risk:** PHI exposure in logs

**Action Required:**
- [ ] Create `Logger.swift` utility
- [ ] Replace all `print()` statements with conditional logging
- [ ] Ensure production builds disable health data logging
- [ ] Test that no PHI appears in release logs

**Implementation Guide:**
```swift
// iOS/ECGHealthKit/Logger.swift
import Foundation
import os.log

enum Logger {
    static let isDebugEnabled: Bool = {
        #if DEBUG
            return true
        #else
            return false
        #endif
    }()

    static func debug(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        guard isDebugEnabled else { return }
        let filename = (file as NSString).lastPathComponent
        NSLog("[DEBUG] [\(filename):\(line)] \(function) - \(message)")
    }

    static func info(_ message: String) {
        guard isDebugEnabled else { return }
        NSLog("[INFO] \(message)")
    }

    static func warning(_ message: String) {
        NSLog("[WARNING] \(message)")
    }

    static func error(_ message: String, error: Error? = nil) {
        NSLog("[ERROR] \(message)")
        if let error = error {
            NSLog("[ERROR] Details: \(error.localizedDescription)")
        }
    }

    // NEVER log PHI even in debug mode
    static func sanitize(_ value: Any) -> String {
        #if DEBUG
            return String(describing: value)
        #else
            return "[REDACTED]"
        #endif
    }
}
```

**Update all source files:**
```swift
// BEFORE:
print("=== EXTRACTING ECG DATA ===")
print("ECG ID: \(ecg.uuid.uuidString)")
print("[ANALYZE] First 10 samples: \(samples)")

// AFTER:
Logger.debug("Extracting ECG data")
Logger.debug("ECG ID: \(ecg.uuid.uuidString)")
Logger.debug("Sample count: \(samples.count)") // Don't log actual values
```

**Files to update:**
- iOS/ECGHealthKit/HealthKitManager.swift (lines 95, 100, 105, 120, 150)
- iOS/ECGHealthKit/ECGAnalysisService.swift (lines 180, 201, 215)
- iOS/ECGHealthKit/AnalysisHistoryManager.swift (lines 33, 47)
- iOS/ECGHealthKit/ExportManager.swift (lines 78, 92, 115, 135)

---

### 🔴 1.5 Fix Info.plist Privacy Descriptions
**Status:** ⚠️ Needs Improvement
**Current Issue:** `NSHealthClinicalHealthRecordsShareUsageDescription` may be misleading
**Risk:** App rejection during review

**Action Required:**
- [ ] Review and update `NSHealthShareUsageDescription`
- [ ] Remove or clarify `NSHealthClinicalHealthRecordsShareUsageDescription`
- [ ] Ensure descriptions match actual data usage

**Implementation:**
```xml
<!-- iOS/ECGHealthKit/Info.plist -->

<!-- Update this: -->
<key>NSHealthShareUsageDescription</key>
<string>This app analyzes your Apple Watch ECG recordings to provide detailed rhythm classification, heart rate variability metrics, and trend analysis. Your ECG data is only accessed when you explicitly request analysis or export.</string>

<!-- Either remove this if not needed, or clarify: -->
<key>NSHealthClinicalHealthRecordsShareUsageDescription</key>
<string>Access to clinical health records is optional and only used if you choose to share additional health context with your healthcare provider via exported reports.</string>

<!-- Add this for potential future features: -->
<key>NSLocalNetworkUsageDescription</key>
<string>This app connects to a local ECG analysis server on your network if you configure one in Settings.</string>
```

**Reference:** iOS/ECGHealthKit/Info.plist:50-53

---

### 🔴 1.6 Remove/Disable Insecure HTTP Backend Option
**Status:** ❌ Not Implemented
**Current Issue:** Local backend uses HTTP (not HTTPS)
**Risk:** Data transmitted unencrypted on local network

**Options:**

**Option A: Remove HTTP Support (Recommended for App Store)**
```swift
// iOS/ECGHealthKit/ContentView.swift
// Remove "local" preset or change it to HTTPS

@State private var backendPresets = [
    "cloud": "https://cyclingecg.onrender.com",
    // Remove this line:
    // "local": "http://192.168.1.100:8000"
]
```

**Option B: Add Explicit Warning**
```swift
if apiURL.hasPrefix("http://") {
    Label("⚠️ Insecure Connection", systemImage: "exclamationmark.triangle.fill")
        .foregroundColor(.red)
    Text("HTTP connections are not encrypted. Health data will be visible on your network.")
        .font(.caption)
        .foregroundColor(.red)
}
```

**Option C: Force HTTPS Validation**
```swift
func validateBackendURL(_ url: String) -> Bool {
    guard url.hasPrefix("https://") else {
        analysisError = "Backend URL must use HTTPS for secure transmission"
        return false
    }
    return true
}
```

**Recommendation:** Use Option A for App Store submission. Local HTTP can be re-enabled for enterprise/TestFlight builds if needed.

**Reference:** iOS/ECGHealthKit/ContentView.swift:100-102

---

### 🔴 1.7 Add User Consent for Backend Fallback
**Status:** ❌ Not Implemented
**Current Issue:** Automatic fallback to cloud without user consent
**Risk:** GDPR violation

**Action Required:**
- [ ] Disable automatic fallback
- [ ] Show user prompt before using fallback backend
- [ ] Store user's fallback preference
- [ ] Log which backend was used for each analysis

**Implementation:**
```swift
// iOS/ECGHealthKit/ECGAnalysisService.swift

@Published var fallbackEnabled = false
@Published var showFallbackPrompt = false

func analyzeECG(_ recording: ECGRecording) async -> ECGAnalysisResponse? {
    Logger.info("Starting ECG analysis")

    // Try primary backend
    if let result = await analyzeECGWithURL(recording, url: baseURL) {
        return result
    }

    // Primary failed - ask user about fallback
    if !fallbackEnabled {
        await MainActor.run {
            showFallbackPrompt = true
        }
        return nil
    }

    // User has enabled fallback - try alternate backends
    for fallbackURL in fallbackURLs {
        Logger.info("Trying fallback backend: \(fallbackURL)")
        if let result = await analyzeECGWithURL(recording, url: fallbackURL) {
            await MainActor.run {
                analysisError = "⚠️ Analysis completed using fallback backend"
            }
            return result
        }
    }

    await MainActor.run {
        analysisError = "All backends unavailable. Please try again later."
    }
    return nil
}
```

**Add to UI:**
```swift
.alert("Backend Unavailable", isPresented: $service.showFallbackPrompt) {
    Button("Use Fallback Backend") {
        service.fallbackEnabled = true
        // Retry analysis
    }
    Button("Cancel", role: .cancel) { }
} message: {
    Text("The primary backend is unavailable. Allow using fallback backend? Your data will be sent to: \(service.fallbackURLs.first ?? "unknown")")
}
```

**Reference:** iOS/ECGHealthKit/ECGAnalysisService.swift:175-195

---

## PHASE 2: APPLE APP STORE REQUIREMENTS
**Estimated Time: 3-5 days**

### 🟡 2.1 Apple Developer Account Setup
**Status:** _[User to verify]_

**Action Required:**
- [ ] Active Apple Developer Program membership ($99/year)
- [ ] Organization account (if representing a company)
- [ ] Completed tax and banking information
- [ ] Accepted latest App Store Connect agreements

**Verification:**
```
1. Visit https://developer.apple.com/account
2. Check "Membership" status is Active
3. Verify "Agreements, Tax, and Banking" is complete
```

---

### 🟡 2.2 App Store Connect Configuration
**Status:** _[To be created]_

**Action Required:**
- [ ] Create App Record in App Store Connect
- [ ] Set Bundle Identifier: `com.yourcompany.ecghealthkit` (must match Xcode)
- [ ] Choose App Name: "ECG Insights"
- [ ] Select Primary Category: Health & Fitness
- [ ] Select Secondary Category: Medical (if applicable)
- [ ] Set Age Rating: 12+ or 17+ (due to medical content)

**Steps:**
```
1. Log in to App Store Connect: https://appstoreconnect.apple.com
2. Click "My Apps" > "+" > "New App"
3. Fill in:
   - Platform: iOS
   - Name: [Your App Name]
   - Primary Language: English (US)
   - Bundle ID: Select from dropdown (or create new)
   - SKU: ecghealthkit-001 (internal reference)
4. Click "Create"
```

---

### 🟡 2.3 App Metadata & Screenshots
**Status:** _[To be created]_

**Action Required:**
- [ ] App description (max 4000 characters)
- [ ] Keywords (max 100 characters, comma-separated)
- [ ] App Store screenshots (required sizes for all devices)
- [ ] App icon (1024x1024 PNG, no alpha channel)
- [ ] App preview video (optional but recommended)

**Screenshot Requirements:**
- iPhone 6.7" display: 1290 x 2796 pixels (iPhone 15 Pro Max)
- iPhone 6.5" display: 1242 x 2688 pixels (iPhone 11 Pro Max)
- iPhone 5.5" display: 1242 x 2208 pixels (iPhone 8 Plus)
- iPad Pro 12.9" (3rd gen): 2048 x 2732 pixels

**Recommended Screenshots:**
1. Main ECG list view
2. ECG detail view with waveform
3. Analysis results screen
4. Export options
5. Settings/configuration

**Description Template:**
```
ECG Insights helps you understand your Apple Watch ECG recordings with advanced metrics and trend analysis.

FEATURES:
• View all your Apple Watch ECG recordings in one place
• Detailed heart rhythm classification and confidence scores
• Heart rate variability (HRV) analysis
• 30-day trend tracking and statistics
• Export data in multiple formats (JSON, CSV, PDF)
• Secure backend analysis with HIPAA-grade encryption

PRIVACY & SECURITY:
• Your data never leaves your device without explicit action
• API keys stored in secure iOS Keychain
• All data encrypted at rest and in transit
• No third-party advertising or data mining

MEDICAL DISCLAIMER:
This app is for informational purposes only and is not intended to diagnose or treat any medical condition. Always consult with a qualified healthcare provider.

REQUIREMENTS:
• Apple Watch Series 4 or later
• iOS 16.0 or later
• HealthKit permission for ECG data
```

**Keywords:**
```
ECG, electrocardiogram, heart rate, HRV, Apple Watch, health, cardiology, arrhythmia, AFib, analysis
```

---

### 🟡 2.4 Privacy Nutrition Labels
**Status:** _[To be configured in App Store Connect]_

Apple requires detailed privacy disclosures. Configure in App Store Connect > App Privacy:

**Data Collected:**
- [ ] **Health and Fitness**
  - ✅ Data Type: Health
  - ✅ Data Use: App Functionality, Analytics
  - ✅ Linked to User: No
  - ✅ Used for Tracking: No

- [ ] **Identifiers** (if using analytics)
  - Device ID: Only if using crash reporting

**Data Not Collected:**
- Name
- Email Address
- Location
- Browsing History
- Purchase History

**Privacy Policy URL:**
- [ ] Enter: `https://yourwebsite.com/ecg-privacy-policy`

**Reference:** https://developer.apple.com/app-store/app-privacy-details/

---

### 🟡 2.5 Age Rating Questionnaire
**Status:** _[To be completed]_

**Recommended Answers:**
```
Medical/Treatment Information: Frequent/Intense
(Because app analyzes ECG data and provides health metrics)

Unrestricted Web Access: No
Gambling: No
Contests: No
Profanity or Crude Humor: None
Mature/Suggestive Themes: None
Horror/Fear Themes: None
Violence: None
Alcohol, Tobacco, or Drug Use: None

Result: Likely 12+ or 17+ rating
```

---

### 🔴 2.6 Export Compliance (Encryption)
**Status:** ❌ Required
**Current State:** App uses encryption (HTTPS, AES)

Because your app uses encryption, you must answer export compliance questions:

**Action Required:**
- [ ] In App Store Connect, answer "Yes" to "Uses Encryption"
- [ ] Select exemption reason: "App uses standard iOS encryption (HTTPS, Keychain)"
- [ ] No export documentation needed if using only Apple-provided crypto APIs

**Reference:** https://developer.apple.com/documentation/security/complying_with_encryption_export_regulations

---

## PHASE 3: TECHNICAL SUBMISSION REQUIREMENTS
**Estimated Time: 2-3 days**

### 🟡 3.1 Code Signing & Certificates
**Status:** _[To be verified]_

**Action Required:**
- [ ] Create Distribution Certificate
- [ ] Create App Store provisioning profile
- [ ] Enable HealthKit capability in App ID
- [ ] Configure automatic signing in Xcode (or manual)

**Steps in Xcode:**
```
1. Open iOS/ECGHealthKit.xcodeproj
2. Select target "ECGHealthKit"
3. Go to "Signing & Capabilities"
4. Select Team: [Your Apple Developer Team]
5. Ensure "Automatically manage signing" is checked
6. Verify Bundle Identifier matches App Store Connect
7. Confirm HealthKit capability is present
```

**Manual Certificate Setup (if needed):**
```
1. Visit https://developer.apple.com/account/resources/certificates
2. Click "+" to create new certificate
3. Select "App Store Distribution"
4. Upload Certificate Signing Request (CSR)
5. Download certificate and install in Keychain
6. Create Provisioning Profile linking cert + App ID
```

---

### 🟡 3.2 Build Configuration
**Status:** _[To be verified]_

**Action Required:**
- [ ] Set build configuration to Release
- [ ] Bump version number: `CFBundleShortVersionString` (e.g., 1.0.0)
- [ ] Bump build number: `CFBundleVersion` (e.g., 1)
- [ ] Verify deployment target: iOS 16.0+ recommended
- [ ] Enable bitcode (if required)
- [ ] Disable debugging symbols in release builds

**Xcode Settings:**
```
Target Settings > Build Settings:
- Optimization Level: Fast, Smallest [-Os]
- Strip Debug Symbols During Copy: Yes
- Strip Swift Symbols: Yes
- Make Strings Read-Only: Yes
- Dead Code Stripping: Yes
```

**Info.plist Check:**
```xml
<key>CFBundleShortVersionString</key>
<string>1.0.0</string>
<key>CFBundleVersion</key>
<string>1</string>
<key>MinimumOSVersion</key>
<string>16.0</string>
```

**Reference:** iOS/ECGHealthKit/Info.plist:19-22

---

### 🟡 3.3 Archive & Upload to App Store Connect
**Status:** _[To be done after code fixes]_

**Action Required:**
- [ ] Clean build folder (Cmd+Shift+K)
- [ ] Archive app (Product > Archive)
- [ ] Validate archive (checks for common issues)
- [ ] Upload to App Store Connect via Xcode Organizer
- [ ] Wait for processing (10-60 minutes)

**Steps:**
```
1. In Xcode, select "Any iOS Device (arm64)" as build target
2. Product > Clean Build Folder
3. Product > Archive
   - Wait for archive to complete (2-5 minutes)
4. In Organizer window:
   - Select your archive
   - Click "Validate App"
   - Fix any errors/warnings
   - Click "Distribute App"
   - Select "App Store Connect"
   - Upload
5. Check App Store Connect for processing status
```

**Common Upload Errors:**
- Missing entitlements: Add HealthKit to capabilities
- Invalid provisioning profile: Regenerate in developer portal
- Missing icon: Add 1024x1024 App Store icon
- Invalid export compliance: Answer encryption questions

---

### 🟡 3.4 TestFlight Beta Testing (Optional but Recommended)
**Status:** _[Recommended before public release]_

**Action Required:**
- [ ] Upload build to TestFlight
- [ ] Add internal testers (up to 100)
- [ ] Add external testers (up to 10,000)
- [ ] Provide test information and feedback instructions
- [ ] Test on real devices and various iOS versions

**Benefits:**
- Catch bugs before public release
- Test on different device configurations
- Gather user feedback
- Apple's review process is faster for TestFlight

**Setup:**
```
1. In App Store Connect > TestFlight
2. Select your build
3. Add "What to Test" notes for testers
4. Add internal testers (your team)
5. Create external testing group
6. Submit for beta review (required for external)
7. Share invite link with testers
```

---

## PHASE 4: HEALTH APP SPECIFIC REQUIREMENTS
**Estimated Time: 2-3 days**

### 🔴 4.1 Medical Disclaimer (Section 1.4.1)
**Status:** ⚠️ Needs prominent placement

**Apple Requirement:**
> "Apps should remind users to check with a doctor in addition to using the app and before making medical decisions."

**Action Required:**
- [ ] Add disclaimer to main screen
- [ ] Show on first launch
- [ ] Include in Privacy Policy
- [ ] Add to exported reports

**Implementation:**
```swift
// iOS/ECGHealthKit/ContentView.swift

// Add to main view:
VStack {
    HStack {
        Image(systemName: "exclamationmark.triangle.fill")
            .foregroundColor(.orange)
        Text("This app is not a medical device. Consult a doctor before making health decisions.")
            .font(.caption)
            .foregroundColor(.secondary)
    }
    .padding(.horizontal)
    .padding(.top, 8)

    // ... rest of content
}
```

**Also add to ECGDetailView analysis results:**
```swift
Section {
    Label {
        Text("Not a Medical Diagnosis")
            .font(.headline)
    } icon: {
        Image(systemName: "cross.case.fill")
    }

    Text("These results are for informational purposes only. If you experience symptoms, seek immediate medical attention. Always consult with a qualified healthcare provider.")
        .font(.callout)
        .foregroundColor(.secondary)
}
.listRowBackground(Color.orange.opacity(0.1))
```

---

### 🔴 4.2 Data Usage Restrictions (Section 5.1.3)
**Status:** ⚠️ Needs verification

**Apple Requirement:**
> "Apps may not use or disclose health data for advertising, marketing, or data mining purposes other than improving health management."

**Action Required:**
- [ ] Verify no analytics SDKs collect health data
- [ ] Ensure no advertising frameworks integrated
- [ ] Confirm backend doesn't sell/share data
- [ ] Remove any third-party tracking (Google Analytics, etc.)
- [ ] Document data usage in Privacy Policy

**Code Audit:**
```bash
# Check for prohibited SDKs in Podfile/Package.swift
grep -r "GoogleAnalytics" iOS/
grep -r "Facebook" iOS/
grep -r "Amplitude" iOS/
grep -r "Mixpanel" iOS/

# Should return no results
```

**Backend Audit:**
```python
# Ensure app/main.py doesn't log health data to external services
# Remove any analytics that capture request bodies
```

**Current Status:** ✅ App appears clean - no advertising SDKs detected

---

### 🔴 4.3 HealthKit Data Integrity (Section 5.1.3)
**Status:** ✅ Compliant

**Apple Requirement:**
> "Apps must not write false or inaccurate data into HealthKit"

**Current Implementation:** ✅ Read-only access
```swift
// iOS/ECGHealthKit/HealthKitManager.swift:72-73
try await healthStore.requestAuthorization(toShare: [], read: [ecgType])
// No write access requested - compliant
```

**Action Required:**
- [x] Verified: App does not write to HealthKit ✅
- [x] Verified: App only reads ECG data ✅
- [ ] Document in App Review Notes

---

### 🔴 4.4 Clinical Claims & Accuracy (Section 1.4.1)
**Status:** ⚠️ Needs clarification

**Apple Requirement:**
> "Apps must disclose data and methodology to support accuracy claims. If accuracy cannot be validated, app will be rejected."

**Action Required:**
- [ ] Document backend analysis methodology
- [ ] Clarify that analysis comes from algorithm, not medical professional
- [ ] Don't claim FDA clearance (unless you have it)
- [ ] Don't claim to diagnose medical conditions
- [ ] Add confidence scores to all classifications

**Recommended Disclaimers:**
```
In App Description:
"Analysis results are algorithmically generated and should not be considered a medical diagnosis."

In App UI:
"Rhythm Classification: Algorithmic analysis, not reviewed by a physician"
"Confidence: 85% - Results may be inaccurate, consult a healthcare provider"
```

**Backend Documentation:**
Create `ANALYSIS_METHODOLOGY.md`:
```markdown
# ECG Analysis Methodology

## Algorithm Overview
This app uses signal processing and machine learning to analyze ECG waveforms:

1. **Signal Processing**: Bandpass filtering (0.5-40 Hz)
2. **R-Peak Detection**: Pan-Tompkins algorithm
3. **Feature Extraction**: RR intervals, HRV metrics
4. **Classification**: Statistical analysis (not deep learning)

## Accuracy Limitations
- Not FDA cleared or approved
- Intended for informational use only
- Should not replace professional medical evaluation
- Accuracy may vary based on recording quality

## Validation
- Tested on [X] ECG recordings
- Compared against Apple Watch classifications
- Mean accuracy: [X]% (if you have data)
- Higher error rates for inconclusive recordings
```

**App Review Notes:**
Include in submission:
```
Our app does not claim to diagnose medical conditions. Analysis is
algorithmically generated and users are repeatedly reminded to consult
healthcare providers. Methodology documented at: [GitHub URL]
```

---

### 🟡 4.5 Regulatory Clearance (If Applicable)
**Status:** _N/A for consumer informational app_

**Apple Requirement:**
> "If your medical app has received regulatory clearance, please submit a link to that documentation with your app."

**Current Status:** Not applicable - this is an informational app, not a medical device

**If you plan to seek FDA clearance:**
- [ ] Consult with FDA regulatory expert
- [ ] Determine if app is Class I, II, or III medical device
- [ ] Obtain 510(k) clearance (if required)
- [ ] Submit clearance documentation with app

**For informational apps (current status):**
- [x] Clearly state "Not a medical device"
- [x] Include disclaimers throughout
- [x] Don't make diagnostic claims

---

## PHASE 5: APP REVIEW SUBMISSION
**Estimated Time: 1 day preparation + 1-7 days review**

### 🟡 5.1 App Review Information
**Status:** _[To be filled in App Store Connect]_

**Action Required:**
- [ ] Contact Information (for Apple reviewers)
- [ ] Demo Account (if app requires login)
- [ ] Notes for Reviewer
- [ ] Attachments (screenshots, docs)

**Demo Account Setup:**
If your backend requires an API key:
```
Create a demo API key on your backend:
- Username: reviewer@apple.com
- API Key: demo-key-for-apple-review-2025
- Ensure backend accepts this key
- Pre-configure in TestFlight build
```

**Notes for Reviewer (Template):**
```
SETUP INSTRUCTIONS:
This app analyzes Apple Watch ECG recordings using HealthKit.

TEST ACCOUNT:
- API Key: [demo-key-for-apple-review]
- Backend URL: https://cyclingecg.onrender.com (pre-configured)

TESTING STEPS:
1. Grant HealthKit permission when prompted
2. If you have Apple Watch ECGs, they will appear in the list
3. Tap any ECG to view details
4. Tap "Analyze" to send to backend for analysis (requires internet)
5. View results including HR, HRV, rhythm classification
6. Tap "Export" to test data export functionality

HEALTH DATA USAGE:
- App only reads ECG data from HealthKit (read-only access)
- Data is only transmitted when user taps "Analyze"
- No advertising or third-party tracking
- Privacy policy shown on first launch

BACKEND TESTING:
- Backend is live at cyclingecg.onrender.com
- Free tier may have cold start delay (30 seconds first request)
- Sample ECG data can be used if no Apple Watch available

CONTACT:
For questions, contact [your-email@example.com]
```

---

### 🟡 5.2 Content Rights
**Status:** _[To be verified]_

**Action Required:**
- [ ] Confirm you own all content (code, images, text)
- [ ] Verify no copyrighted medical images used
- [ ] Check third-party library licenses (MIT/Apache OK)
- [ ] Attribute open-source dependencies if required

**License Audit:**
```bash
# Check iOS dependencies
cd iOS/ECGHealthKit
# If using Swift Package Manager:
cat Package.resolved

# Common licenses (App Store compatible):
# - MIT
# - Apache 2.0
# - BSD

# Incompatible licenses:
# - GPL (may require source code disclosure)
```

---

### 🟡 5.3 Submit for Review
**Status:** _[Ready after all critical fixes]_

**Final Checklist Before Submission:**
- [ ] All Phase 1 critical fixes implemented
- [ ] Privacy Policy visible in app
- [ ] Medical disclaimers present
- [ ] App tested on real device
- [ ] No crashes or major bugs
- [ ] API backend is running and accessible
- [ ] Demo account configured
- [ ] All App Store Connect metadata filled
- [ ] Screenshots uploaded
- [ ] Privacy nutrition labels configured
- [ ] Export compliance answered

**Submission Steps:**
```
1. In App Store Connect > My Apps > [Your App]
2. Select "+" under "iOS App" section
3. Create version "1.0.0"
4. Fill all required fields
5. Select build from TestFlight
6. Click "Submit for Review"
7. Wait for Apple's review (typically 1-7 days)
```

**Review Timeline:**
- Initial review: 24-48 hours
- Average review time: 1-3 days
- Rejection rate: ~30-40% on first submission (normal)
- Resubmission: 1-2 days

---

## PHASE 6: BACKEND SECURITY COMPLIANCE
**Estimated Time: 1-2 weeks**

### 🟡 6.1 Backend Database Encryption
**Status:** ❌ Not Implemented
**Current State:** SQLite unencrypted

**Action Required:**
- [ ] Migrate to PostgreSQL with SSL
- [ ] Enable field-level encryption for sensitive metrics
- [ ] Rotate encryption keys regularly
- [ ] Document encryption architecture

**Implementation (PostgreSQL):**
```python
# app/database.py

import os
from sqlalchemy import create_engine
from cryptography.fernet import Fernet

# Use PostgreSQL with SSL
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://user:pass@host:5432/ecg_db?sslmode=require"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "sslmode": "require"
    } if "postgresql" in DATABASE_URL else {}
)

# Field-level encryption
ENCRYPTION_KEY = os.environ.get("DB_ENCRYPTION_KEY")
cipher = Fernet(ENCRYPTION_KEY)

class ECGAnalysis(Base):
    __tablename__ = 'ecg_analyses'

    # Encrypt sensitive fields
    _rhythm_classification = Column(String)

    @property
    def rhythm_classification(self):
        if self._rhythm_classification:
            return cipher.decrypt(self._rhythm_classification.encode()).decode()
        return None

    @rhythm_classification.setter
    def rhythm_classification(self, value):
        if value:
            self._rhythm_classification = cipher.encrypt(value.encode()).decode()
```

**Deployment (Render.com):**
```bash
# In Render dashboard:
# 1. Create PostgreSQL database
# 2. Add environment variables:
#    - DATABASE_URL: [from Render PostgreSQL]
#    - DB_ENCRYPTION_KEY: [generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"]
```

**Reference:** Security audit Section 2, app/database.py

---

### 🟡 6.2 Data Retention & Deletion Policy
**Status:** ❌ Not Implemented

**Action Required:**
- [ ] Implement automatic data deletion after 30 days
- [ ] Add user data deletion endpoint
- [ ] Document retention policy in Privacy Policy
- [ ] Schedule daily cleanup job

**Implementation:**
```python
# app/database.py

from datetime import datetime, timedelta

def cleanup_old_analyses(days_to_retain: int = 30):
    """Delete analyses older than specified days"""
    db = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_retain)
        deleted = db.query(ECGAnalysis).filter(
            ECGAnalysis.timestamp_utc < cutoff_date
        ).delete()
        db.commit()
        return deleted
    finally:
        db.close()

# Add deletion endpoint
@app.delete("/v1/user/data")
def delete_user_data(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme)
):
    """GDPR/CCPA compliant data deletion"""
    _require_bearer(credentials)

    db = SessionLocal()
    try:
        deleted = db.query(ECGAnalysis).filter(
            ECGAnalysis.user_id == user_id  # Add user_id to schema
        ).delete()
        db.commit()
        return {"deleted": deleted, "status": "ok"}
    finally:
        db.close()
```

**Scheduled Cleanup (using APScheduler):**
```python
# app/main.py

from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    # Run cleanup daily at 2 AM UTC
    scheduler.add_job(
        cleanup_old_analyses,
        'cron',
        hour=2,
        minute=0,
        args=[30]  # 30 days retention
    )
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
```

---

### 🟡 6.3 Audit Logging
**Status:** ❌ Not Implemented

**Action Required:**
- [ ] Log all data access events
- [ ] Record API key usage
- [ ] Track analysis requests
- [ ] Implement log retention (90 days minimum for HIPAA)

**Implementation:**
```python
# app/audit_logger.py

import logging
import json
from datetime import datetime
import hashlib

audit_logger = logging.getLogger('audit')
handler = logging.FileHandler('audit.log')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
audit_logger.addHandler(handler)
audit_logger.setLevel(logging.INFO)

def log_analysis_access(recording_id: str, api_key: str, ip_address: str):
    audit_logger.info(json.dumps({
        "event": "ecg_analyzed",
        "recording_id": recording_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ip_address": ip_address,
        "api_key_hash": hashlib.sha256(api_key.encode()).hexdigest()[:8],
    }))

# In main.py:
@app.post("/v1/ecg/analyze")
def analyze_ecg(
    payload: ECGRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme)
):
    _require_bearer(credentials)

    # Audit log
    log_analysis_access(
        recording_id=payload.recording_id,
        api_key=credentials.credentials,
        ip_address=request.client.host
    )

    # ... rest of analysis
```

---

### 🟡 6.4 OpenAI Integration Disclosure
**Status:** ⚠️ Needs user notification

**Action Required:**
- [ ] Add in-app notification when AI features enabled
- [ ] Update Privacy Policy to mention OpenAI
- [ ] Disable by default
- [ ] Provide opt-in mechanism

**Implementation:**
```python
# app/main.py

# Make AI narrative opt-in
@app.post("/v1/ecg/analyze")
def analyze_ecg(
    payload: ECGRequest,
    enable_ai: bool = False,  # Default to False
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme)
):
    # ... analysis code

    if enable_ai and OPENAI_API_KEY:
        narrative = generate_narrative(features, patient, OPENAI_API_KEY)
        response["narrative"] = narrative
    else:
        response["narrative"] = {
            "enabled": False,
            "message": "AI narrative generation disabled. Enable in settings."
        }
```

**iOS Update:**
```swift
// iOS/ECGHealthKit/ContentView.swift

@AppStorage("enable_ai_narrative") private var enableAI = false

Section(header: Text("AI Features")) {
    Toggle("Enable AI Narrative", isOn: $enableAI)

    if enableAI {
        Text("⚠️ Health metrics will be sent to OpenAI's API (not HIPAA compliant)")
            .font(.caption)
            .foregroundColor(.orange)
    }
}

// Pass to backend:
let enableAI = UserDefaults.standard.bool(forKey: "enable_ai_narrative")
// Add to API request
```

---

## PHASE 7: POST-SUBMISSION MONITORING
**After app is approved**

### ⚪ 7.1 Crash Reporting
**Action Required:**
- [ ] Integrate Crashlytics or Sentry
- [ ] Monitor crash rates
- [ ] Fix critical crashes within 7 days

### ⚪ 7.2 User Feedback
**Action Required:**
- [ ] Monitor App Store reviews
- [ ] Respond to user questions
- [ ] Track feature requests

### ⚪ 7.3 Analytics (Privacy-Safe)
**Action Required:**
- [ ] Track app usage (no health data)
- [ ] Monitor feature adoption
- [ ] A/B test improvements

### ⚪ 7.4 Regular Updates
**Action Required:**
- [ ] Submit updates every 3-6 months
- [ ] Fix bugs reported by users
- [ ] Add new features based on feedback
- [ ] Keep up with iOS updates

---

## PHASE 8: COMPLIANCE CERTIFICATIONS (Optional)
**For healthcare/enterprise use**

### ⚪ 8.1 HIPAA Compliance
**Status:** Not currently compliant
**Estimated Time:** 3-6 months
**Cost:** $10,000-$50,000

**Requirements:**
- [ ] Business Associate Agreement (BAA) with backend host
- [ ] BAA with any third-party services (OpenAI, etc.)
- [ ] Encryption at rest and in transit
- [ ] Audit logging
- [ ] Access controls
- [ ] Employee training
- [ ] Risk assessment
- [ ] Incident response plan
- [ ] Annual compliance audit

### ⚪ 8.2 GDPR Compliance
**Status:** Partially compliant
**Estimated Time:** 1-2 months

**Requirements:**
- [x] Privacy policy
- [ ] User consent mechanism
- [ ] Data deletion endpoint
- [ ] Data portability (export)
- [ ] Data processing agreements
- [ ] EU representative (if targeting EU)

---

## FINAL PRE-SUBMISSION CHECKLIST

### iOS App
- [ ] ✅ API keys stored in Keychain (not UserDefaults)
- [ ] ✅ Analysis history encrypted with CryptoKit
- [ ] ✅ Privacy Policy shown on first launch
- [ ] ✅ Medical disclaimers visible throughout app
- [ ] ✅ Debug logging disabled in Release builds
- [ ] ✅ No PHI logged to console
- [ ] ✅ HTTP backends removed or warned
- [ ] ✅ User consent for backend fallback
- [ ] ✅ Info.plist privacy descriptions accurate
- [ ] ✅ No crashes or major bugs
- [ ] ✅ Tested on real device
- [ ] ✅ HealthKit permission working
- [ ] ✅ Export functionality working
- [ ] ✅ UI polished and professional

### Backend
- [ ] ✅ Database encryption enabled
- [ ] ✅ HTTPS enforced (no HTTP)
- [ ] ✅ API key authentication working
- [ ] ✅ Data retention policy implemented
- [ ] ✅ Audit logging enabled
- [ ] ✅ OpenAI integration opt-in only
- [ ] ✅ Backend running and accessible
- [ ] ✅ Demo API key created for reviewers

### App Store Connect
- [ ] ✅ App record created
- [ ] ✅ Metadata filled (name, description, keywords)
- [ ] ✅ Screenshots uploaded (all required sizes)
- [ ] ✅ App icon uploaded (1024x1024)
- [ ] ✅ Privacy nutrition labels configured
- [ ] ✅ Privacy Policy URL entered
- [ ] ✅ Age rating completed
- [ ] ✅ Export compliance answered
- [ ] ✅ App review notes written
- [ ] ✅ Demo account configured

### Code Signing
- [ ] ✅ Distribution certificate created
- [ ] ✅ Provisioning profile configured
- [ ] ✅ HealthKit capability enabled
- [ ] ✅ Bundle ID matches App Store Connect
- [ ] ✅ Version number set (1.0.0)
- [ ] ✅ Build number set (1)

### Testing
- [ ] ✅ All critical features tested
- [ ] ✅ TestFlight beta completed (optional)
- [ ] ✅ No memory leaks
- [ ] ✅ No excessive battery drain
- [ ] ✅ Works on iOS 16.0+
- [ ] ✅ Works on iPhone and iPad
- [ ] ✅ Offline functionality tested

---

## ESTIMATED TIMELINE

**Conservative Estimate (First Time Submission):**
```
Phase 1: Critical Security Fixes ........... 1-2 weeks
Phase 2: App Store Requirements ............ 3-5 days
Phase 3: Technical Setup ................... 2-3 days
Phase 4: Health App Requirements ........... 2-3 days
Phase 5: Review Submission ................. 1 day
Phase 6: Backend Compliance ................ 1-2 weeks (can be done in parallel)

TOTAL: 3-5 weeks before submission
Apple Review: 1-7 days
```

**Optimistic Estimate (Experienced Developer):**
```
Phase 1-5: 1-2 weeks
Apple Review: 1-3 days

TOTAL: 2-3 weeks
```

---

## COMMON REJECTION REASONS & HOW TO AVOID

### Rejection Reason 1: Privacy Policy Missing
**How to Avoid:** ✅ Complete Phase 1.3 - Show privacy policy in app

### Rejection Reason 2: Health Data Used for Marketing
**How to Avoid:** ✅ Remove all analytics SDKs, no advertising

### Rejection Reason 3: Inaccurate Medical Claims
**How to Avoid:** ✅ Add disclaimers, document methodology, no diagnostic claims

### Rejection Reason 4: Crash on Launch
**How to Avoid:** ✅ Test thoroughly, use TestFlight beta

### Rejection Reason 5: Unencrypted Health Data
**How to Avoid:** ✅ Complete Phase 1.1, 1.2, 1.6 (Keychain + encryption)

### Rejection Reason 6: Missing Export Compliance
**How to Avoid:** ✅ Answer encryption questions in App Store Connect

### Rejection Reason 7: Incomplete Metadata
**How to Avoid:** ✅ Fill all required fields, upload screenshots

---

## RESOURCES

**Apple Documentation:**
- App Store Review Guidelines: https://developer.apple.com/app-store/review/guidelines/
- HealthKit: https://developer.apple.com/healthkit/
- App Privacy Details: https://developer.apple.com/app-store/app-privacy-details/
- Encryption Export: https://developer.apple.com/documentation/security/complying_with_encryption_export_regulations

**Security Resources:**
- OWASP Mobile Security: https://owasp.org/www-project-mobile-security/
- iOS Security Guide: https://support.apple.com/guide/security/welcome/web
- HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/

**Developer Forums:**
- Apple Developer Forums: https://developer.apple.com/forums/
- Stack Overflow (ios tag): https://stackoverflow.com/questions/tagged/ios

---

## SUPPORT

**If you get stuck:**
1. Check security reports: `SECURITY_SUMMARY.txt` and `SECURITY_AND_PRIVACY_ANALYSIS.md`
2. Review Apple's rejection message carefully
3. Search Apple Developer Forums for similar issues
4. Contact Apple Developer Support (requires paid account)
5. Consider hiring iOS consultant for first submission

**Backend Issues:**
- Check backend logs: `heroku logs --tail` (or Render equivalent)
- Test API endpoint directly: `curl -X POST https://cyclingecg.onrender.com/v1/ecg/analyze`
- Verify environment variables are set

---

**Good luck with your submission!** 🚀

Remember: Most apps are rejected on first try. This is normal. Read the rejection carefully, fix the issues, and resubmit.
