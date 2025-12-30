//
//  ContentView.swift
//  ECGHealthKit
//
//  Main view for the ECG Insights app
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var healthKitManager: HealthKitManager
    @StateObject private var analysisService = ECGAnalysisService()
    @StateObject private var historyManager = AnalysisHistoryManager()
    @State private var showingSettings = false

    var body: some View {
        NavigationView {
            Group {
                if !healthKitManager.isAuthorized {
                    AuthorizationView()
                } else if healthKitManager.isLoading {
                    LoadingView()
                } else {
                    ECGListView()
                        .environmentObject(analysisService)
                        .environmentObject(historyManager)
                }
            }
            .navigationTitle("ECG Recordings")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        showingSettings = true
                    }) {
                        Image(systemName: "gear")
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        Task {
                            await healthKitManager.refresh()
                        }
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(healthKitManager.isLoading)
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView()
                    .environmentObject(analysisService)
            }
            .task {
                if healthKitManager.isAuthorized {
                    await healthKitManager.fetchECGRecordings()
                }
            }
        }
    }
}

// MARK: - Authorization View

struct AuthorizationView: View {
    @EnvironmentObject var healthKitManager: HealthKitManager

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "heart.text.square.fill")
                .font(.system(size: 80))
                .foregroundColor(.red)

            Text("HealthKit Authorization Required")
                .font(.title2)
                .fontWeight(.bold)

            Text("This app needs permission to access your ECG recordings from Apple Watch.")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal)

            if let error = healthKitManager.authorizationError {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Button(action: {
                Task {
                    await healthKitManager.requestAuthorization()
                }
            }) {
                Text("Grant Access")
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            .padding(.horizontal)
        }
        .padding()
    }
}

// MARK: - Loading View

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)

            Text("Loading ECG Recordings...")
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @EnvironmentObject var analysisService: ECGAnalysisService
    @Environment(\.dismiss) var dismiss

    @AppStorage("api_url") private var apiURL = "https://cyclingecg.onrender.com"
    @AppStorage("api_key") private var apiKey = ""
    @AppStorage("backend_preset") private var backendPreset = "cloud"

    @State private var customURL = ""
    @State private var showingInfo = false

    enum BackendPreset: String, CaseIterable {
        case cloud = "cloud"
        case local = "local"
        case custom = "custom"

        var displayName: String {
            switch self {
            case .cloud: return "Cloud (Render)"
            case .local: return "Local Network"
            case .custom: return "Custom"
            }
        }

        var defaultURL: String {
            switch self {
            case .cloud: return "https://cyclingecg.onrender.com"
            case .local: return "http://192.168.1.100:8000"
            case .custom: return ""
            }
        }

        var icon: String {
            switch self {
            case .cloud: return "cloud.fill"
            case .local: return "network"
            case .custom: return "pencil"
            }
        }

        var description: String {
            switch self {
            case .cloud: return "Hosted backend (may have cold start delay)"
            case .local: return "Backend on your local network"
            case .custom: return "Enter your own backend URL"
            }
        }
    }

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Backend Configuration")) {
                    // Preset selector
                    ForEach(BackendPreset.allCases, id: \.self) { preset in
                        Button(action: {
                            backendPreset = preset.rawValue
                            if preset != .custom {
                                apiURL = preset.defaultURL
                            }
                        }) {
                            HStack {
                                Image(systemName: preset.icon)
                                    .foregroundColor(backendPreset == preset.rawValue ? .blue : .gray)
                                    .frame(width: 30)

                                VStack(alignment: .leading, spacing: 4) {
                                    Text(preset.displayName)
                                        .foregroundColor(.primary)
                                        .fontWeight(backendPreset == preset.rawValue ? .semibold : .regular)

                                    Text(preset.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                Spacer()

                                if backendPreset == preset.rawValue {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }

                Section(header: Text("Backend Details")) {
                    if backendPreset == BackendPreset.custom.rawValue {
                        TextField("API URL", text: $apiURL)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                            .keyboardType(.URL)
                    } else {
                        HStack {
                            Text("URL")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(apiURL)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    SecureField("API Key (optional)", text: $apiKey)

                    HStack {
                        Text("Status")
                            .foregroundColor(.secondary)
                        Spacer()
                        ServerStatusIndicator(apiURL: apiURL)
                    }
                }

                if backendPreset == BackendPreset.local.rawValue {
                    Section(header: Text("Local Network Setup")) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("To use a local backend:")
                                .font(.caption)
                                .fontWeight(.semibold)

                            Text("1. Find your computer's IP address")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text("2. Start the backend server")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text("3. Update the IP in settings if needed")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Button(action: { showingInfo = true }) {
                                Text("View Setup Instructions")
                                    .font(.caption)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }

                Section(header: Text("About")) {
                    HStack {
                        Text("App Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                }

                Section {
                    Button(action: {
                        // Update the analysis service with new settings
                        analysisService.updateConfiguration(
                            baseURL: apiURL,
                            apiKey: apiKey.isEmpty ? nil : apiKey
                        )
                        dismiss()
                    }) {
                        HStack {
                            Spacer()
                            Text("Save Settings")
                                .fontWeight(.semibold)
                            Spacer()
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $showingInfo) {
                LocalBackendInfoView()
            }
        }
    }
}

// MARK: - Local Backend Info View

struct LocalBackendInfoView: View {
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Setting Up Local Backend")
                            .font(.title2)
                            .fontWeight(.bold)

                        Text("Follow these steps to connect to a backend running on your local network:")
                            .foregroundColor(.secondary)
                    }

                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            StepView(
                                number: 1,
                                title: "Find Your Computer's IP Address",
                                details: [
                                    "On Mac: System Settings → Network → Wi-Fi → Details → TCP/IP",
                                    "On Windows: ipconfig in Command Prompt",
                                    "Look for something like: 192.168.1.100"
                                ]
                            )
                        }
                    }

                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            StepView(
                                number: 2,
                                title: "Start the Backend Server",
                                details: [
                                    "Navigate to the project directory",
                                    "Run: uvicorn app.main:app --host 0.0.0.0 --port 8000",
                                    "Make sure firewall allows connections on port 8000"
                                ]
                            )
                        }
                    }

                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            StepView(
                                number: 3,
                                title: "Update IP Address in App",
                                details: [
                                    "Tap on 'Local Network' backend option",
                                    "The default URL uses 192.168.1.100",
                                    "Change it to match your computer's IP address",
                                    "Keep the port as :8000"
                                ]
                            )
                        }
                    }

                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: "lightbulb.fill")
                                    .foregroundColor(.yellow)
                                Text("Tips")
                                    .fontWeight(.semibold)
                            }

                            Text("• Both iPhone and computer must be on the same Wi-Fi network")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text("• Local backend is faster than cloud (no cold start)")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text("• Use Cloud backend when not on the same network")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Local Setup")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct StepView: View {
    let number: Int
    let title: String
    let details: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Text("\(number)")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .frame(width: 36, height: 36)
                    .background(Circle().fill(Color.blue))

                Text(title)
                    .font(.headline)
            }

            VStack(alignment: .leading, spacing: 4) {
                ForEach(details, id: \.self) { detail in
                    Text("• \(detail)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.leading, 48)
        }
    }
}

// MARK: - Server Status Indicator

struct ServerStatusIndicator: View {
    let apiURL: String
    @State private var isOnline = false
    @State private var isChecking = true

    var body: some View {
        HStack {
            if isChecking {
                ProgressView()
                    .scaleEffect(0.7)
            } else {
                Circle()
                    .fill(isOnline ? Color.green : Color.red)
                    .frame(width: 8, height: 8)

                Text(isOnline ? "Online" : "Offline")
                    .foregroundColor(.secondary)
            }
        }
        .task {
            await checkServerStatus()
        }
    }

    private func checkServerStatus() async {
        isChecking = true

        let service = ECGAnalysisService(baseURL: apiURL)
        isOnline = await service.checkServerHealth()

        isChecking = false
    }
}
