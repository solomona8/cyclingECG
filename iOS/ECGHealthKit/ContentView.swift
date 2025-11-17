//
//  ContentView.swift
//  ECGHealthKit
//
//  Main view for the ECG HealthKit app
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var healthKitManager: HealthKitManager
    @StateObject private var analysisService = ECGAnalysisService()
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

    @AppStorage("api_url") private var apiURL = "http://localhost:8000"
    @AppStorage("api_key") private var apiKey = ""

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Backend API Settings")) {
                    TextField("API URL", text: $apiURL)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .keyboardType(.URL)

                    SecureField("API Key (optional)", text: $apiKey)
                }

                Section(header: Text("About")) {
                    HStack {
                        Text("App Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }

                    HStack {
                        Text("Backend Status")
                        Spacer()
                        ServerStatusIndicator(apiURL: apiURL)
                    }
                }

                Section {
                    Button("Save Settings") {
                        // Update the analysis service with new settings
                        dismiss()
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
