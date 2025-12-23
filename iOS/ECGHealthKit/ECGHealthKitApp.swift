//
//  ECGHealthKitApp.swift
//  ECGHealthKit
//
//  Apple Watch ECG Data Extractor
//  Extracts, interprets, and exports Apple Watch ECG recordings from HealthKit
//

import SwiftUI

@main
struct ECGHealthKitApp: App {
    @StateObject private var healthKitManager = HealthKitManager()
    @AppStorage("has_accepted_privacy_policy") private var hasAcceptedPolicy = false
    @State private var showPrivacyPolicy = false

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(healthKitManager)
                .sheet(isPresented: $showPrivacyPolicy) {
                    PrivacyPolicyView(hasAcceptedPolicy: $hasAcceptedPolicy, isFirstLaunch: true)
                }
                .onAppear {
                    // Show privacy policy on first launch
                    if !hasAcceptedPolicy {
                        showPrivacyPolicy = true
                    }
                }
        }
    }
}
