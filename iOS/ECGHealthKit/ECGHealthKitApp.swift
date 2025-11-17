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

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(healthKitManager)
        }
    }
}
