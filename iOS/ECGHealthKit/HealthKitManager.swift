//
//  HealthKitManager.swift
//  ECGHealthKit
//
//  Manages HealthKit permissions and ECG data extraction
//

import Foundation
import HealthKit

class HealthKitManager: ObservableObject {
    private let healthStore = HKHealthStore()

    @Published var ecgRecordings: [ECGRecording] = []
    @Published var isAuthorized = false
    @Published var authorizationError: String?
    @Published var isLoading = false

    init() {
        checkAuthorization()
    }

    // MARK: - Authorization

    func checkAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            authorizationError = "HealthKit is not available on this device"
            return
        }

        let ecgType = HKObjectType.electrocardiogramType()

        let authStatus = healthStore.authorizationStatus(for: ecgType)
        isAuthorized = authStatus == .sharingAuthorized
    }

    func requestAuthorization() async {
        guard HKHealthStore.isHealthDataAvailable() else {
            await MainActor.run {
                authorizationError = "HealthKit is not available on this device"
            }
            return
        }

        let ecgType = HKObjectType.electrocardiogramType()

        do {
            try await healthStore.requestAuthorization(toShare: [], read: [ecgType])
            await MainActor.run {
                isAuthorized = true
                authorizationError = nil
            }
        } catch {
            await MainActor.run {
                authorizationError = "Failed to authorize HealthKit: \(error.localizedDescription)"
                isAuthorized = false
            }
        }
    }

    // MARK: - Fetch ECG Recordings

    func fetchECGRecordings() async {
        await MainActor.run {
            isLoading = true
        }

        let ecgType = HKObjectType.electrocardiogramType()
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)

        let query = HKSampleQuery(
            sampleType: ecgType,
            predicate: nil,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [sortDescriptor]
        ) { query, samples, error in

            if let error = error {
                Task { @MainActor [weak self] in
                    self?.authorizationError = "Error fetching ECGs: \(error.localizedDescription)"
                    self?.isLoading = false
                }
                return
            }

            guard let ecgSamples = samples as? [HKElectrocardiogram] else {
                Task { @MainActor [weak self] in
                    self?.isLoading = false
                }
                return
            }

            Task { [weak self] in
                guard let self = self else { return }

                var recordings: [ECGRecording] = []

                for ecgSample in ecgSamples {
                    if let recording = await self.extractECGData(from: ecgSample) {
                        recordings.append(recording)
                    }
                }

                await MainActor.run {
                    self.ecgRecordings = recordings
                    self.isLoading = false
                }
            }
        }

        healthStore.execute(query)
    }

    // MARK: - Extract ECG Data

    private func extractECGData(from ecg: HKElectrocardiogram) async -> ECGRecording? {
        return await withCheckedContinuation { continuation in
            var voltageMeasurements: [Double] = []
            var samplingFrequency: Double = 0

            print("=== EXTRACTING ECG DATA ===")
            print("ECG ID: \(ecg.uuid.uuidString)")
            print("Start date: \(ecg.startDate)")
            print("End date: \(ecg.endDate)")
            print("Number of voltage measurements (metadata): \(ecg.numberOfVoltageMeasurements)")

            let query = HKElectrocardiogramQuery(ecg) { query, result in
                switch result {
                case .measurement(let measurement):
                    // Extract voltage measurements
                    if let voltageQuantity = measurement.quantity(for: .appleWatchSimilarToLeadI) {
                        let voltage = voltageQuantity.doubleValue(for: HKUnit.volt())
                        voltageMeasurements.append(voltage)
                    } else {
                        // This is important - if we can't get voltage, log it
                        print("WARNING: Could not extract voltage from measurement")
                    }

                case .done:
                    // Calculate sampling frequency
                    if voltageMeasurements.count > 0 {
                        let duration = ecg.endDate.timeIntervalSince(ecg.startDate)
                        samplingFrequency = Double(voltageMeasurements.count) / duration
                    }

                    print("Extraction complete:")
                    print("- Extracted \(voltageMeasurements.count) voltage measurements")
                    print("- Expected \(ecg.numberOfVoltageMeasurements) measurements")
                    print("- Sampling frequency: \(samplingFrequency) Hz")
                    print("=== END EXTRACTION ===")

                    if voltageMeasurements.isEmpty {
                        print("ERROR: No voltage measurements were extracted!")
                    } else if voltageMeasurements.count < ecg.numberOfVoltageMeasurements {
                        print("WARNING: Extracted fewer measurements than expected")
                    }

                    // Create ECG recording
                    let recording = ECGRecording(
                        id: ecg.uuid.uuidString,
                        startDate: ecg.startDate,
                        endDate: ecg.endDate,
                        classification: ecg.classification,
                        symptomsStatus: ecg.symptomsStatus,
                        averageHeartRate: ecg.averageHeartRate,
                        samplingFrequency: samplingFrequency,
                        voltageMeasurements: voltageMeasurements,
                        numberOfVoltageMeasurements: ecg.numberOfVoltageMeasurements
                    )

                    continuation.resume(returning: recording)

                case .error(let error):
                    print("Error extracting ECG data: \(error.localizedDescription)")
                    continuation.resume(returning: nil)

                @unknown default:
                    continuation.resume(returning: nil)
                }
            }

            healthStore.execute(query)
        }
    }

    // MARK: - Refresh Data

    func refresh() async {
        await fetchECGRecordings()
    }
}
