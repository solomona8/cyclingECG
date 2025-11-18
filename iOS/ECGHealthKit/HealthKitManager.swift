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
        ) { [weak self] query, samples, error in

            guard let self = self else { return }

            if let error = error {
                Task { @MainActor in
                    self.authorizationError = "Error fetching ECGs: \(error.localizedDescription)"
                    self.isLoading = false
                }
                return
            }

            guard let ecgSamples = samples as? [HKElectrocardiogram] else {
                Task { @MainActor in
                    self.isLoading = false
                }
                return
            }

            Task {
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

            let query = HKElectrocardiogramQuery(ecg) { query, result in
                switch result {
                case .measurement(let measurement):
                    // Extract voltage measurements
                    if let voltageQuantity = measurement.quantity(for: .appleWatchSimilarToLeadI) {
                        let voltage = voltageQuantity.doubleValue(for: HKUnit.volt())
                        voltageMeasurements.append(voltage)
                    }

                case .done:
                    // Calculate sampling frequency
                    if voltageMeasurements.count > 0 {
                        let duration = ecg.endDate.timeIntervalSince(ecg.startDate)
                        samplingFrequency = Double(voltageMeasurements.count) / duration
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
