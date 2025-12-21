//
//  HealthKitManager.swift
//  ECGHealthKit
//
//  Manages HealthKit permissions and ECG data extraction
//

import Foundation
import HealthKit

@MainActor
final class HealthKitManager: ObservableObject {
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
            authorizationError = "HealthKit is not available on this device"
            return
        }

        let ecgType = HKObjectType.electrocardiogramType()

        do {
            try await healthStore.requestAuthorization(toShare: [], read: [ecgType])
            isAuthorized = true
            authorizationError = nil
        } catch {
            authorizationError = "Failed to authorize HealthKit: \(error.localizedDescription)"
            isAuthorized = false
        }
    }

    // MARK: - Fetch ECG Recordings

    func fetchECGRecordings() async {
        isLoading = true

        do {
            let ecgSamples = try await queryECGSamples()
            await processECGSamples(ecgSamples)
        } catch {
            authorizationError = "Error fetching ECGs: \(error.localizedDescription)"
            isLoading = false
        }
    }

    nonisolated private func queryECGSamples() async throws -> [HKElectrocardiogram] {
        return try await withCheckedThrowingContinuation { continuation in
            let ecgType = HKObjectType.electrocardiogramType()
            let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)

            // Filter for most recent 30 days
            let thirtyDaysAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date())!
            let datePredicate = HKQuery.predicateForSamples(withStart: thirtyDaysAgo, end: nil, options: .strictStartDate)

            let query = HKSampleQuery(
                sampleType: ecgType,
                predicate: datePredicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: [sortDescriptor]
            ) { query, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let ecgSamples = samples as? [HKElectrocardiogram] else {
                    continuation.resume(returning: [])
                    return
                }

                continuation.resume(returning: ecgSamples)
            }

            healthStore.execute(query)
        }
    }

    private func processECGSamples(_ ecgSamples: [HKElectrocardiogram]) async {
        // Extract recordings sequentially
        var recordings: [ECGRecording] = []
        for ecgSample in ecgSamples {
            if let recording = await extractECGData(from: ecgSample) {
                recordings.append(recording)
            }
        }

        self.ecgRecordings = recordings
        self.isLoading = false
    }

    // MARK: - Extract ECG Data

    nonisolated private func extractECGData(from ecg: HKElectrocardiogram) async -> ECGRecording? {
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

                        // Log first few samples to verify real ECG data
                        if voltageMeasurements.count <= 5 || voltageMeasurements.count % 1000 == 0 {
                            print("[HEALTHKIT] Sample \(voltageMeasurements.count): \(voltage) V")
                        }
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

                    // Calculate statistics to detect test data
                    if !voltageMeasurements.isEmpty {
                        let minVoltage = voltageMeasurements.min() ?? 0
                        let maxVoltage = voltageMeasurements.max() ?? 0
                        let meanVoltage = voltageMeasurements.reduce(0.0, +) / Double(voltageMeasurements.count)
                        print("[HEALTHKIT] Voltage stats (Volts):")
                        print("  - Min: \(minVoltage)")
                        print("  - Max: \(maxVoltage)")
                        print("  - Mean: \(meanVoltage)")
                        print("  - First 5 samples: \(voltageMeasurements.prefix(5))")
                        print("  - Last 5 samples: \(voltageMeasurements.suffix(5))")

                        // Check for linear ramp pattern (test data indicator)
                        if voltageMeasurements.count >= 10 {
                            let diffs = zip(voltageMeasurements.dropFirst(), voltageMeasurements).map { $0.0 - $0.1 }
                            let avgDiff = diffs.reduce(0.0, +) / Double(diffs.count)
                            let diffStd = sqrt(diffs.map { pow($0 - avgDiff, 2) }.reduce(0.0, +) / Double(diffs.count))
                            print("  - Average increment: \(avgDiff) V")
                            print("  - Increment std dev: \(diffStd) V")
                            if diffStd < abs(avgDiff) * 0.01 {
                                print("  ⚠️ WARNING: Data appears to be a linear ramp (test data?)")
                            }
                        }
                    }

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
