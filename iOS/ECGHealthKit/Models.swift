//
//  Models.swift
//  ECGHealthKit
//
//  Data models for ECG recordings and analysis
//

import Foundation
import HealthKit

// MARK: - ECG Recording

struct ECGRecording: Identifiable {
    let id: String
    let startDate: Date
    let endDate: Date
    let classification: HKElectrocardiogram.Classification
    let symptomsStatus: HKElectrocardiogram.SymptomsStatus
    let averageHeartRate: HKQuantity?
    let samplingFrequency: Double
    let voltageMeasurements: [Double]
    let numberOfVoltageMeasurements: Int

    var duration: TimeInterval {
        endDate.timeIntervalSince(startDate)
    }

    var averageHeartRateBPM: Double? {
        averageHeartRate?.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
    }

    var classificationDescription: String {
        switch classification {
        case .notSet:
            return "Not Set"
        case .sinusRhythm:
            return "Sinus Rhythm"
        case .atrialFibrillation:
            return "Atrial Fibrillation"
        case .inconclusiveLowHeartRate:
            return "Inconclusive - Low Heart Rate"
        case .inconclusiveHighHeartRate:
            return "Inconclusive - High Heart Rate"
        case .inconclusivePoorReading:
            return "Inconclusive - Poor Reading"
        case .inconclusiveOther:
            return "Inconclusive - Other"
        case .unrecognized:
            return "Unrecognized"
        @unknown default:
            return "Unknown"
        }
    }

    var symptomsDescription: String {
        switch symptomsStatus {
        case .notSet:
            return "Not Set"
        case .none:
            return "No Symptoms"
        case .present:
            return "Symptoms Present"
        @unknown default:
            return "Unknown"
        }
    }

    // Convert to backend API format
    func toAPIRequest(apiURL: String) -> ECGAnalysisRequest {
        // Convert voltage measurements from volts to microvolts
        let samplesInMicrovolts = voltageMeasurements.map { $0 * 1_000_000 }

        return ECGAnalysisRequest(
            recording_id: id,
            samples: samplesInMicrovolts,
            sampling_rate_hz: Int(samplingFrequency),
            units: "uV",
            lead: "I",
            start_timestamp_utc: ISO8601DateFormatter().string(from: startDate),
            device_info: DeviceInfo(
                manufacturer: "Apple",
                model: "Apple Watch",
                software_version: nil
            ),
            context: RecordingContext(
                symptoms: symptomsStatus == .present ? ["User reported symptoms"] : nil,
                activity: nil,
                position: nil
            )
        )
    }
}

// MARK: - ECGRecording Codable Conformance

extension ECGRecording: Codable {
    enum CodingKeys: String, CodingKey {
        case id
        case startDate
        case endDate
        case classification
        case symptomsStatus
        case averageHeartRate
        case samplingFrequency
        case voltageMeasurements
        case numberOfVoltageMeasurements
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let decodedId = try container.decode(String.self, forKey: .id)
        let decodedStartDate = try container.decode(Date.self, forKey: .startDate)
        let decodedEndDate = try container.decode(Date.self, forKey: .endDate)

        // Decode classification as raw value
        let classificationRawValue = try container.decode(Int.self, forKey: .classification)
        let decodedClassification = HKElectrocardiogram.Classification(rawValue: classificationRawValue) ?? .notSet

        // Decode symptoms status as raw value
        let symptomsRawValue = try container.decode(Int.self, forKey: .symptomsStatus)
        let decodedSymptomsStatus = HKElectrocardiogram.SymptomsStatus(rawValue: symptomsRawValue) ?? .notSet

        // Handle HKQuantity (decode as optional double in BPM)
        let decodedAverageHeartRate: HKQuantity?
        if let heartRateBPM = try container.decodeIfPresent(Double.self, forKey: .averageHeartRate) {
            decodedAverageHeartRate = HKQuantity(unit: HKUnit.count().unitDivided(by: .minute()), doubleValue: heartRateBPM)
        } else {
            decodedAverageHeartRate = nil
        }

        let decodedSamplingFrequency = try container.decode(Double.self, forKey: .samplingFrequency)
        let decodedVoltageMeasurements = try container.decode([Double].self, forKey: .voltageMeasurements)
        let decodedNumberOfVoltageMeasurements = try container.decode(Int.self, forKey: .numberOfVoltageMeasurements)

        // Initialize all stored properties
        self.id = decodedId
        self.startDate = decodedStartDate
        self.endDate = decodedEndDate
        self.classification = decodedClassification
        self.symptomsStatus = decodedSymptomsStatus
        self.averageHeartRate = decodedAverageHeartRate
        self.samplingFrequency = decodedSamplingFrequency
        self.voltageMeasurements = decodedVoltageMeasurements
        self.numberOfVoltageMeasurements = decodedNumberOfVoltageMeasurements
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(id, forKey: .id)
        try container.encode(startDate, forKey: .startDate)
        try container.encode(endDate, forKey: .endDate)

        // Encode classification as raw value
        try container.encode(classification.rawValue, forKey: .classification)

        // Encode symptoms status as raw value
        try container.encode(symptomsStatus.rawValue, forKey: .symptomsStatus)

        // Encode HKQuantity as double in BPM
        if let heartRate = averageHeartRate {
            let bpm = heartRate.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            try container.encode(bpm, forKey: .averageHeartRate)
        } else {
            try container.encodeNil(forKey: .averageHeartRate)
        }

        try container.encode(samplingFrequency, forKey: .samplingFrequency)
        try container.encode(voltageMeasurements, forKey: .voltageMeasurements)
        try container.encode(numberOfVoltageMeasurements, forKey: .numberOfVoltageMeasurements)
    }
}

// MARK: - API Request Models

struct ECGAnalysisRequest: Codable {
    let recording_id: String
    let samples: [Double]
    let sampling_rate_hz: Int
    let units: String
    let lead: String
    let start_timestamp_utc: String
    let device_info: DeviceInfo?
    let context: RecordingContext?
}

struct DeviceInfo: Codable {
    let manufacturer: String
    let model: String
    let software_version: String?
}

struct RecordingContext: Codable {
    let symptoms: [String]?
    let activity: String?
    let position: String?
}

// MARK: - API Response Models

struct ECGAnalysisResponse: Codable {
    let recording_id: String
    let timestamp_utc: String
    let features: ECGFeatures
    let narrative: ECGNarrative?
    let analyzer_version: String
}

struct ECGFeatures: Codable {
    let rhythm_classification: String?
    let rhythm_confidence: Double?
    let heart_rate_bpm: HeartRateInfo?
    let rr_intervals_ms: RRIntervalInfo?
    let hrv: HRVInfo?
    let intervals: IntervalInfo?
    let signal_quality: SignalQuality?
    let morphology: MorphologyInfo?
}

struct HeartRateInfo: Codable {
    let mean: Double?
    let min: Double?
    let max: Double?
}

struct RRIntervalInfo: Codable {
    let mean: Double?
    let min: Double?
    let max: Double?
    let coefficient_of_variation: Double?
}

struct HRVInfo: Codable {
    let sdnn_ms: Double?
    let rmssd_ms: Double?
}

struct IntervalInfo: Codable {
    let qrs_duration_ms: Double?
    let qt_interval_ms: Double?
    let qtc_ms: Double?
}

struct SignalQuality: Codable {
    let overall_quality: String?
    let artifact_burden_percent: Double?
}

struct MorphologyInfo: Codable {
    let ectopy_burden_percent: Double?
}

struct ECGNarrative: Codable {
    let patient_summary: String?
    let clinician_notes: String?
    let safety_flags: [String]?
}
