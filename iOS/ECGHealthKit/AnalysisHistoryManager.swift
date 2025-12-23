//
//  AnalysisHistoryManager.swift
//  ECG Export Analysis
//
//  Manages persistent storage and retrieval of ECG analysis results with encryption
//

import Foundation
import CryptoKit

struct AnalysisHistoryItem: Codable, Identifiable {
    let id: String
    let recordingDate: Date
    let analysis: ECGAnalysisResponse

    init(id: String = UUID().uuidString, recordingDate: Date, analysis: ECGAnalysisResponse) {
        self.id = id
        self.recordingDate = recordingDate
        self.analysis = analysis
    }
}

@MainActor
class AnalysisHistoryManager: ObservableObject {
    @Published var historyItems: [AnalysisHistoryItem] = []

    private let storageKey = "ecg_analysis_history"
    private let fileURL: URL

    init() {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        fileURL = documentsDirectory.appendingPathComponent("ecg_analysis_history.json")
        loadHistory()
    }

    // MARK: - Storage

    func saveAnalysis(_ analysis: ECGAnalysisResponse, recordingDate: Date) {
        // Check if analysis for this recording already exists
        if let existingIndex = historyItems.firstIndex(where: { $0.analysis.recording_id == analysis.recording_id }) {
            // Update existing entry
            historyItems[existingIndex] = AnalysisHistoryItem(
                id: historyItems[existingIndex].id,
                recordingDate: recordingDate,
                analysis: analysis
            )
        } else {
            // Add new entry
            let item = AnalysisHistoryItem(recordingDate: recordingDate, analysis: analysis)
            historyItems.append(item)
        }

        // Sort by date (most recent first)
        historyItems.sort { $0.recordingDate > $1.recordingDate }

        // Keep only last 90 days
        let ninetyDaysAgo = Calendar.current.date(byAdding: .day, value: -90, to: Date())!
        historyItems = historyItems.filter { $0.recordingDate >= ninetyDaysAgo }

        persistHistory()
    }

    private func persistHistory() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let jsonData = try encoder.encode(historyItems)

            // Encrypt the data before writing to disk
            let encryptedData = try encryptData(jsonData)
            try encryptedData.write(to: fileURL)

            #if DEBUG
            print("[STORAGE] Successfully saved \(historyItems.count) encrypted analysis history items")
            #endif
        } catch {
            #if DEBUG
            print("[STORAGE] Error saving analysis history: \(error)")
            #endif
        }
    }

    private func loadHistory() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            #if DEBUG
            print("[STORAGE] No existing analysis history file found")
            #endif
            return
        }

        do {
            let encryptedData = try Data(contentsOf: fileURL)

            // Try to decrypt the data (handles encrypted files)
            let jsonData: Data
            do {
                jsonData = try decryptData(encryptedData)
            } catch {
                // If decryption fails, try to load as plaintext (migration from old format)
                #if DEBUG
                print("[STORAGE] Decryption failed, attempting to load as plaintext for migration")
                #endif
                jsonData = encryptedData
            }

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            historyItems = try decoder.decode([AnalysisHistoryItem].self, from: jsonData)

            // Filter to last 90 days
            let ninetyDaysAgo = Calendar.current.date(byAdding: .day, value: -90, to: Date())!
            historyItems = historyItems.filter { $0.recordingDate >= ninetyDaysAgo }

            // If we successfully loaded plaintext data, re-save as encrypted
            if encryptedData == jsonData {
                #if DEBUG
                print("[STORAGE] Migrating plaintext history to encrypted format")
                #endif
                persistHistory()
            }

            #if DEBUG
            print("[STORAGE] Successfully loaded \(historyItems.count) encrypted analysis history items")
            #endif
        } catch {
            #if DEBUG
            print("[STORAGE] Error loading analysis history: \(error)")
            #endif
            historyItems = []
        }
    }

    // MARK: - Encryption

    /// Encrypt data using AES-GCM with a key from Keychain
    private func encryptData(_ data: Data) throws -> Data {
        // Get or create encryption key from Keychain
        let encryptionKey = try KeychainManager.shared.getOrCreateEncryptionKey()
        let symmetricKey = SymmetricKey(data: encryptionKey)

        // Encrypt using AES-GCM
        let sealedBox = try AES.GCM.seal(data, using: symmetricKey)

        // Return the combined nonce + ciphertext + tag
        guard let combined = sealedBox.combined else {
            throw EncryptionError.encryptionFailed
        }

        return combined
    }

    /// Decrypt data using AES-GCM with a key from Keychain
    private func decryptData(_ encryptedData: Data) throws -> Data {
        // Get encryption key from Keychain
        let encryptionKey = try KeychainManager.shared.getOrCreateEncryptionKey()
        let symmetricKey = SymmetricKey(data: encryptionKey)

        // Decrypt using AES-GCM
        let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
        let decryptedData = try AES.GCM.open(sealedBox, using: symmetricKey)

        return decryptedData
    }

    enum EncryptionError: Error {
        case encryptionFailed
        case decryptionFailed
        case keyGenerationFailed

        var localizedDescription: String {
            switch self {
            case .encryptionFailed:
                return "Failed to encrypt data"
            case .decryptionFailed:
                return "Failed to decrypt data"
            case .keyGenerationFailed:
                return "Failed to generate encryption key"
            }
        }
    }

    // MARK: - Query Methods

    func getHistory(for days: Int = 30) -> [AnalysisHistoryItem] {
        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        return historyItems.filter { $0.recordingDate >= startDate }
            .sorted { $0.recordingDate < $1.recordingDate } // Oldest first for charting
    }

    func getMetricHistory(metric: MetricType, days: Int = 30) -> [(date: Date, value: Double)] {
        let items = getHistory(for: days)
        return items.compactMap { item in
            if let value = extractMetricValue(from: item.analysis, metric: metric) {
                return (date: item.recordingDate, value: value)
            }
            return nil
        }
    }

    private func extractMetricValue(from analysis: ECGAnalysisResponse, metric: MetricType) -> Double? {
        switch metric {
        case .heartRateMean:
            return analysis.features.heart_rate_bpm?.mean
        case .hrvSDNN:
            return analysis.features.hrv?.sdnn_ms
        case .hrvRMSSD:
            return analysis.features.hrv?.rmssd_ms
        case .pWaveDuration:
            return analysis.features.intervals?.p_wave_duration_ms
        case .prInterval:
            return analysis.features.intervals?.pr_interval_ms
        case .qrsDuration:
            return analysis.features.intervals?.qrs_duration_ms
        case .stSegment:
            return analysis.features.intervals?.st_segment_ms
        case .tWaveDuration:
            return analysis.features.intervals?.t_wave_duration_ms
        case .qtInterval:
            return analysis.features.intervals?.qt_interval_ms
        case .qtcInterval:
            return analysis.features.intervals?.qtc_ms
        }
    }
}

// MARK: - Metric Types

enum MetricType: String, CaseIterable {
    case heartRateMean = "Mean Heart Rate"
    case hrvSDNN = "HRV SDNN"
    case hrvRMSSD = "HRV RMSSD"
    case pWaveDuration = "P Wave Duration"
    case prInterval = "PR Interval"
    case qrsDuration = "QRS Duration"
    case stSegment = "ST Segment"
    case tWaveDuration = "T Wave Duration"
    case qtInterval = "QT Interval"
    case qtcInterval = "QTc Interval"

    var unit: String {
        switch self {
        case .heartRateMean:
            return "bpm"
        default:
            return "ms"
        }
    }

    var icon: String {
        switch self {
        case .heartRateMean:
            return "heart.fill"
        case .hrvSDNN, .hrvRMSSD:
            return "waveform"
        default:
            return "waveform.path.ecg"
        }
    }
}
