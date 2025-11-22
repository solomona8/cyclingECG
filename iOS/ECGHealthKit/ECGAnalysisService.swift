//
//  ECGAnalysisService.swift
//  ECGHealthKit
//
//  Service for communicating with the ECG analysis backend
//

import Foundation

class ECGAnalysisService: ObservableObject {
    @Published var analysisResults: [String: ECGAnalysisResponse] = [:]
    @Published var isAnalyzing = false
    @Published var analysisError: String?

    private let baseURL: String
    private let apiKey: String?

    init(baseURL: String = "http://localhost:8000", apiKey: String? = nil) {
        self.baseURL = baseURL
        self.apiKey = apiKey
    }

    // MARK: - Analyze ECG

    func analyzeECG(_ recording: ECGRecording) async -> ECGAnalysisResponse? {
        await MainActor.run {
            isAnalyzing = true
            analysisError = nil
        }

        let apiRequest = recording.toAPIRequest(apiURL: baseURL)

        guard let url = URL(string: "\(baseURL)/v1/ecg/analyze") else {
            await MainActor.run {
                analysisError = "Invalid URL"
                isAnalyzing = false
            }
            return nil
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let apiKey = apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            request.httpBody = try encoder.encode(apiRequest)

            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                await MainActor.run {
                    analysisError = "Invalid response from server"
                    isAnalyzing = false
                }
                return nil
            }

            // DEBUG: Print raw response
            print("=== SERVER RESPONSE DEBUG ===")
            print("Status code: \(httpResponse.statusCode)")
            print("Content-Type: \(httpResponse.value(forHTTPHeaderField: "Content-Type") ?? "none")")
            print("Data length: \(data.count) bytes")
            if let rawJSON = String(data: data, encoding: .utf8) {
                print("Raw JSON response:")
                print(rawJSON)
            } else {
                print("ERROR: Could not decode data as UTF-8 string")
            }
            print("=== END DEBUG ===")

            guard httpResponse.statusCode == 200 else {
                let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
                await MainActor.run {
                    analysisError = "Server error (\(httpResponse.statusCode)): \(errorMessage)"
                    isAnalyzing = false
                }
                return nil
            }

            let decoder = JSONDecoder()
            let analysisResponse = try decoder.decode(ECGAnalysisResponse.self, from: data)

            await MainActor.run {
                analysisResults[recording.id] = analysisResponse
                isAnalyzing = false
            }

            return analysisResponse

        } catch {
            await MainActor.run {
                analysisError = "Analysis failed: \(error.localizedDescription)"
                isAnalyzing = false
            }
            return nil
        }
    }

    // MARK: - Batch Analysis

    func analyzeMultipleECGs(_ recordings: [ECGRecording]) async -> [String: ECGAnalysisResponse] {
        var results: [String: ECGAnalysisResponse] = [:]

        for recording in recordings {
            if let result = await analyzeECG(recording) {
                results[recording.id] = result
            }
        }

        return results
    }

    // MARK: - Health Check

    func checkServerHealth() async -> Bool {
        guard let url = URL(string: "\(baseURL)/health") else {
            return false
        }

        do {
            let (_, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse else {
                return false
            }

            return httpResponse.statusCode == 200

        } catch {
            return false
        }
    }

    // MARK: - Get Analysis Result

    func getAnalysisResult(for recordingID: String) -> ECGAnalysisResponse? {
        return analysisResults[recordingID]
    }
}
