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
    @Published var showFallbackConsentAlert = false
    @Published var fallbackConsentGranted = false

    private var baseURL: String
    private var apiKey: String?
    private var fallbackURLs: [String] = []
    private var pendingRecording: ECGRecording?
    var fallbackURL: String?

    init(baseURL: String = "https://cyclingecg.onrender.com", apiKey: String? = nil) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        setupFallbackURLs()
    }

    // MARK: - Configuration Update

    func updateConfiguration(baseURL: String, apiKey: String?) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        setupFallbackURLs()
    }

    private func setupFallbackURLs() {
        fallbackURLs = []
        // Add fallback URLs based on current backend
        if baseURL.contains("localhost") || baseURL.contains("192.168") || baseURL.contains("127.0.0.1") {
            // If local, fallback to cloud
            fallbackURLs.append("https://cyclingecg.onrender.com")
        } else if baseURL.contains("cyclingecg.onrender.com") {
            // If cloud, no fallback (or could add alternative cloud URLs)
        }
    }

    // MARK: - Analyze ECG

    func analyzeECG(_ recording: ECGRecording) async -> ECGAnalysisResponse? {
        await MainActor.run {
            isAnalyzing = true
            analysisError = nil
        }

        // Try primary backend first
        if let result = await analyzeECGWithURL(recording, url: baseURL) {
            return result
        }

        // If primary failed and we have fallbacks, request user consent
        if !fallbackURLs.isEmpty {
            #if DEBUG
            print("Primary backend failed, requesting fallback consent")
            #endif

            // Request user consent for fallback
            await MainActor.run {
                pendingRecording = recording
                fallbackURL = fallbackURLs.first
                showFallbackConsentAlert = true
                analysisError = "Primary backend unavailable. Requesting permission to use fallback..."
            }

            // Wait for user decision (the UI will call proceedWithFallback or cancelFallback)
            return nil
        }

        // No fallbacks available
        await MainActor.run {
            analysisError = "Backend unavailable and no fallback configured."
            isAnalyzing = false
        }
        return nil
    }

    /// Called by UI when user grants consent to use fallback backend
    func proceedWithFallback() async {
        guard let recording = pendingRecording, let fallbackURL = fallbackURL else {
            await MainActor.run {
                analysisError = "Internal error: No pending fallback request"
                isAnalyzing = false
            }
            return
        }

        #if DEBUG
        print("User granted fallback consent, trying: \(fallbackURL)")
        #endif

        await MainActor.run {
            analysisError = "Connecting to fallback backend..."
        }

        if let result = await analyzeECGWithURL(recording, url: fallbackURL) {
            await MainActor.run {
                analysisError = "✓ Analysis completed using fallback backend"
                analysisResults[recording.id] = result
            }
        } else {
            await MainActor.run {
                analysisError = "Fallback backend also unavailable. Please try again later."
                isAnalyzing = false
            }
        }

        // Clear pending request
        await MainActor.run {
            pendingRecording = nil
            self.fallbackURL = nil
        }
    }

    /// Called by UI when user denies consent to use fallback backend
    func cancelFallback() async {
        await MainActor.run {
            analysisError = "Analysis cancelled. Primary backend unavailable."
            isAnalyzing = false
            pendingRecording = nil
            fallbackURL = nil
        }
    }

    private func analyzeECGWithURL(_ recording: ECGRecording, url backendURL: String) async -> ECGAnalysisResponse? {

        #if DEBUG
        // Validate that we have sufficient voltage measurements (DEBUG ONLY)
        print("=== ECG DATA VALIDATION ===")
        print("Backend URL: \(backendURL)")
        print("Recording ID: \(recording.id)")
        print("Voltage measurements count: \(recording.voltageMeasurements.count)")
        print("Sampling frequency: \(recording.samplingFrequency)")
        print("Duration: \(recording.duration) seconds")
        #endif

        if recording.voltageMeasurements.isEmpty {
            await MainActor.run {
                analysisError = "No voltage measurements found in this ECG recording. The ECG data may be corrupted or incomplete."
                isAnalyzing = false
            }
            #if DEBUG
            print("ERROR: Empty voltage measurements array")
            #endif
            return nil
        }

        if recording.voltageMeasurements.count < 100 {
            await MainActor.run {
                analysisError = "Insufficient ECG data: only \(recording.voltageMeasurements.count) samples (minimum 100 required). The recording may be too short or incomplete."
                isAnalyzing = false
            }
            #if DEBUG
            print("ERROR: Only \(recording.voltageMeasurements.count) samples, need at least 100")
            #endif
            return nil
        }

        let apiRequest = recording.toAPIRequest(apiURL: backendURL)

        #if DEBUG
        // Log the request data (DEBUG ONLY)
        print("Request data - samples: \(apiRequest.samples.count), sampling rate: \(apiRequest.sampling_rate_hz) Hz")
        print("=== END VALIDATION ===")
        #endif


        guard let url = URL(string: "\(backendURL)/v1/ecg/analyze") else {
            #if DEBUG
            print("ERROR: Invalid URL - \(backendURL)/v1/ecg/analyze")
            #endif
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

            // Encode the request with specific error handling
            do {
                request.httpBody = try encoder.encode(apiRequest)
                #if DEBUG
                print("Successfully encoded request body (\(request.httpBody?.count ?? 0) bytes)")
                #endif
            } catch {
                await MainActor.run {
                    analysisError = "Failed to encode ECG data: \(error.localizedDescription)"
                    isAnalyzing = false
                }
                #if DEBUG
                print("ERROR: Failed to encode API request: \(error)")
                #endif
                return nil
            }

            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                #if DEBUG
                print("ERROR: Invalid response from server")
                #endif
                return nil
            }

            #if DEBUG
            // DEBUG: Print raw response (CONTAINS PHI - DEBUG ONLY!)
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
            #endif

            guard httpResponse.statusCode == 200 else {
                #if DEBUG
                let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
                print("ERROR: Server returned status \(httpResponse.statusCode): \(errorMessage)")
                #endif
                return nil
            }

            // Decode the response with specific error handling
            let decoder = JSONDecoder()
            let analysisResponse: ECGAnalysisResponse
            do {
                analysisResponse = try decoder.decode(ECGAnalysisResponse.self, from: data)
                #if DEBUG
                print("Successfully decoded analysis response")
                #endif
            } catch {
                #if DEBUG
                print("ERROR: Failed to decode server response: \(error)")
                if let decodingError = error as? DecodingError {
                    switch decodingError {
                    case .keyNotFound(let key, let context):
                        print("Missing key: \(key.stringValue) at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                    case .typeMismatch(let type, let context):
                        print("Type mismatch for type: \(type) at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                        print("Expected \(type) but got something else")
                    case .valueNotFound(let type, let context):
                        print("Value not found for type: \(type) at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                    case .dataCorrupted(let context):
                        print("Data corrupted at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                        print("Debug description: \(context.debugDescription)")
                    @unknown default:
                        print("Unknown decoding error: \(decodingError)")
                    }
                }
                #endif
                return nil
            }

            await MainActor.run {
                analysisResults[recording.id] = analysisResponse
                isAnalyzing = false
            }

            return analysisResponse

        } catch {
            #if DEBUG
            print("ERROR: Network or unknown error: \(error)")
            #endif
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
