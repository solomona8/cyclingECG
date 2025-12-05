//
//  ECGDetailView.swift
//  ECGHealthKit
//
//  Detail view for individual ECG recordings with analysis and export
//

import SwiftUI
import HealthKit

struct ECGDetailView: View {
    let recording: ECGRecording

    @EnvironmentObject var analysisService: ECGAnalysisService
    @AppStorage("api_url") private var apiURL = "https://cyclingecg.onrender.com"
    @AppStorage("api_key") private var apiKey = ""

    @State private var showingExportSheet = false
    @State private var showingShareSheet = false
    @State private var fileToShare: URL?

    var analysisResult: ECGAnalysisResponse? {
        analysisService.getAnalysisResult(for: recording.id)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header Card
                HeaderCard(recording: recording)

                // Analysis Section
                if analysisResult != nil || analysisService.isAnalyzing {
                    AnalysisCard(
                        recording: recording,
                        analysis: analysisResult,
                        isAnalyzing: analysisService.isAnalyzing
                    )
                } else {
                    AnalyzeButton(recording: recording)
                }

                // Error message
                if let error = analysisService.analysisError {
                    ErrorCard(error: error)
                }

                // Recording Details
                RecordingDetailsCard(recording: recording)

                // Export Button
                ExportButton(action: {
                    showingExportSheet = true
                })
            }
            .padding()
        }
        .navigationTitle("ECG Details")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showingExportSheet) {
            ExportOptionsView(
                recording: recording,
                analysis: analysisResult,
                onExport: { url in
                    fileToShare = url
                    showingShareSheet = true
                }
            )
        }
        .sheet(isPresented: $showingShareSheet) {
            if let url = fileToShare {
                ShareSheet(items: [url])
            }
        }
    }
}

// MARK: - Header Card

struct HeaderCard: View {
    let recording: ECGRecording

    var body: some View {
        VStack(spacing: 12) {
            // Classification
            ClassificationBadge(classification: recording.classification)

            // Date and Time
            Text(recording.startDate, style: .date)
                .font(.headline)

            Text(recording.startDate, style: .time)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Divider()

            // Heart Rate
            if let avgHR = recording.averageHeartRateBPM {
                HStack {
                    Image(systemName: "heart.fill")
                        .foregroundColor(.red)

                    Text("\(Int(avgHR))")
                        .font(.system(size: 48, weight: .bold))

                    Text("bpm")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
            }

            // Symptoms
            if recording.symptomsStatus == .present {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)

                    Text("Symptoms Recorded")
                        .font(.subheadline)
                        .foregroundColor(.orange)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }
}

// MARK: - Analyze Button

struct AnalyzeButton: View {
    let recording: ECGRecording

    @EnvironmentObject var analysisService: ECGAnalysisService
    @AppStorage("api_url") private var apiURL = "https://cyclingecg.onrender.com"
    @AppStorage("api_key") private var apiKey = ""

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 40))
                .foregroundColor(.blue)

            Text("Analyze ECG")
                .font(.headline)

            Text("Send this recording to the backend for detailed analysis")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Button(action: {
                Task {
                    // Set analyzing state on the environment object so UI updates
                    await MainActor.run {
                        analysisService.isAnalyzing = true
                        analysisService.analysisError = nil
                    }

                    do {
                        // Update service with current settings
                        let service = ECGAnalysisService(
                            baseURL: apiURL,
                            apiKey: apiKey.isEmpty ? nil : apiKey
                        )

                        // Copy the published state
                        service.analysisResults = analysisService.analysisResults

                        let _ = await service.analyzeECG(recording)

                        // Update the main service with all results
                        await MainActor.run {
                            analysisService.analysisResults = service.analysisResults
                            analysisService.analysisError = service.analysisError
                            analysisService.isAnalyzing = false
                        }
                    } catch {
                        // Handle any unexpected errors
                        await MainActor.run {
                            analysisService.analysisError = "Analysis failed: \(error.localizedDescription)"
                            analysisService.isAnalyzing = false
                        }
                    }
                }
            }) {
                Text("Analyze Now")
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }
}

// MARK: - Analysis Card

struct AnalysisCard: View {
    let recording: ECGRecording
    let analysis: ECGAnalysisResponse?
    let isAnalyzing: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "waveform.path.ecg")
                    .foregroundColor(.blue)

                Text("Analysis Results")
                    .font(.headline)

                Spacer()

                if isAnalyzing {
                    ProgressView()
                }
            }

            if isAnalyzing {
                HStack {
                    Spacer()
                    VStack(spacing: 8) {
                        ProgressView()
                        Text("Analyzing...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                }
                .padding()
            } else if let analysis = analysis {
                Divider()

                // Rhythm
                if let rhythm = analysis.features.rhythm_classification {
                    AnalysisRow(
                        icon: "heart.circle.fill",
                        label: "Rhythm",
                        value: rhythm.capitalized,
                        color: .blue
                    )
                }

                // Heart Rate
                if let hrMean = analysis.features.heart_rate_bpm?.mean {
                    AnalysisRow(
                        icon: "heart.fill",
                        label: "Mean Heart Rate",
                        value: "\(Int(hrMean)) bpm",
                        color: .red
                    )
                }

                // HRV SDNN
                if let sdnn = analysis.features.hrv?.sdnn_ms {
                    AnalysisRow(
                        icon: "waveform",
                        label: "HRV SDNN",
                        value: String(format: "%.1f ms", sdnn),
                        color: .green
                    )
                }

                // Signal Quality
                if let quality = analysis.features.signal_quality?.overall_quality {
                    AnalysisRow(
                        icon: "checkmark.seal.fill",
                        label: "Signal Quality",
                        value: quality.capitalized,
                        color: qualityColor(quality)
                    )
                }

                // Narrative
                if let narrative = analysis.narrative {
                    Divider()

                    if let summary = narrative.patient_summary {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Summary")
                                .font(.subheadline)
                                .fontWeight(.semibold)

                            Text(summary)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    if let flags = narrative.safety_flags, !flags.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Safety Flags")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.orange)

                            ForEach(flags, id: \.self) { flag in
                                HStack {
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundColor(.orange)
                                        .font(.caption)

                                    Text(flag)
                                        .font(.caption)
                                }
                            }
                        }
                        .padding()
                        .background(Color.orange.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    private func qualityColor(_ quality: String) -> Color {
        switch quality.lowercased() {
        case "good":
            return .green
        case "moderate":
            return .orange
        default:
            return .red
        }
    }
}

// MARK: - Analysis Row

struct AnalysisRow: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24)

            Text(label)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
        }
    }
}

// MARK: - Recording Details Card

struct RecordingDetailsCard: View {
    let recording: ECGRecording

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "info.circle.fill")
                    .foregroundColor(.gray)

                Text("Recording Details")
                    .font(.headline)
            }

            Divider()

            DetailRow(label: "Recording ID", value: recording.id)
            DetailRow(label: "Duration", value: String(format: "%.2f seconds", recording.duration))
            DetailRow(label: "Sampling Frequency", value: String(format: "%.0f Hz", recording.samplingFrequency))
            DetailRow(label: "Number of Samples", value: "\(recording.numberOfVoltageMeasurements)")
            DetailRow(label: "Classification", value: recording.classificationDescription)
            DetailRow(label: "Symptoms Status", value: recording.symptomsDescription)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }
}

// MARK: - Detail Row

struct DetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Error Card

struct ErrorCard: View {
    let error: String

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)

            Text(error)
                .font(.caption)
                .foregroundColor(.red)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.red.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Export Button

struct ExportButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: "square.and.arrow.up")
                    .font(.headline)

                Text("Export Recording")
                    .fontWeight(.semibold)
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.green)
            .cornerRadius(10)
        }
    }
}

// MARK: - Export Options View

struct ExportOptionsView: View {
    let recording: ECGRecording
    let analysis: ECGAnalysisResponse?
    let onExport: (URL) -> Void

    @Environment(\.dismiss) var dismiss
    @State private var selectedFormat: ExportManager.ExportFormat = .json
    @State private var isExporting = false
    @State private var exportError: String?

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Choose Export Format")
                    .font(.headline)
                    .padding(.top)

                VStack(spacing: 12) {
                    FormatButton(
                        title: "JSON",
                        description: "Complete data with analysis results",
                        icon: "doc.text.fill",
                        isSelected: selectedFormat == .json,
                        action: { selectedFormat = .json }
                    )

                    FormatButton(
                        title: "CSV",
                        description: "Voltage measurements for analysis tools",
                        icon: "tablecells.fill",
                        isSelected: selectedFormat == .csv,
                        action: { selectedFormat = .csv }
                    )

                    FormatButton(
                        title: "PDF",
                        description: "Formatted report for sharing",
                        icon: "doc.richtext.fill",
                        isSelected: selectedFormat == .pdf,
                        action: { selectedFormat = .pdf }
                    )

                    FormatButton(
                        title: "Text",
                        description: "Plain text summary",
                        icon: "doc.plaintext.fill",
                        isSelected: selectedFormat == .txt,
                        action: { selectedFormat = .txt }
                    )
                }
                .padding()

                if let error = exportError {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding()
                }

                Spacer()

                Button(action: exportFile) {
                    if isExporting {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Text("Export")
                            .fontWeight(.semibold)
                    }
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .cornerRadius(10)
                .padding()
                .disabled(isExporting)
            }
            .navigationTitle("Export ECG")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }

    private func exportFile() {
        isExporting = true
        exportError = nil

        DispatchQueue.global(qos: .userInitiated).async {
            if let url = ExportManager.exportECGRecording(
                recording,
                analysis: analysis,
                format: selectedFormat
            ) {
                DispatchQueue.main.async {
                    isExporting = false
                    dismiss()
                    onExport(url)
                }
            } else {
                DispatchQueue.main.async {
                    isExporting = false
                    exportError = "Failed to export file"
                }
            }
        }
    }
}

// MARK: - Format Button

struct FormatButton: View {
    let title: String
    let description: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(isSelected ? .blue : .gray)
                    .frame(width: 40)

                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(.primary)

                    Text(description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.blue)
                }
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(isSelected ? Color.blue : Color.gray.opacity(0.3), lineWidth: 2)
            )
        }
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: items,
            applicationActivities: nil
        )
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
