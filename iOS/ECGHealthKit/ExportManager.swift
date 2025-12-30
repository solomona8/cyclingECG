//
//  ExportManager.swift
//  ECGHealthKit
//
//  Handles exporting ECG data in various formats
//

import Foundation
import UIKit

class ExportManager {

    // MARK: - Export Formats

    enum ExportFormat {
        case json
        case csv
        case pdf
        case txt
    }

    // MARK: - Export ECG Recording

    static func exportECGRecording(
        _ recording: ECGRecording,
        analysis: ECGAnalysisResponse?,
        format: ExportFormat
    ) -> URL? {
        switch format {
        case .json:
            return exportAsJSON(recording, analysis: analysis)
        case .csv:
            return exportAsCSV(recording, analysis: analysis)
        case .pdf:
            return exportAsPDF(recording, analysis: analysis)
        case .txt:
            return exportAsText(recording, analysis: analysis)
        }
    }

    // MARK: - JSON Export

    private static func exportAsJSON(
        _ recording: ECGRecording,
        analysis: ECGAnalysisResponse?
    ) -> URL? {
        // Validate we have measurements
        guard !recording.voltageMeasurements.isEmpty else {
            #if DEBUG
            print("ERROR: Cannot export JSON - no voltage measurements")
            #endif
            return nil
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let exportData = ECGExportData(
            recording: recording,
            analysis: analysis,
            exportDate: Date()
        )

        do {
            let jsonData = try encoder.encode(exportData)
            #if DEBUG
            print("JSON Export: Successfully created \(jsonData.count) bytes of JSON data")
            #endif

            return saveToTemporaryFile(
                data: jsonData,
                filename: "ECG_\(recording.id).json"
            )
        } catch {
            #if DEBUG
            print("ERROR: Failed to encode JSON: \(error.localizedDescription)")
            #endif
            return nil
        }
    }

    // MARK: - CSV Export

    private static func exportAsCSV(
        _ recording: ECGRecording,
        analysis: ECGAnalysisResponse?
    ) -> URL? {
        // Validate we have measurements
        guard !recording.voltageMeasurements.isEmpty else {
            #if DEBUG
            print("ERROR: Cannot export CSV - no voltage measurements")
            #endif
            return nil
        }

        // Add metadata as comments at the top
        var metadata = "# ECG Recording Metadata\n"
        metadata += "# Recording ID: \(recording.id)\n"
        metadata += "# Start Date: \(recording.startDate)\n"
        metadata += "# Duration: \(String(format: "%.2f", recording.duration)) seconds\n"
        metadata += "# Sampling Frequency: \(String(format: "%.2f", recording.samplingFrequency)) Hz\n"
        metadata += "# Classification: \(recording.classificationDescription)\n"
        metadata += "# Number of Samples: \(recording.voltageMeasurements.count)\n"

        if let avgHR = recording.averageHeartRateBPM {
            metadata += "# Average Heart Rate: \(String(format: "%.1f", avgHR)) bpm\n"
        }

        if let analysis = analysis {
            metadata += "# Analysis Rhythm: \(analysis.features.rhythm_classification ?? "N/A")\n"

            if let hrMean = analysis.features.heart_rate_bpm?.mean {
                metadata += "# Analyzed HR (mean): \(String(format: "%.1f", hrMean)) bpm\n"
            }

            if let sdnn = analysis.features.hrv?.sdnn_ms {
                metadata += "# HRV SDNN: \(String(format: "%.2f", sdnn)) ms\n"
            }

            if let rmssd = analysis.features.hrv?.rmssd_ms {
                metadata += "# HRV RMSSD: \(String(format: "%.2f", rmssd)) ms\n"
            }
        }

        metadata += "#\n"
        metadata += "# Data Format: Time (seconds), Voltage (microvolts)\n"
        metadata += "#\n"

        // CSV header
        var csvContent = "Timestamp (seconds),Voltage (microvolts)\n"

        // Calculate time interval between samples
        let timeInterval = recording.duration / Double(recording.voltageMeasurements.count - 1)

        // Export voltage measurements
        for (index, voltage) in recording.voltageMeasurements.enumerated() {
            let time = Double(index) * timeInterval
            let voltageInMicrovolts = voltage * 1_000_000
            csvContent += "\(String(format: "%.6f", time)),\(String(format: "%.2f", voltageInMicrovolts))\n"
        }

        let fullCSV = metadata + csvContent

        guard let csvData = fullCSV.data(using: .utf8) else {
            #if DEBUG
            print("ERROR: Failed to encode CSV data as UTF-8")
            #endif
            return nil
        }

        #if DEBUG
        print("CSV Export: Successfully created \(csvData.count) bytes of CSV data with \(recording.voltageMeasurements.count) samples")
        #endif

        return saveToTemporaryFile(
            data: csvData,
            filename: "ECG_\(recording.id).csv"
        )
    }

    // MARK: - Text Export

    private static func exportAsText(
        _ recording: ECGRecording,
        analysis: ECGAnalysisResponse?
    ) -> URL? {
        var textContent = "ECG RECORDING REPORT\n"
        textContent += "====================\n\n"

        textContent += "RECORDING INFORMATION\n"
        textContent += "---------------------\n"
        textContent += "Recording ID: \(recording.id)\n"
        textContent += "Start Date: \(formatDate(recording.startDate))\n"
        textContent += "Duration: \(String(format: "%.2f", recording.duration)) seconds\n"
        textContent += "Sampling Frequency: \(String(format: "%.2f", recording.samplingFrequency)) Hz\n"
        textContent += "Number of Samples: \(recording.numberOfVoltageMeasurements)\n"
        textContent += "Classification: \(recording.classificationDescription)\n"
        textContent += "Symptoms: \(recording.symptomsDescription)\n"

        if let avgHR = recording.averageHeartRateBPM {
            textContent += "Average Heart Rate: \(String(format: "%.1f", avgHR)) bpm\n"
        }

        textContent += "\n"

        if let analysis = analysis {
            textContent += "ANALYSIS RESULTS\n"
            textContent += "----------------\n"

            if let rhythm = analysis.features.rhythm_classification {
                textContent += "Rhythm: \(rhythm)\n"
            }

            if let confidence = analysis.features.rhythm_confidence {
                textContent += "Rhythm Confidence: \(String(format: "%.1f%%", confidence * 100))\n"
            }

            if let hrInfo = analysis.features.heart_rate_bpm {
                textContent += "\nHeart Rate:\n"
                if let mean = hrInfo.mean {
                    textContent += "  Mean: \(String(format: "%.1f", mean)) bpm\n"
                }
                if let min = hrInfo.min {
                    textContent += "  Min: \(String(format: "%.1f", min)) bpm\n"
                }
                if let max = hrInfo.max {
                    textContent += "  Max: \(String(format: "%.1f", max)) bpm\n"
                }
            }

            if let hrvInfo = analysis.features.hrv {
                textContent += "\nHeart Rate Variability:\n"
                if let sdnn = hrvInfo.sdnn_ms {
                    textContent += "  SDNN: \(String(format: "%.2f", sdnn)) ms\n"
                }
                if let rmssd = hrvInfo.rmssd_ms {
                    textContent += "  RMSSD: \(String(format: "%.2f", rmssd)) ms\n"
                }
            }

            if let intervals = analysis.features.intervals {
                textContent += "\nIntervals:\n"
                if let qrs = intervals.qrs_duration_ms {
                    textContent += "  QRS Duration: \(String(format: "%.1f", qrs)) ms\n"
                }
                if let qt = intervals.qt_interval_ms {
                    textContent += "  QT Interval: \(String(format: "%.1f", qt)) ms\n"
                }
                if let qtc = intervals.qtc_ms {
                    textContent += "  QTc: \(String(format: "%.1f", qtc)) ms\n"
                }
            }

            if let quality = analysis.features.signal_quality {
                textContent += "\nSignal Quality:\n"
                if let overall = quality.overall_quality {
                    textContent += "  Overall: \(overall)\n"
                }
                if let artifact = quality.artifact_burden_percent {
                    textContent += "  Artifact Burden: \(String(format: "%.1f%%", artifact))\n"
                }
            }

            if let narrative = analysis.narrative {
                textContent += "\nNARRATIVE\n"
                textContent += "---------\n"

                if let patientSummary = narrative.patient_summary {
                    textContent += "\nPatient Summary:\n\(patientSummary)\n"
                }

                if let clinicianNotes = narrative.clinician_notes {
                    textContent += "\nClinician Notes:\n\(clinicianNotes)\n"
                }

                if let safetyFlags = narrative.safety_flags, !safetyFlags.isEmpty {
                    textContent += "\nSafety Flags:\n"
                    for flag in safetyFlags {
                        textContent += "  - \(flag)\n"
                    }
                }
            }
        }

        textContent += "\n"
        textContent += "Generated: \(formatDate(Date()))\n"

        guard let textData = textContent.data(using: .utf8) else {
            return nil
        }

        return saveToTemporaryFile(
            data: textData,
            filename: "ECG_Report_\(recording.id).txt"
        )
    }

    // MARK: - PDF Export

    private static func exportAsPDF(
        _ recording: ECGRecording,
        analysis: ECGAnalysisResponse?
    ) -> URL? {
        let pdfMetaData = [
            kCGPDFContextCreator: "ECG Insights App",
            kCGPDFContextAuthor: "Apple Watch",
            kCGPDFContextTitle: "ECG Recording Report"
        ]

        let format = UIGraphicsPDFRendererFormat()
        format.documentInfo = pdfMetaData as [String: Any]

        let pageWidth = 8.5 * 72.0
        let pageHeight = 11 * 72.0
        let pageRect = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)

        let renderer = UIGraphicsPDFRenderer(bounds: pageRect, format: format)

        let data = renderer.pdfData { context in
            context.beginPage()

            let titleFont = UIFont.boldSystemFont(ofSize: 24)
            let headingFont = UIFont.boldSystemFont(ofSize: 16)
            let bodyFont = UIFont.systemFont(ofSize: 12)

            var yPosition: CGFloat = 40

            // Title
            let title = "ECG Recording Report"
            title.draw(at: CGPoint(x: 40, y: yPosition), withAttributes: [
                .font: titleFont,
                .foregroundColor: UIColor.black
            ])
            yPosition += 40

            // Recording Information
            drawSection(
                context: context,
                title: "Recording Information",
                yPosition: &yPosition,
                headingFont: headingFont,
                bodyFont: bodyFont
            ) {
                var info = ""
                info += "Recording ID: \(recording.id)\n"
                info += "Date: \(formatDate(recording.startDate))\n"
                info += "Duration: \(String(format: "%.2f", recording.duration)) seconds\n"
                info += "Sampling Frequency: \(String(format: "%.0f", recording.samplingFrequency)) Hz\n"
                info += "Classification: \(recording.classificationDescription)\n"
                info += "Symptoms: \(recording.symptomsDescription)\n"

                if let avgHR = recording.averageHeartRateBPM {
                    info += "Average Heart Rate: \(String(format: "%.0f", avgHR)) bpm"
                }

                return info
            }

            // Analysis Results
            if let analysis = analysis {
                yPosition += 20

                drawSection(
                    context: context,
                    title: "Analysis Results",
                    yPosition: &yPosition,
                    headingFont: headingFont,
                    bodyFont: bodyFont
                ) {
                    var analysisText = ""

                    if let rhythm = analysis.features.rhythm_classification {
                        analysisText += "Rhythm: \(rhythm)\n"
                    }

                    if let hrMean = analysis.features.heart_rate_bpm?.mean {
                        analysisText += "Heart Rate (mean): \(String(format: "%.0f", hrMean)) bpm\n"
                    }

                    if let sdnn = analysis.features.hrv?.sdnn_ms {
                        analysisText += "HRV SDNN: \(String(format: "%.1f", sdnn)) ms\n"
                    }

                    if let quality = analysis.features.signal_quality?.overall_quality {
                        analysisText += "Signal Quality: \(quality)"
                    }

                    return analysisText
                }

                // Narrative
                if let narrative = analysis.narrative {
                    if let patientSummary = narrative.patient_summary {
                        yPosition += 20
                        drawSection(
                            context: context,
                            title: "Summary",
                            yPosition: &yPosition,
                            headingFont: headingFont,
                            bodyFont: bodyFont
                        ) {
                            return patientSummary
                        }
                    }
                }
            }

            // Footer
            let footer = "Generated: \(formatDate(Date()))"
            let footerY = pageHeight - 40
            footer.draw(at: CGPoint(x: 40, y: footerY), withAttributes: [
                .font: UIFont.systemFont(ofSize: 10),
                .foregroundColor: UIColor.gray
            ])
        }

        return saveToTemporaryFile(
            data: data,
            filename: "ECG_Report_\(recording.id).pdf"
        )
    }

    // MARK: - Helper Methods

    private static func drawSection(
        context: UIGraphicsPDFRendererContext,
        title: String,
        yPosition: inout CGFloat,
        headingFont: UIFont,
        bodyFont: UIFont,
        content: () -> String
    ) {
        // Draw section title
        title.draw(at: CGPoint(x: 40, y: yPosition), withAttributes: [
            .font: headingFont,
            .foregroundColor: UIColor.black
        ])
        yPosition += 25

        // Draw section content
        let contentText = content()
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineSpacing = 4

        let contentRect = CGRect(x: 40, y: yPosition, width: 500, height: 1000)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: bodyFont,
            .foregroundColor: UIColor.darkGray,
            .paragraphStyle: paragraphStyle
        ]

        let attributedText = NSAttributedString(string: contentText, attributes: attributes)
        let textSize = attributedText.boundingRect(
            with: CGSize(width: 500, height: CGFloat.greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            context: nil
        )

        contentText.draw(in: contentRect, withAttributes: attributes)
        yPosition += textSize.height + 10
    }

    private static func saveToTemporaryFile(data: Data, filename: String) -> URL? {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(filename)

        do {
            try data.write(to: fileURL)
            return fileURL
        } catch {
            #if DEBUG
            print("Error saving file: \(error)")
            #endif
            return nil
        }
    }

    private static func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .medium
        return formatter.string(from: date)
    }
}

// MARK: - Export Data Model

struct ECGExportData: Codable {
    let recording: ECGRecording
    let analysis: ECGAnalysisResponse?
    let exportDate: Date
}
