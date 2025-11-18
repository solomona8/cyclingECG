//
//  ECGListView.swift
//  ECGHealthKit
//
//  List view displaying all ECG recordings
//

import SwiftUI

struct ECGListView: View {
    @EnvironmentObject var healthKitManager: HealthKitManager
    @EnvironmentObject var analysisService: ECGAnalysisService

    var body: some View {
        Group {
            if healthKitManager.ecgRecordings.isEmpty {
                EmptyStateView()
            } else {
                List(healthKitManager.ecgRecordings) { recording in
                    NavigationLink(destination: ECGDetailView(recording: recording)
                        .environmentObject(analysisService)) {
                        ECGRecordingRow(recording: recording)
                    }
                }
            }
        }
    }
}

// MARK: - ECG Recording Row

struct ECGRecordingRow: View {
    let recording: ECGRecording

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                // Classification badge
                ClassificationBadge(classification: recording.classification)

                Spacer()

                // Date
                Text(recording.startDate, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(recording.startDate, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Heart rate if available
            if let avgHR = recording.averageHeartRateBPM {
                HStack {
                    Image(systemName: "heart.fill")
                        .foregroundColor(.red)
                        .font(.caption)

                    Text("\(Int(avgHR)) bpm")
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    Spacer()

                    // Duration
                    Text("\(String(format: "%.0f", recording.duration))s")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Symptoms indicator
            if recording.symptomsStatus == .present {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                        .font(.caption)

                    Text("Symptoms Recorded")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Classification Badge

struct ClassificationBadge: View {
    let classification: HKElectrocardiogram.Classification

    var badgeColor: Color {
        switch classification {
        case .sinusRhythm:
            return .green
        case .atrialFibrillation:
            return .red
        case .inconclusiveLowHeartRate, .inconclusiveHighHeartRate,
             .inconclusivePoorReading, .inconclusiveOther:
            return .orange
        default:
            return .gray
        }
    }

    var badgeIcon: String {
        switch classification {
        case .sinusRhythm:
            return "checkmark.circle.fill"
        case .atrialFibrillation:
            return "exclamationmark.circle.fill"
        default:
            return "questionmark.circle.fill"
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: badgeIcon)
                .font(.caption)

            Text(classificationText)
                .font(.caption)
                .fontWeight(.medium)
        }
        .foregroundColor(.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(badgeColor)
        .cornerRadius(8)
    }

    var classificationText: String {
        switch classification {
        case .sinusRhythm:
            return "Sinus"
        case .atrialFibrillation:
            return "AFib"
        case .inconclusiveLowHeartRate:
            return "Low HR"
        case .inconclusiveHighHeartRate:
            return "High HR"
        case .inconclusivePoorReading:
            return "Poor Reading"
        default:
            return "Inconclusive"
        }
    }
}

// MARK: - Empty State View

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "heart.text.square")
                .font(.system(size: 60))
                .foregroundColor(.gray)

            Text("No ECG Recordings Found")
                .font(.title3)
                .fontWeight(.semibold)

            Text("Take an ECG recording on your Apple Watch to see it here.")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal)
        }
        .padding()
    }
}

import HealthKit
