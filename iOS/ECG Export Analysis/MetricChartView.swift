//
//  MetricChartView.swift
//  ECG Export Analysis
//
//  Displays historical chart for a specific ECG metric
//

import SwiftUI
import Charts

struct MetricChartView: View {
    let metric: MetricType
    @ObservedObject var historyManager: AnalysisHistoryManager

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Image(systemName: metric.icon)
                    .foregroundColor(.blue)
                Text(metric.rawValue)
                    .font(.title2)
                    .fontWeight(.bold)
            }
            .padding(.horizontal)

            // Chart
            let data = historyManager.getMetricHistory(metric: metric, days: 30)

            if data.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 48))
                        .foregroundColor(.gray)
                    Text("No data available for the last 30 days")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Text("Analyze more ECG recordings to see trends")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 60)
            } else {
                Chart {
                    ForEach(data.indices, id: \.self) { index in
                        LineMark(
                            x: .value("Date", data[index].date),
                            y: .value(metric.rawValue, data[index].value)
                        )
                        .foregroundStyle(.blue)
                        .interpolationMethod(.catmullRom)

                        PointMark(
                            x: .value("Date", data[index].date),
                            y: .value(metric.rawValue, data[index].value)
                        )
                        .foregroundStyle(.blue)
                    }

                    // Add reference line for average
                    if let average = calculateAverage(data: data) {
                        RuleMark(y: .value("Average", average))
                            .foregroundStyle(.green.opacity(0.5))
                            .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
                            .annotation(position: .top, alignment: .trailing) {
                                Text("Avg: \(formatValue(average))")
                                    .font(.caption)
                                    .foregroundColor(.green)
                                    .padding(4)
                                    .background(Color.green.opacity(0.1))
                                    .cornerRadius(4)
                            }
                    }
                }
                .chartXAxis {
                    AxisMarks(values: .automatic) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel(format: .dateTime.month().day())
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel {
                            if let doubleValue = value.as(Double.self) {
                                Text(formatValue(doubleValue))
                            }
                        }
                    }
                }
                .frame(height: 300)
                .padding(.horizontal)

                // Statistics
                if !data.isEmpty {
                    VStack(spacing: 12) {
                        Divider()

                        HStack(spacing: 24) {
                            StatisticView(
                                label: "Min",
                                value: formatValue(data.map { $0.value }.min() ?? 0),
                                unit: metric.unit,
                                color: .blue
                            )

                            StatisticView(
                                label: "Avg",
                                value: formatValue(calculateAverage(data: data) ?? 0),
                                unit: metric.unit,
                                color: .green
                            )

                            StatisticView(
                                label: "Max",
                                value: formatValue(data.map { $0.value }.max() ?? 0),
                                unit: metric.unit,
                                color: .red
                            )
                        }
                        .padding(.horizontal)

                        Text("\(data.count) measurements in last 30 days")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical)
                }
            }

            Spacer()
        }
        .navigationTitle(metric.rawValue)
        .navigationBarTitleDisplayMode(.inline)
    }

    private func calculateAverage(data: [(date: Date, value: Double)]) -> Double? {
        guard !data.isEmpty else { return nil }
        return data.map { $0.value }.reduce(0, +) / Double(data.count)
    }

    private func formatValue(_ value: Double) -> String {
        if metric == .heartRateMean {
            return String(format: "%.0f", value)
        } else {
            return String(format: "%.1f", value)
        }
    }
}

struct StatisticView: View {
    let label: String
    let value: String
    let unit: String
    let color: Color

    var body: some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)

            HStack(spacing: 4) {
                Text(value)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(color)

                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    NavigationView {
        MetricChartView(
            metric: .heartRateMean,
            historyManager: AnalysisHistoryManager()
        )
    }
}
