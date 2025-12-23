//
//  PrivacyPolicyView.swift
//  ECGHealthKit
//
//  Privacy Policy and Terms of Service view for App Store compliance
//

import SwiftUI

struct PrivacyPolicyView: View {
    @Environment(\.dismiss) var dismiss
    @Binding var hasAcceptedPolicy: Bool
    let isFirstLaunch: Bool

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Privacy Policy")
                            .font(.largeTitle)
                            .fontWeight(.bold)

                        Text("Last Updated: December 2024")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.bottom, 8)

                    // Important Medical Disclaimer
                    DisclaimerCard(
                        icon: "exclamationmark.triangle.fill",
                        iconColor: .orange,
                        title: "Medical Disclaimer",
                        content: "This app is for informational purposes only and is NOT a medical device. Do not use this app for medical diagnosis or treatment decisions. Always consult with a qualified healthcare provider for medical advice."
                    )

                    Divider()

                    // Data Collection
                    PolicySection(
                        title: "1. What Data We Collect",
                        icon: "doc.text.fill"
                    ) {
                        Text("We collect and process the following data:")
                        BulletPoint("ECG (electrocardiogram) recordings from Apple Watch")
                        BulletPoint("Recording timestamps and metadata")
                        BulletPoint("Calculated heart rate and rhythm metrics")
                        BulletPoint("Heart rate variability (HRV) measurements")
                        BulletPoint("ECG waveform characteristics (P wave, QRS, QT intervals)")

                        Text("\n**We do NOT collect:**")
                            .fontWeight(.semibold)
                        BulletPoint("Personal identifying information (name, email, phone)")
                        BulletPoint("Location data")
                        BulletPoint("Contacts or photos")
                        BulletPoint("Any data from other apps")
                    }

                    Divider()

                    // How We Use Data
                    PolicySection(
                        title: "2. How We Use Your Data",
                        icon: "cpu.fill"
                    ) {
                        Text("Your health data is used exclusively for:")
                        BulletPoint("Analyzing ECG waveforms and detecting patterns")
                        BulletPoint("Calculating heart rate metrics and trends")
                        BulletPoint("Generating analysis reports for your review")
                        BulletPoint("Creating historical trend charts (stored locally)")

                        Text("\n**We do NOT:**")
                            .fontWeight(.semibold)
                        BulletPoint("Sell your data to third parties")
                        BulletPoint("Use your data for advertising")
                        BulletPoint("Share your data with insurance companies")
                        BulletPoint("Store your data permanently on our servers")
                    }

                    Divider()

                    // Data Storage
                    PolicySection(
                        title: "3. Where Your Data Is Stored",
                        icon: "lock.shield.fill"
                    ) {
                        Text("**Local Storage (On Your Device):**")
                            .fontWeight(.semibold)
                        BulletPoint("Analysis history for the last 90 days")
                        BulletPoint("Encrypted using AES-256-GCM encryption")
                        BulletPoint("Stored in app's secure sandbox")
                        BulletPoint("Protected by iOS security features")

                        Text("\n**Backend Server (Optional):**")
                            .fontWeight(.semibold)
                        BulletPoint("When you request analysis, ECG data is sent to our backend server")
                        BulletPoint("Data is transmitted over encrypted HTTPS connection")
                        BulletPoint("Server processes the data and immediately returns results")
                        BulletPoint("Data retention: 30 days maximum, then automatically deleted")
                        BulletPoint("You can use a local backend to keep all data on your network")
                    }

                    Divider()

                    // Third-Party Services
                    PolicySection(
                        title: "4. Third-Party Services",
                        icon: "network"
                    ) {
                        Text("**OpenAI Integration (Optional):**")
                            .fontWeight(.semibold)
                        Text("If enabled in backend configuration, anonymized ECG metrics (NOT raw waveforms) may be sent to OpenAI to generate narrative summaries. This feature is:")
                        BulletPoint("Opt-in only (disabled by default)")
                        BulletPoint("Configured by backend administrator")
                        BulletPoint("Does NOT include personally identifiable information")
                        BulletPoint("Governed by OpenAI's privacy policy")

                        Text("\n**Cloud Hosting:**")
                            .fontWeight(.semibold)
                        Text("Our default backend is hosted on Render.com. Data is:")
                        BulletPoint("Encrypted in transit (HTTPS/TLS)")
                        BulletPoint("Stored temporarily in memory during processing")
                        BulletPoint("Automatically deleted after 30 days")
                        BulletPoint("Not shared with Render.com for any purpose")
                    }

                    Divider()

                    // Your Rights (GDPR/CCPA)
                    PolicySection(
                        title: "5. Your Privacy Rights (GDPR/CCPA)",
                        icon: "person.fill.checkmark"
                    ) {
                        Text("You have the following rights regarding your data:")

                        Text("\n**Right to Access:**")
                            .fontWeight(.semibold)
                        BulletPoint("View all your stored analysis history in the app")
                        BulletPoint("Export your data in JSON, CSV, or PDF format")

                        Text("\n**Right to Delete:**")
                            .fontWeight(.semibold)
                        BulletPoint("Swipe to delete individual analysis records")
                        BulletPoint("Uninstall the app to delete all local data")
                        BulletPoint("Contact us to delete data from backend servers")

                        Text("\n**Right to Portability:**")
                            .fontWeight(.semibold)
                        BulletPoint("Export your data in machine-readable formats")
                        BulletPoint("Transfer data to other health applications")

                        Text("\n**Right to Opt-Out:**")
                            .fontWeight(.semibold)
                        BulletPoint("Use local backend to prevent cloud data transmission")
                        BulletPoint("Disable backend analysis entirely (app still functions)")

                        Text("\n**Right to Object:**")
                            .fontWeight(.semibold)
                        BulletPoint("Revoke HealthKit permissions in iOS Settings")
                        BulletPoint("Request deletion of all backend data")
                    }

                    Divider()

                    // Data Retention
                    PolicySection(
                        title: "6. Data Retention",
                        icon: "calendar"
                    ) {
                        Text("**Local Device:**")
                            .fontWeight(.semibold)
                        BulletPoint("Analysis history retained for 90 days")
                        BulletPoint("Older records automatically deleted")
                        BulletPoint("Deleted immediately when app is uninstalled")

                        Text("\n**Backend Server:**")
                            .fontWeight(.semibold)
                        BulletPoint("Analysis results stored for maximum 30 days")
                        BulletPoint("Automatic deletion after retention period")
                        BulletPoint("You can request immediate deletion by contacting us")
                    }

                    Divider()

                    // Security
                    PolicySection(
                        title: "7. Security Measures",
                        icon: "checkmark.shield.fill"
                    ) {
                        Text("We implement industry-standard security practices:")
                        BulletPoint("AES-256-GCM encryption for local data storage")
                        BulletPoint("TLS 1.3 encryption for network transmission")
                        BulletPoint("API keys stored in iOS Keychain (not plaintext)")
                        BulletPoint("Regular security audits and updates")
                        BulletPoint("No debug logging of health data in production")
                    }

                    Divider()

                    // Children's Privacy
                    PolicySection(
                        title: "8. Children's Privacy",
                        icon: "figure.2.and.child.holdinghands"
                    ) {
                        Text("This app is not intended for children under 13 years of age. We do not knowingly collect data from children. Apple Watch ECG feature requires users to be 22 years or older in most regions.")
                    }

                    Divider()

                    // International Users
                    PolicySection(
                        title: "9. International Data Transfers",
                        icon: "globe"
                    ) {
                        Text("If you use the cloud backend, your data may be processed on servers located in the United States or other countries. We ensure appropriate safeguards are in place:")
                        BulletPoint("Encryption in transit and at rest")
                        BulletPoint("Limited data retention periods")
                        BulletPoint("Compliance with GDPR and CCPA requirements")
                        BulletPoint("Option to use local backend for data sovereignty")
                    }

                    Divider()

                    // Changes to Policy
                    PolicySection(
                        title: "10. Changes to This Policy",
                        icon: "arrow.triangle.2.circlepath"
                    ) {
                        Text("We may update this privacy policy periodically. Changes will be:")
                        BulletPoint("Communicated through app updates")
                        BulletPoint("Posted with a new \"Last Updated\" date")
                        BulletPoint("Requiring re-acceptance for material changes")
                    }

                    Divider()

                    // Contact Information
                    PolicySection(
                        title: "11. Contact Us",
                        icon: "envelope.fill"
                    ) {
                        Text("For privacy-related questions, data deletion requests, or GDPR/CCPA inquiries:")

                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Email:")
                                    .fontWeight(.semibold)
                                Text("privacy@cyclingecg.app")
                                    .foregroundColor(.blue)
                            }
                            .font(.subheadline)

                            Text("Data Protection Officer: Available upon request")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text("Response time: We respond to privacy requests within 30 days as required by GDPR.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 8)
                    }

                    Divider()

                    // Legal Jurisdiction
                    PolicySection(
                        title: "12. Governing Law",
                        icon: "building.columns.fill"
                    ) {
                        Text("This privacy policy is governed by applicable data protection laws including:")
                        BulletPoint("GDPR (General Data Protection Regulation) - EU")
                        BulletPoint("CCPA (California Consumer Privacy Act) - California, USA")
                        BulletPoint("HIPAA (where applicable) - USA")
                        BulletPoint("Local health data protection regulations")
                    }

                    // Acceptance Checkbox (for first launch)
                    if isFirstLaunch {
                        VStack(spacing: 16) {
                            Divider()

                            Button(action: {
                                hasAcceptedPolicy = true
                                dismiss()
                            }) {
                                HStack {
                                    Image(systemName: "checkmark.circle.fill")
                                    Text("I Accept the Privacy Policy")
                                        .fontWeight(.semibold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                            }

                            Button(action: {
                                // Exit app if user declines
                                dismiss()
                            }) {
                                Text("Decline and Exit")
                                    .foregroundColor(.red)
                                    .font(.subheadline)
                            }

                            Text("You must accept the privacy policy to use this app.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                        .padding(.top, 8)
                    }
                }
                .padding()
            }
            .navigationTitle("Privacy Policy")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if !isFirstLaunch {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Done") {
                            dismiss()
                        }
                    }
                }
            }
        }
        .interactiveDismissDisabled(isFirstLaunch)
    }
}

// MARK: - Supporting Views

struct PolicySection<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                    .font(.title3)

                Text(title)
                    .font(.title3)
                    .fontWeight(.bold)
            }

            VStack(alignment: .leading, spacing: 8) {
                content
            }
            .font(.subheadline)
            .foregroundColor(.primary)
        }
    }
}

struct BulletPoint: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .fontWeight(.bold)
            Text(text)
        }
        .font(.subheadline)
        .foregroundColor(.secondary)
    }
}

struct DisclaimerCard: View {
    let icon: String
    let iconColor: Color
    let title: String
    let content: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(iconColor)
                .font(.title2)

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.bold)

                Text(content)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(iconColor.opacity(0.1))
        )
    }
}

// MARK: - Preview

#if DEBUG
struct PrivacyPolicyView_Previews: PreviewProvider {
    static var previews: some View {
        PrivacyPolicyView(hasAcceptedPolicy: .constant(false), isFirstLaunch: true)
    }
}
#endif
