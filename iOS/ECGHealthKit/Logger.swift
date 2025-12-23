//
//  Logger.swift
//  ECGHealthKit
//
//  Secure logging utility that prevents PHI exposure in production builds
//

import Foundation
import os.log

/// Secure logger that only outputs in DEBUG builds
/// Prevents accidental PHI (Protected Health Information) exposure in production logs
class AppLogger {

    /// Shared singleton instance
    static let shared = AppLogger()

    private let osLog: OSLog

    private init() {
        osLog = OSLog(subsystem: Bundle.main.bundleIdentifier ?? "com.cyclingecg", category: "app")
    }

    // MARK: - Logging Levels

    /// Log debug information (only in DEBUG builds)
    /// - Parameters:
    ///   - message: The message to log
    ///   - category: Optional category for organizing logs
    ///   - file: Source file (automatically filled)
    ///   - function: Function name (automatically filled)
    ///   - line: Line number (automatically filled)
    func debug(_ message: String, category: String = "general", file: String = #file, function: String = #function, line: Int = #line) {
        #if DEBUG
        let fileName = (file as NSString).lastPathComponent
        os_log(.debug, log: osLog, "[%@:%d] %@ - %@", fileName, line, category.uppercased(), message)
        #endif
    }

    /// Log informational messages (only in DEBUG builds)
    /// - Parameters:
    ///   - message: The message to log
    ///   - category: Optional category for organizing logs
    func info(_ message: String, category: String = "general") {
        #if DEBUG
        os_log(.info, log: osLog, "[%@] %@", category.uppercased(), message)
        #endif
    }

    /// Log warnings (shown in both DEBUG and RELEASE with sanitization)
    /// - Parameters:
    ///   - message: The message to log
    ///   - category: Optional category for organizing logs
    func warning(_ message: String, category: String = "general") {
        #if DEBUG
        os_log(.error, log: osLog, "[%@] WARNING: %@", category.uppercased(), message)
        #else
        // In production, only log generic warnings without details
        os_log(.error, log: osLog, "[%@] WARNING: An issue occurred", category.uppercased())
        #endif
    }

    /// Log errors (shown in both DEBUG and RELEASE with sanitization)
    /// - Parameters:
    ///   - message: The error message to log
    ///   - error: Optional error object
    ///   - category: Optional category for organizing logs
    func error(_ message: String, error: Error? = nil, category: String = "general") {
        #if DEBUG
        if let error = error {
            os_log(.error, log: osLog, "[%@] ERROR: %@ - %@", category.uppercased(), message, error.localizedDescription)
        } else {
            os_log(.error, log: osLog, "[%@] ERROR: %@", category.uppercased(), message)
        }
        #else
        // In production, only log that an error occurred without details (no PHI)
        os_log(.error, log: osLog, "[%@] ERROR: An error occurred", category.uppercased())
        #endif
    }

    /// Log critical errors (always shown, even in production)
    /// - Parameters:
    ///   - message: The critical error message
    ///   - category: Optional category for organizing logs
    func critical(_ message: String, category: String = "general") {
        #if DEBUG
        os_log(.fault, log: osLog, "[%@] CRITICAL: %@", category.uppercased(), message)
        #else
        // In production, log that a critical error occurred
        os_log(.fault, log: osLog, "[%@] CRITICAL: A critical error occurred", category.uppercased())
        #endif
    }

    // MARK: - Category-Specific Loggers

    /// Log HealthKit-related events (NEVER logs PHI)
    /// - Parameter message: The message to log (must not contain PHI)
    func healthKit(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        #if DEBUG
        let fileName = (file as NSString).lastPathComponent
        os_log(.info, log: osLog, "[HEALTHKIT] %@:%d - %@", fileName, line, message)
        #endif
    }

    /// Log network-related events (NEVER logs request/response bodies with PHI)
    /// - Parameter message: The message to log (must not contain PHI)
    func network(_ message: String) {
        #if DEBUG
        os_log(.info, log: osLog, "[NETWORK] %@", message)
        #endif
    }

    /// Log storage-related events (NEVER logs data contents)
    /// - Parameter message: The message to log (must not contain data)
    func storage(_ message: String) {
        #if DEBUG
        os_log(.info, log: osLog, "[STORAGE] %@", message)
        #endif
    }

    /// Log security-related events
    /// - Parameter message: The message to log
    func security(_ message: String) {
        #if DEBUG
        os_log(.info, log: osLog, "[SECURITY] %@", message)
        #endif
    }

    /// Log analysis-related events (NEVER logs results or metrics)
    /// - Parameter message: The message to log (must not contain analysis results)
    func analysis(_ message: String) {
        #if DEBUG
        os_log(.info, log: osLog, "[ANALYSIS] %@", message)
        #endif
    }
}

// MARK: - Convenience Global Functions

#if DEBUG
/// Debug log (only in DEBUG builds)
func logDebug(_ message: String, category: String = "general", file: String = #file, function: String = #function, line: Int = #line) {
    AppLogger.shared.debug(message, category: category, file: file, function: function, line: line)
}

/// Info log (only in DEBUG builds)
func logInfo(_ message: String, category: String = "general") {
    AppLogger.shared.info(message, category: category)
}
#endif

/// Warning log (sanitized in production)
func logWarning(_ message: String, category: String = "general") {
    AppLogger.shared.warning(message, category: category)
}

/// Error log (sanitized in production)
func logError(_ message: String, error: Error? = nil, category: String = "general") {
    AppLogger.shared.error(message, error: error, category: category)
}

/// Critical error log (always shown)
func logCritical(_ message: String, category: String = "general") {
    AppLogger.shared.critical(message, category: category)
}

// MARK: - PHI Safety Guidelines
/*
 IMPORTANT: Never log the following in production:

 ❌ DO NOT LOG:
 - ECG voltage measurements or waveforms
 - Heart rate values
 - HRV measurements
 - Analysis results or metrics
 - Recording IDs or timestamps
 - API request/response bodies containing health data
 - User-entered symptoms or notes
 - Any personally identifiable health information (PHI)

 ✅ SAFE TO LOG (in DEBUG only):
 - Number of recordings fetched (count only)
 - Success/failure status of operations
 - Network connection status
 - File save/load success status
 - Generic error messages without details
 - UI state transitions

 ✅ SAFE TO LOG (even in production):
 - Critical system errors (without details)
 - App lifecycle events
 - Permission status changes (without data)

 Example of SAFE logging:
 ```
 logDebug("Successfully fetched \(count) ECG recordings")  // ✅ OK
 logInfo("Analysis completed successfully")                // ✅ OK
 logWarning("Backend connection failed, using fallback")   // ✅ OK
 ```

 Example of UNSAFE logging:
 ```
 print("ECG voltage: \(voltage) V")                        // ❌ NEVER
 print("Heart rate: \(hr) bpm")                            // ❌ NEVER
 print("Analysis result: \(json)")                         // ❌ NEVER
 print("Recording ID: \(id)")                              // ❌ NEVER
 ```
 */
