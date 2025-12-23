//
//  KeychainManager.swift
//  ECGHealthKit
//
//  Secure storage manager for sensitive data using iOS Keychain.
//  Provides encrypted storage for API keys and other credentials.
//

import Foundation
import Security

/// Manages secure storage of sensitive data in the iOS Keychain
class KeychainManager {

    /// Shared singleton instance
    static let shared = KeychainManager()

    private init() {}

    // MARK: - Error Types

    enum KeychainError: Error {
        case duplicateItem
        case itemNotFound
        case invalidData
        case unhandledError(status: OSStatus)

        var localizedDescription: String {
            switch self {
            case .duplicateItem:
                return "Item already exists in keychain"
            case .itemNotFound:
                return "Item not found in keychain"
            case .invalidData:
                return "Invalid data format"
            case .unhandledError(let status):
                return "Keychain error: \(status)"
            }
        }
    }

    // MARK: - Public Methods

    /// Save a string value securely to Keychain
    /// - Parameters:
    ///   - value: The string value to save
    ///   - key: The identifier key for this value
    /// - Throws: KeychainError if the operation fails
    func save(_ value: String, forKey key: String) throws {
        guard let data = value.data(using: .utf8) else {
            throw KeychainError.invalidData
        }

        try save(data, forKey: key)
    }

    /// Save data securely to Keychain
    /// - Parameters:
    ///   - data: The data to save
    ///   - key: The identifier key for this data
    /// - Throws: KeychainError if the operation fails
    func save(_ data: Data, forKey key: String) throws {
        // First, try to delete any existing item with this key
        try? delete(key)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            if status == errSecDuplicateItem {
                throw KeychainError.duplicateItem
            }
            throw KeychainError.unhandledError(status: status)
        }
    }

    /// Retrieve a string value from Keychain
    /// - Parameter key: The identifier key for the value
    /// - Returns: The stored string value, or nil if not found
    func getString(forKey key: String) -> String? {
        guard let data = getData(forKey: key) else {
            return nil
        }

        return String(data: data, encoding: .utf8)
    }

    /// Retrieve data from Keychain
    /// - Parameter key: The identifier key for the data
    /// - Returns: The stored data, or nil if not found
    func getData(forKey key: String) -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess else {
            return nil
        }

        return result as? Data
    }

    /// Update an existing value in Keychain
    /// - Parameters:
    ///   - value: The new string value
    ///   - key: The identifier key for this value
    /// - Throws: KeychainError if the operation fails
    func update(_ value: String, forKey key: String) throws {
        guard let data = value.data(using: .utf8) else {
            throw KeychainError.invalidData
        }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        let attributes: [String: Any] = [
            kSecValueData as String: data
        ]

        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)

        guard status == errSecSuccess else {
            if status == errSecItemNotFound {
                // If item doesn't exist, create it
                try save(value, forKey: key)
                return
            }
            throw KeychainError.unhandledError(status: status)
        }
    }

    /// Delete a value from Keychain
    /// - Parameter key: The identifier key for the value to delete
    /// - Throws: KeychainError if the operation fails
    func delete(_ key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unhandledError(status: status)
        }
    }

    /// Check if a key exists in Keychain
    /// - Parameter key: The identifier key to check
    /// - Returns: True if the key exists, false otherwise
    func exists(key: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: false
        ]

        let status = SecItemCopyMatching(query as CFDictionary, nil)
        return status == errSecSuccess
    }

    /// Delete all items from Keychain (use with caution)
    /// - Throws: KeychainError if the operation fails
    func deleteAll() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword
        ]

        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unhandledError(status: status)
        }
    }
}

// MARK: - Convenience Extensions

extension KeychainManager {

    /// Keychain keys used throughout the app
    enum Keys {
        static let apiKey = "com.cyclingecg.apikey"
        static let customAPIURL = "com.cyclingecg.customapiurl"
        static let encryptionKey = "com.cyclingecg.encryptionkey"
    }

    /// Save API key securely
    func saveAPIKey(_ apiKey: String) throws {
        try save(apiKey, forKey: Keys.apiKey)
    }

    /// Retrieve API key
    func getAPIKey() -> String? {
        return getString(forKey: Keys.apiKey)
    }

    /// Delete API key
    func deleteAPIKey() throws {
        try delete(Keys.apiKey)
    }

    /// Save custom API URL securely
    func saveCustomAPIURL(_ url: String) throws {
        try save(url, forKey: Keys.customAPIURL)
    }

    /// Retrieve custom API URL
    func getCustomAPIURL() -> String? {
        return getString(forKey: Keys.customAPIURL)
    }

    /// Get or create encryption key for local data encryption
    /// - Returns: 32-byte encryption key
    func getOrCreateEncryptionKey() throws -> Data {
        if let existingKey = getData(forKey: Keys.encryptionKey) {
            return existingKey
        }

        // Generate new 256-bit (32-byte) encryption key
        var keyData = Data(count: 32)
        let result = keyData.withUnsafeMutableBytes { bytes in
            SecRandomCopyBytes(kSecRandomDefault, 32, bytes.baseAddress!)
        }

        guard result == errSecSuccess else {
            throw KeychainError.unhandledError(status: result)
        }

        try save(keyData, forKey: Keys.encryptionKey)
        return keyData
    }
}
