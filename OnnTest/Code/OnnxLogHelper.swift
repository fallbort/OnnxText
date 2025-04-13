//
//  OnnxLogHelper.swift
//  OnnTest
//
//  Created by xfb on 2025/4/13.
//

class OnnxLogHelper {
     static func log(_ message: String) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        let timestamp = formatter.string(from: Date())
        print("[\(timestamp)] \(message)\n")
    }
}
