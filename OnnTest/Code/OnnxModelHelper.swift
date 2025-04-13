//
//  OnnxModelHelper.swift
//  Onnx
//
//  Created by xfb on 2025/3/25.
//

import NumiOS

class OnnxModelHelper {
    private let ortEnv: ORTEnv
    private let ortSession: ORTSession
    
    fileprivate var isUsing = false
    fileprivate var lock = NSLock()
    
    func getAndSetIsUsing() -> Bool {
        self.lock.lock()
        let isUsing = self.isUsing
        var setted = false
        if isUsing == false {
            self.isUsing = true
            setted = true
        }
        self.lock.unlock()
        return setted
    }
    
    func releaseIsUsing() {
        self.lock.lock()
        self.isUsing = false
        self.lock.unlock()
    }
    
    enum SpeechRecognizerError: Error {
      case Error(_ message: String)
    }
    
    init?() {
        do {
            ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            guard let modelPath = Bundle.main.path(forResource: "cnceleb_resnet34", ofType: "onnx") else {
              throw SpeechRecognizerError.Error("Failed to find model file.")
            }
            ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
        }catch {
            return nil
        }
        
    }
    
    func generate(data:[[Float]]) -> [Float]? {
        let swiftData = self.floatArrayToNSMutableData([data]);
        var error: NSError? = nil
        var shape = NumiOS.shape(data)
        shape.insert(1, at: 0)
        guard let inputValue = Self.createOrtValueForFloat(swiftData, shape: shape, error: &error) else {
            return nil
        }
        
        var outputs: [String: ORTValue]?
        do {
            outputs = try ortSession.run(withInputs: ["feats": inputValue], outputNames: ["embs"], runOptions: nil)
        } catch let err as NSError {
            error = err
            return nil
        }
        
        guard let output = outputs?["embs"] else {
          return nil
        }
        
        let outputData = try? output.tensorData() as Data
        
        // Convert Data to [Float]
        var floatArray: [Float]? =  self.dataToFloatArrayCopy(outputData ?? Data())
        
        return floatArray
    }
    
    func dataToFloatArrayCopy(_ data: Data) -> [Float] {
//        let floatArrayTest: [Float] = [1.5, 1.5, 1.5]
//        let data = Data(buffer: UnsafeBufferPointer(start: floatArrayTest, count: floatArrayTest.count))
        let floatCount = data.count / MemoryLayout<Float>.size
        let floatArray = data.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))[0..<floatCount]
        }
        let result = Array(floatArray)
        return result
    }

    
    fileprivate func floatArrayToNSMutableData(_ floatArray: [[[Float]]]) -> NSMutableData {
        let mutableData = NSMutableData()
            
        for matrix in floatArray {
            for row in matrix {
                var rowCopy = row  // 需要一个可变数组
                mutableData.append(&rowCopy, length: rowCopy.count * MemoryLayout<Float>.size)
            }
        }
        
        return mutableData
    }
    
    static func createOrtValueForFloat(_ data:NSMutableData,shape:[Int], error: inout NSError?) -> ORTValue? {
        // `data` will hold the memory of the input ORT value.
        // We set it to refer to the memory of the given float (*fp).
        
        // This will create a value with a tensor with the given float's data,
        // of type float, and with shape [1].
        do {
            let inputShape: [NSNumber] = shape.map { NSNumber(value: $0) }
            let ortValue = try ORTValue(tensorData: data, elementType: .float, shape: inputShape)
            return ortValue
        } catch let err as NSError {
            error = err
            return nil
        }
    }
}
