//
//  OnnxModelHelper.swift
//  Onnx
//
//  Created by xfb on 2025/3/25.
//

class OnnxModelHelper {
    private let ortEnv: ORTEnv
    private let ortSession: ORTSession
    
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
    
    func generate(data:[[[Float]]]) -> [Float]? {
        let data = self.floatArrayToNSMutableData(data);
        var error: NSError? = nil
        guard let inputValue = Self.createOrtValueForFloat(data, error: &error) else {
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
        var floatArray: [Float]?

        outputData?.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) in
            let floatPointer = pointer.bindMemory(to: Float.self)
            floatArray = Array(floatPointer)
        }
        
        return floatArray
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
    
    static func createOrtValueForFloat(_ data:NSMutableData, error: inout NSError?) -> ORTValue? {
        // `data` will hold the memory of the input ORT value.
        // We set it to refer to the memory of the given float (*fp).
        
        // This will create a value with a tensor with the given float's data,
        // of type float, and with shape [1].
        do {
            let inputShape: [NSNumber] = [1,1,80]
            let ortValue = try ORTValue(tensorData: data, elementType: .float, shape: inputShape)
            return ortValue
        } catch let err as NSError {
            error = err
            return nil
        }
    }
}
