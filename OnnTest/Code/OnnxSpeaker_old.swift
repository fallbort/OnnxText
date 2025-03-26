//
//  OnnxSpeaker.swift
//  Onnx
//
//  Created by xfb on 2025/3/25.
//

import Foundation
import Accelerate

// MARK: - 假想的 NumPy 替代库（需手动替换）
struct NumPy {
    // 计算数组的均值
    static func mean(_ array: [Float]) -> Float {
        var result: Float = 0
        let count = vDSP_Length(array.count)
        
        // 使用vDSP_vmean来计算均值
        vDSP_meanv(array, 1, &result, count)
        
        return result
    }
    
    // 填充数组到指定大小，使用指定的值
    static func pad(_ array: [Float], to size: Int, with value: Float = 0) -> [Float] {
        var paddedArray = array
        
        // 如果数组长度小于目标大小，填充至目标大小
        if array.count < size {
            paddedArray.append(contentsOf: Array(repeating: value, count: size - array.count))
        }
        
        return paddedArray
    }
    
    // 将数组中的每个元素提取到指定的指数
    static func power(_ array: [Float], exponent: Float) -> [Float] {
        var result = [Float](repeating: 0, count: array.count)
        
        // 使用vDSP_vsq来计算每个元素的平方
        if exponent == 2 {
            vDSP_vsq(array, 1, &result, 1, vDSP_Length(array.count))
        } else {
            // 如果需要处理其他指数，可以在这里添加逻辑
            for i in 0..<array.count {
                result[i] = pow(array[i], exponent)
            }
        }
        
        return result
    }
}



// MARK: - Speaker 类
class OnnxSpeaker_old {
    let resampleRate: Int = 16000
    
    init() {
        let data1 = try! Data(contentsOf: Bundle.main.url(forResource: "wepeaker_window", withExtension: "pkl")!)
        let value1 = self.deserializeWithPlist(data1) as? Data
        let data2 = try! Data(contentsOf: Bundle.main.url(forResource: "wepeaker_mel_basis", withExtension: "pkl")!)
        let value2 = self.deserializeWithPlist(data2) as? Data
        self.value1 = self.dataToFloatArray(value1 ?? Data())
        let rowSize = 1  // 每行包含4个浮点数
        self.value2 = self.dataToFloat2DArray(value2 ?? Data(), rowSize: rowSize)
    }
    
    func extractEmbedding(from wav: [Float]) -> [Float] {
        var embeddings = [Float](repeating: 0, count: 256)
        let segmentSize = 16000 * 3
        var index = 0
        for i in stride(from: 0, to: wav.count, by: segmentSize) {
            let segment = Array(wav[i..<min(i + segmentSize, wav.count)])
            let weight = Float(segment.count) / 48000
            let paddedSegment = NumPy.pad(segment, to: segmentSize)
            let processor = OnnxFBankProcessor(melBasis: [[]], window: [Float]())
            let feats = processor.fbankAccelerate(paddedSegment)
            let featsNormalized = meanNormalize(feats)
            let featsAdded = addBatchDimension(featsNormalized)
            
            let embedding = helper?.generate(data: featsAdded)?.first
            if index < 256 {
                embeddings[index] = (embedding ?? 0) * weight
            }
            index += 1
        }
        
        return embeddings
    }
    
    
    func meanNormalize(_ matrix: [[Float]]) -> [[Float]] {
        let rowCount = matrix.count
        let colCount = matrix.first?.count ?? 0
        guard rowCount > 0, colCount > 0 else { return matrix }

        var meanValues = [Float](repeating: 0, count: colCount)

        // 计算每列的均值
        for col in 0..<colCount {
            let columnValues = matrix.map { $0[col] }
            var mean: Float = 0
            vDSP_meanv(columnValues, 1, &mean, vDSP_Length(rowCount))
            meanValues[col] = mean
        }

        // 进行均值归一化
        var normalizedMatrix = matrix
        for i in 0..<rowCount {
            for j in 0..<colCount {
                normalizedMatrix[i][j] -= meanValues[j]
            }
        }
        
        return normalizedMatrix
    }

    // 增加 batch 维度
    func addBatchDimension(_ matrix: [[Float]]) -> [[[Float]]] {
        return [matrix] // 直接用 Swift 数组即可
    }
    
    func cosineSimilarity(_ v1: [Float], _ v2: [Float]) -> Float {
        let dotProduct = zip(v1, v2).map(*).reduce(0, +)
        let magnitude1 = sqrt(v1.map { $0 * $0 }.reduce(0, +))
        let magnitude2 = sqrt(v2.map { $0 * $0 }.reduce(0, +))
        return dotProduct / (magnitude1 * magnitude2)
    }
    
    func run(_ wav: [Float]) -> Float {
        let segmentSize = 16000 * 6
        var sentenceEmbeddings = [[Float]]()
        var sentenceRMSSWeight = [Float]()
        
        for i in stride(from: 0, to: wav.count, by: segmentSize) {
            let segment = Array(wav[i..<min(i + segmentSize, wav.count)])
            let embedding = extractEmbedding(from: segment)
            let normalizedEmbedding = embedding.map { $0 / sqrt(embedding.map { $0 * $0 }.reduce(0, +)) }
            sentenceEmbeddings.append(normalizedEmbedding)
            sentenceRMSSWeight.append(NumPy.mean(toRMSTrack(segment)))
        }
        
        let simMatrix = sentenceEmbeddings.map { row in
            sentenceEmbeddings.map { cosineSimilarity(row, $0) }
        }
        
        let maxClusterIndex = simMatrix.map { $0.filter { $0 > 0.6 }.count }.enumerated().max { $0.element < $1.element }?.offset ?? 0
        var kernelEmbedding = sentenceEmbeddings[maxClusterIndex]
        
        for _ in 0..<5 {
            let filteredEmbeddings = sentenceEmbeddings.enumerated().filter { simMatrix[maxClusterIndex][$0.offset] > 0.6 }.map { $0.element }
            kernelEmbedding = filteredEmbeddings.reduce(Array(repeating: 0, count: 256)) { acc, vec in
                zip(acc, vec).map(+)
            }.map { $0 / Float(filteredEmbeddings.count) }
        }
        
        let kernelDistances = sentenceEmbeddings.map { cosineSimilarity(kernelEmbedding, $0) }
        let purityZip = zip(kernelDistances, sentenceRMSSWeight)
        let purityZipMapped = purityZip.filter { $0.0 > 0.6 }.map { $0.1 }.reduce(0, +)
        let purity = purityZipMapped / sentenceRMSSWeight.reduce(0, +)
        
        return purity
    }
    
    func toRMSTrack(_ wav: [Float], windowSize: Int = 1024, hopSize: Int = 160) -> [Float] {
        var rmss = [Float]()
        
        for i in stride(from: 0, to: wav.count, by: hopSize) {
            let segment = Array(wav[i..<min(i + windowSize, wav.count)])
            let meanSquare = segment.map { $0 * $0 }.reduce(0, +) / Float(segment.count)
            rmss.append(sqrt(meanSquare))
        }
        
        return rmss
    }
    
    
    
    func deserializeWithPlist(_ data: Data) -> Any? {
        return try? PropertyListSerialization.propertyList(from: data, format: nil)
    }
    
    func dataToFloatArray(_ data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self).prefix(count))
        }
    }
    
    func dataToFloat2DArray(_ data: Data, rowSize: Int) -> [[Float]] {
        let totalFloats = data.count / MemoryLayout<Float>.size
        let rows = totalFloats / rowSize
        var result: [[Float]] = []
        
        data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) in
            let floatPointer = pointer.bindMemory(to: Float.self)
            for i in 0..<rows {
                let startIndex = i * rowSize
                let endIndex = startIndex + rowSize
                let row = Array(floatPointer[startIndex..<endIndex])
                result.append(row)
            }
        }
        
        return result
    }
    
    fileprivate lazy var helper = OnnxModelHelper()
    
    fileprivate var value1: [Float]?
    fileprivate var value2: [[Float]]?
}
