//
//  OnnxSpeaker.swift
//  OnnTest
//
//  Created by xfb on 2025/3/25.
//

import Foundation
import Accelerate

class OnnxSpeaker {
    var resampleRate: Int = 16000
    var melBasis: [[Float]] = []
    var window: [[Float]] = []
    
    fileprivate lazy var modelHelper = OnnxModelHelper()
    
    init() {
        // 加载 JSON 文件中的 Mel 基础和 Window 数据
        loadWindowAndMelBasis()
    }
    
    func loadWindowAndMelBasis() {
        // 读取 window.json 和 mel_basis.json 文件
        if let windowData = readJSON(fileName: "wepeaker_window"),
           let melBasisData = readJSON(fileName: "wepeaker_mel_basis") {
            self.window = windowData
            self.melBasis = melBasisData
        }
    }
    
    func readJSON(fileName: String) -> [[Float]]? {
        // 读取 JSON 文件并解析为二维浮点数组
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []),
              let array = json as? [[Double]]  else {
            return nil
        }
        let floatArray = array.map {$0.map { Float($0) }}
        return floatArray
    }
    
    func extractEmbedding(fromWav wav: [Float]) -> [Float] {
        var embeddings = [Float](repeating: 0.0, count: 256)
        var index = 0
        for i in stride(from: 0, to: wav.count, by: 16000 * 3) {
            let wavSegment = Array(wav[i..<min(i + 16000 * 3, wav.count)])
            let weight = Float(wavSegment.count) / 48000.0
            let paddedWav = padWav(wavSegment, toLength: 16000 * 3)
            let processor = OnnxFBankProcessor(melBasis: melBasis, window: window.first ?? [])
            let feats = processor.fbankAccelerate(paddedWav)
            let normFeats = normalize(feats)
            
            // 使用 ONNX 模型推理
            
            let embedding = modelHelper?.generate(data: [normFeats])?.first
            if index < 256 {
                embeddings[index] = (embedding ?? 0) * weight
            }
            index += 1
        }
        return embeddings
    }

    func normalize(_ matrix: [[Float]]) -> [[Float]] {
        // 标准化矩阵
        var matrix = matrix
//        self.subtractColumnMean(feats: &matrix)
        return matrix  // 这里只是示范，实际实现需要标准化
    }
    
    func subtractColumnMean(feats: inout [[Float]]) {
        // 假设 feats 是一个二维数组，即数组的每个元素是 [Float] 数组
        let rowCount = feats.count
        let colCount = feats[0].count
        
        // 将 feats 转换为一维数组，以便使用 vDSP_vmean 计算均值
        var flattenedFeats = feats.flatMap { $0 }
        
        // 计算每列的均值
        var columnMeans = [Float](repeating: 0.0, count: colCount)
        for colIndex in 0..<colCount {
            let column = Array(feats.map { $0[colIndex] })
            columnMeans[colIndex] = mean(array: column)
        }
        
        // 从 feats 中减去对应列的均值
        for i in 0..<rowCount {
            for j in 0..<colCount {
                feats[i][j] -= columnMeans[j]
            }
        }
    }

    // 帮助函数：计算均值
    func mean(array: [Float]) -> Float {
        var sum: Float = 0.0
        
        // 使用 vDSP_vadd 来计算数组元素的和
        vDSP_vadd(array, 1, [0.0], 1, &sum, 1, vDSP_Length(array.count))
        
        // 计算均值
        return sum / Float(array.count)
    }
    
    
    func toRMSS(wav: [Float], hopSize: Int = 160, windowSize: Int = 1024) -> [Float] {
        let fNums = wav.count / hopSize
        var paddedWav = wav

        // 反射填充
        let padSize = windowSize / 2
        let leftPad = wav.prefix(padSize).reversed()
        let rightPad = wav.suffix(padSize).reversed()
        paddedWav = Array(leftPad) + wav + Array(rightPad)
        
        var rmss = [Float](repeating: 0, count: fNums)
        
        for i in 0..<fNums {
            let start = i * hopSize
            let end = start + windowSize
            let segment = paddedWav[start..<end]
            
            // 计算均方根（RMS）
            var sumSquares: Float = 0.0
            vDSP_svesq(segment.map { $0 }, 1, &sumSquares, vDSP_Length(windowSize))
            rmss[i] = sqrt(sumSquares / Float(windowSize))
        }
        
        return rmss
    }
    
    func run(wav: [Float]) -> Float {
        let rmss = toRMSS(wav: wav)
        var sentenceEmbeddings = [[Float]]()
        var sentenceRmssWeight = [Float]()
        
        for i in stride(from: 0, to: wav.count, by: 16000 * 6) {
            let wavPiece = Array(wav[i..<min(i + 16000 * 6, wav.count)])
            let embedding = extractEmbedding(fromWav: wavPiece)
            sentenceEmbeddings.append(embedding)
            sentenceRmssWeight.append(rmss[i / 160])
        }
        
        // 计算相似度矩阵和核心嵌入向量的过程
        // 具体的实现会涉及线性代数计算，可以利用 Accelerate 或手动实现矩阵计算
        var simMatrix = [[Float]](repeating: [Float](repeating: 0.0, count: sentenceEmbeddings.count), count: sentenceEmbeddings.count)
        for i in 0..<sentenceEmbeddings.count {
            for j in 0..<sentenceEmbeddings.count {
                simMatrix[i][j] = dotProduct(sentenceEmbeddings[i], sentenceEmbeddings[j])
            }
        }
        
        let maxCluster = findMaxCluster(simMatrix)
        var kernelEmbedding = sentenceEmbeddings[maxCluster].map { $0 / sqrt($0 * $0) }  // 归一化
        for _ in 0..<5 {
            var kernelDis = [Bool]()
            for i in 0..<sentenceEmbeddings.count {
                kernelDis.append(dotProduct(kernelEmbedding, sentenceEmbeddings[i]) > 0.6)
            }
            kernelEmbedding = updateKernelEmbedding(sentenceEmbeddings, kernelDis)
        }
        
        var kernelDis = [Float]()
        for i in 0..<sentenceEmbeddings.count {
            kernelDis.append(dotProduct(kernelEmbedding, sentenceEmbeddings[i]))
        }
        
        var purity: Float = 0
        for i in 0..<kernelDis.count {
            if kernelDis[i] > 0.6 {
                purity += sentenceRmssWeight[i]
            }
        }
        purity /= sentenceRmssWeight.reduce(0, +)
        
        return purity
    }
    
    func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0.0
        for i in 0..<a.count {
            result += a[i] * b[i]
        }
        return result
    }

    func findMaxCluster(_ simMatrix: [[Float]]) -> Int {
        var maxCluster = 0
        var maxCount = 0
        for i in 0..<simMatrix.count {
            let count = simMatrix[i].filter { $0 > 0.6 }.count
            if count > maxCount {
                maxCount = count
                maxCluster = i
            }
        }
        return maxCluster
    }

    func updateKernelEmbedding(_ embeddings: [[Float]], _ kernelDis: [Bool]) -> [Float] {
        var sum = [Float](repeating: 0.0, count: embeddings[0].count)
        var count = 0
        for i in 0..<kernelDis.count {
            if kernelDis[i] {
                for j in 0..<embeddings[i].count {
                    sum[j] += embeddings[i][j]
                }
                count += 1
            }
        }
        return sum.map { $0 / Float(count) }
    }

    func padWav(_ wav: [Float], toLength length: Int) -> [Float] {
        var padded = wav
        while padded.count < length {
            padded.append(0)
        }
        return padded
    }
}
