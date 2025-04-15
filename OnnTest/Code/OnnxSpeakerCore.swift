//
//  OnnxSpeakerCore.swift
//  OnnTest
//
//  Created by xfb on 2025/3/25.
//

import Foundation
import Accelerate
import NumiOS

class OnnxSpeakerCore {
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
        if let windowData = Self.readJSONTwo(fileName: "wepeaker_window"),
           let melBasisData = Self.readJSONTwo(fileName: "wepeaker_mel_basis") {
            self.window = windowData
            self.melBasis = melBasisData
        }
    }
    
    static func readJSONTwo(fileName: String) -> [[Float]]? {
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
    
    static func readJSONOne(fileName: String) -> [Float]? {
        // 读取 JSON 文件并解析为二维浮点数组
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []),
              let array = json as? [Double]  else {
            return nil
        }
        let floatArray = array.map { Float($0) }
        return floatArray
    }
    
    func extractEmbedding(fromWav wav: [Float]) -> [Float] {
        self.log_____(data: wav)
        var embeddings = [Float](repeating: 0.0, count: 256)
        var index = 0
        let listCount = stride(from: 0, to: wav.count, by: 16000 * 3)
        for i in listCount {
            let wavSegment = Array(wav[i..<min(i + 16000 * 3, wav.count)])
            let weight = Float(wavSegment.count) / 48000.0
            var paddedWav = padWav(wavSegment, toLength: 16000 * 3)
            paddedWav = scaleWav(paddedWav, scale: 32768.0)
            self.log_____(data: paddedWav)
            let processor = OnnxFBankProcessor(melBasis: melBasis, window: window.first ?? [])
            let feats = processor.fbankAccelerate(paddedWav)
            self.log_____(data: feats)
            let normFeats = normalize(feats)
            self.log_____(data: normFeats)
            OnnxLogHelper.log("time test 32227")
            // 使用 ONNX 模型推理
            let onnxOutput = modelHelper?.generate(data: normFeats)
            OnnxLogHelper.log("time test 32228")
            self.log_____(data: onnxOutput)
            
            for outputIndex in 0..<(onnxOutput?.count ?? 0) {
                let embedding = onnxOutput?[outputIndex]
                embeddings[outputIndex] += (embedding ?? 0) * weight
            }
            index += 1
        }
        self.log_____(data: embeddings)
        return embeddings
    }
    
    func scaleWav(_ input: [Float], scale: Float) -> [Float] {
        var output = [Float](repeating: 0.0, count: input.count)
        var scalar = scale
        input.withUnsafeBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                vDSP_vsmul(inPtr.baseAddress!, 1, &scalar, outPtr.baseAddress!, 1, vDSP_Length(input.count))
            }
        }
        return output
    }

    func normalize(_ matrix: [[Float]]) -> [[Float]] {
        // 标准化矩阵
        let matrix = matrix
        self.log_____(data: matrix)
        let result = meanNormalize(feats: matrix)
        self.log_____(data: result)
        return result
    }
//    
//    func meanNormalize(feats: [[Float]]) -> [[Float]] {
//        if #available(iOS 14.6, *) {
//            // 适用于 iOS 14.6 及以上系统的代码
//            let rowCount = feats.count
//            let colCount = feats[0].count
//
//            // Step 1: 转置 feats 成 [col][row]，便于按列计算（并发）
//            var transposed = Array(repeating: [Float](repeating: 0, count: rowCount), count: colCount)
//            DispatchQueue.concurrentPerform(iterations: colCount) { col in
//                for row in 0..<rowCount {
//                    transposed[col][row] = feats[row][col]
//                }
//            }
//
//            // Step 2: 使用 vDSP 并发计算每列的平均值
//            var columnMeans = [Float](repeating: 0, count: colCount)
//            DispatchQueue.concurrentPerform(iterations: colCount) { j in
//                vDSP_meanv(transposed[j], 1, &columnMeans[j], vDSP_Length(rowCount))
//            }
//
//            // Step 3: 每列减去均值（向量加上 -mean），并发
//            DispatchQueue.concurrentPerform(iterations: colCount) { j in
//                var negMean = -columnMeans[j]
//                transposed[j].withUnsafeMutableBufferPointer { ptr in
//                    vDSP_vsadd(ptr.baseAddress!, 1, &negMean, ptr.baseAddress!, 1, vDSP_Length(rowCount))
//                }
//            }
//
//            // Step 4: 转置回来，变回 [row][col]，并发
//            var normalized = Array(repeating: [Float](repeating: 0, count: colCount), count: rowCount)
//            DispatchQueue.concurrentPerform(iterations: rowCount) { row in
//                for col in 0..<colCount {
//                    normalized[row][col] = transposed[col][row]
//                }
//            }
//
//            return normalized
//        } else {
//            // iOS 14.6 以下的系统
//            let rowCount = feats.count
//            let colCount = feats[0].count
//
//            // 1. 计算每一列的平均值
//            var columnMeans = [Float](repeating: 0, count: colCount)
//            for row in feats {
//                for j in 0..<colCount {
//                    columnMeans[j] += row[j]
//                }
//            }
//            for j in 0..<colCount {
//                columnMeans[j] /= Float(rowCount)
//            }
//
//            // 2. 每个元素减去对应列的均值
//            var normalized = feats
//            for i in 0..<rowCount {
//                for j in 0..<colCount {
//                    normalized[i][j] -= columnMeans[j]
//                }
//            }
//
//            return normalized
//        }
//    }
    
    func meanNormalize(feats: [[Float]]) -> [[Float]] {
        let rowCount = feats.count
        let colCount = feats[0].count

        // Step 1: 扁平化转置 feats -> [col][row]
        var transposedFlat = [Float](repeating: 0, count: rowCount * colCount)
        for row in 0..<rowCount {
            for col in 0..<colCount {
                transposedFlat[col * rowCount + row] = feats[row][col]
            }
        }

        // Step 2: 计算每列的均值（并发 + 避免 bounds-check）
        var columnMeans = [Float](repeating: 0, count: colCount)
        transposedFlat.withUnsafeMutableBytes { rawBuffer in
            let transPtr = rawBuffer.baseAddress!.assumingMemoryBound(to: Float.self)

            columnMeans.withUnsafeMutableBytes { meanRaw in
                let meanPtr = meanRaw.baseAddress!.assumingMemoryBound(to: Float.self)

                DispatchQueue.concurrentPerform(iterations: colCount) { col in
                    let start = col * rowCount
                    vDSP_meanv(transPtr.advanced(by: start), 1, meanPtr.advanced(by: col), vDSP_Length(rowCount))
                }
            }

            // Step 3: 每列减去均值（并发）
            columnMeans.withUnsafeBytes { meanRaw in
                let meanPtr = meanRaw.baseAddress!.assumingMemoryBound(to: Float.self)

                DispatchQueue.concurrentPerform(iterations: colCount) { col in
                    var negMean = -meanPtr[col]
                    let start = col * rowCount
                    vDSP_vsadd(transPtr.advanced(by: start), 1, &negMean, transPtr.advanced(by: start), 1, vDSP_Length(rowCount))
                }
            }
        }

        // Step 4: 扁平化转置回来 -> [row][col]
        var normalizedFlat = [Float](repeating: 0, count: rowCount * colCount)
        normalizedFlat.withUnsafeMutableBytes { normRaw in
            let normPtr = normRaw.baseAddress!.assumingMemoryBound(to: Float.self)

            transposedFlat.withUnsafeBytes { transRaw in
                let transPtr = transRaw.baseAddress!.assumingMemoryBound(to: Float.self)

                DispatchQueue.concurrentPerform(iterations: rowCount) { row in
                    for col in 0..<colCount {
                        normPtr[row * colCount + col] = transPtr[col * rowCount + row]
                    }
                }
            }
        }

        // Step 5: 拆分成 [[Float]]
        var normalized = [[Float]]()
        normalized.reserveCapacity(rowCount)
        for row in 0..<rowCount {
            let start = row * colCount
            let rowSlice = normalizedFlat[start..<start + colCount]
            normalized.append(Array(rowSlice))
        }

        return normalized
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
        OnnxLogHelper.log("time test 311")
        // 反射填充
        let padSize = windowSize / 2
        let leftPad = wav.prefix(padSize).reversed()
        let rightPad = wav.suffix(padSize).reversed()
        paddedWav = Array(leftPad) + wav + Array(rightPad)
        
        var rmss = [Float](repeating: 0, count: fNums)
        paddedWav.withUnsafeBufferPointer { ptr in
            for i in 0..<fNums {
                let start = i * hopSize
                let segmentPointer = ptr.baseAddress! + start
                var sumSquares: Float = 0.0
                vDSP_svesq(segmentPointer, 1, &sumSquares, vDSP_Length(windowSize))
                rmss[i] = sqrt(sumSquares / Float(windowSize))
            }
        }
        OnnxLogHelper.log("time test 313")
        return rmss
    }
    
    func run(wav: [Float]) -> Float {
        OnnxLogHelper.log("time test 31")
        let rmss = toRMSS(wav: wav)
        self.log_____(data: rmss)
        var sentenceEmbeddings = [[Float]]()
        var sentenceRmssWeight = [Float]()
        OnnxLogHelper.log("time test 32")
        let listCount = stride(from: 0, to: wav.count, by: 16000 * 6)
        for i in listCount {
            let wavPiece = Array(wav[i..<min(i + 16000 * 6, wav.count)])
            let embedding = extractEmbedding(fromWav: wavPiece)
            self.log_____(data: embedding)
            
            let sentenceEmbedding = normalize(embedding)
            sentenceEmbeddings.append(sentenceEmbedding)
            // 计算均值
            let start = i / 160             // start = 2
            let end = min(start + 600, rmss.count)  // max index = 602
            let slice = Array(rmss[start..<end])
            let avg = mean(slice)
            sentenceRmssWeight.append(avg)
            self.log_____(data: sentenceRmssWeight)
        }
        OnnxLogHelper.log("time test 33")
        // 计算相似度矩阵和核心嵌入向量的过程
        // 具体的实现会涉及线性代数计算，可以利用 Accelerate 或手动实现矩阵计算
        let simMatrix = computeSimilarityMatrix(from: sentenceEmbeddings)
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
        NSLog("final result purity check %@", "\(purity)")
        OnnxLogHelper.log("time test 37")
        return purity
    }
    
    
    func normalize(_ vector: [Float]) -> [Float] {
        var norm: Float = 0.0
        vDSP_svesq(vector, 1, &norm, vDSP_Length(vector.count))
        norm = sqrt(norm)
        return norm > 0 ? vector.map { $0 / norm } : vector
    }

    func mean(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0.0 }
        var result: Float = 0.0
        vDSP_meanv(values, 1, &result, vDSP_Length(values.count))
        return result
    }
    
    
    func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0.0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    func computeSimilarityMatrix(from sentenceEmbeddings: [[Float]]) -> [[Float]] {
        let count = sentenceEmbeddings.count
        var simMatrix = [[Float]](repeating: [Float](repeating: 0.0, count: count), count: count)

        for i in 0..<count {
            for j in i..<count {  // 优化：利用对称性，减少一半计算
                let sim = dotProduct(sentenceEmbeddings[i], sentenceEmbeddings[j])
                simMatrix[i][j] = sim
                simMatrix[j][i] = sim  // 对称矩阵
            }
        }

        return simMatrix
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
        let dim = embeddings[0].count
        var sum = [Float](repeating: 0.0, count: dim)
        var count = 0

        for i in 0..<kernelDis.count {
            if kernelDis[i] {
                let emb = embeddings[i]
                emb.withUnsafeBufferPointer { embPtr in
                    sum.withUnsafeMutableBufferPointer { sumPtr in
                        vDSP_vadd(sumPtr.baseAddress!, 1,
                                  embPtr.baseAddress!, 1,
                                  sumPtr.baseAddress!, 1,
                                  vDSP_Length(dim))
                    }
                }
                count += 1
            }
        }

        // 防止除以 0
        guard count > 0 else { return sum }

        var floatCount = Float(count)
        var result = [Float](repeating: 0.0, count: dim)
        sum.withUnsafeBufferPointer { sumPtr in
            result.withUnsafeMutableBufferPointer { resPtr in
                vDSP_vsdiv(sumPtr.baseAddress!, 1,
                           &floatCount,
                           resPtr.baseAddress!, 1,
                           vDSP_Length(dim))
            }
        }

        return result
    }

    func padWav(_ wav: [Float], toLength length: Int) -> [Float] {
        var padded = wav
        while padded.count < length {
            padded.append(0)
        }
        return padded
    }
    
    func log_____(data:[Float]?) {
//        NSLog("data: \(NumiOS.sum(data ?? [0]) as (Float,Float)),count = \(NumiOS.shape(data ?? []))")
    }
    func log_____(data:[[Float]]?) {
//        NSLog("data: \(NumiOS.sum(data ?? [0]) as (Float,Float)),count = \(NumiOS.shape(data ?? []))")
    }
}
