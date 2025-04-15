//
//  OnnxFBankProcessor.swift
//  OnnTest
//
//  Created by xfb on 2025/3/25.
//

import Accelerate
import NumiOS

struct OnnxFBankProcessor {
    let melBasis: [[Float]]
    let window: [Float]
    let preemphasisCoefficient: Float = 0.97
    let usePower: Bool = true
    let windowShift: Int = 160
    let windowSize: Int = 400
    let paddedWindowSize: Int = 512

    func fbankAccelerate(_ waveform: [Float]) -> [[Float]] {
        let numSamples = waveform.count
        let m = 1 + (numSamples - windowSize) / windowShift
        
        // 1. **创建滑动窗口输入 (strided_input)**
        var stridedInput: [[Float]] = strideInput(waveform, windowSize: windowSize, windowShift: windowShift)
        self.log_____(data: stridedInput)
        // 2. **去均值**
        meanNormalize(&stridedInput)
        self.log_____(data: stridedInput)

        // 3. **Pre-emphasis 预加重**
        stridedInput = preEmphasize(matrix: stridedInput, coefficient: preemphasisCoefficient)
        self.log_____(data: stridedInput)

        // 4. **乘窗函数**
        applyWindow(&stridedInput, window: window)
        self.log_____(data: stridedInput)

        // 5. **填充**
        padInput(&stridedInput, targetSize: paddedWindowSize)
        self.log_____(data: stridedInput)

        // 6. **FFT 幅值谱**
        var spectrum = computeFFT(inputData: stridedInput)
        self.log_____(data: spectrum)

        // 7. **功率谱**
        if usePower {
            spectrum = powerSpectrum(spectrum)
        }
        self.log_____(data: spectrum)
        // 8. **Mel 滤波**
        var melEnergies = applyMelFilter(spectrum, melBasis: melBasis)
        self.log_____(data: melEnergies)

        // 9. **取 Log**
        logTransform(&melEnergies)
        self.log_____(data: melEnergies)

        return melEnergies
    }

    /// 创建滑动窗口输入
    private func strideInput(_ waveform: [Float], windowSize: Int, windowShift: Int) -> [[Float]] {
        let m = (waveform.count - windowSize) / windowShift + 1
        var output = [[Float]](repeating: [Float](repeating: 0, count: windowSize), count: m)
        
        for i in 0..<m {
            let start = i * windowShift
            output[i] = Array(waveform[start..<start + windowSize])
        }
        
        return output
    }

    /// 对每一帧去均值
    private func meanNormalize(_ matrix: inout [[Float]]) {
        for i in 0..<matrix.count {
            let mean = vDSP.mean(matrix[i]) // 计算均值

            matrix[i].withUnsafeMutableBufferPointer { buffer in
                var mean = 0-mean // 需要可变变量
                vDSP_vsadd(buffer.baseAddress!, 1, &mean, buffer.baseAddress!, 1, vDSP_Length(buffer.count))
            }
        }
    }

    /// 预加重
    func preEmphasize(matrix: [[Float]], coefficient: Float) -> [[Float]] {
        self.log_____(data: matrix)
        let result = self.preEmphasizeParallel(matrix: matrix, coefficient: coefficient)
        self.log_____(data: result)
        return result
    }
    
    func preEmphasizeParallel(matrix: [[Float]], coefficient: Float) -> [[Float]] {
        let rowCount = matrix.count
        let colCount = matrix[0].count

        // Step 1: 扁平化结果数组
        var resultFlat = [Float](repeating: 0, count: rowCount * colCount)

        resultFlat.withUnsafeMutableBufferPointer { resultPtr in
            DispatchQueue.concurrentPerform(iterations: rowCount) { i in
                let input = matrix[i]
                let start = i * colCount
                resultPtr[start] = input[0]
                for j in 1..<colCount {
                    resultPtr[start + j] = input[j] - coefficient * input[j - 1]
                }
            }
        }

        // Step 2: 拆成 [[Float]]
        var result = [[Float]]()
        for i in 0..<rowCount {
            let start = i * colCount
            result.append(Array(resultFlat[start..<start + colCount]))
        }

        return result
    }

    /// 乘窗函数
    private func applyWindow(_ matrix: inout [[Float]], window: [Float]) {
        for i in 0..<matrix.count {
            vDSP.multiply(window, matrix[i], result: &matrix[i])
        }
    }

    /// 填充至 512 维
    private func padInput(_ matrix: inout [[Float]], targetSize: Int) {
        let paddingRight = targetSize - matrix[0].count
        for i in 0..<matrix.count {
            matrix[i] += [Float](repeating: 0, count: paddingRight)
        }
    }
    
    func computeFFT(inputData: [[Float]]) -> [[Float]] {
        return computeFFTConcurrent(inputData: inputData)
    }

    func computeSingleFFT(row: [Float], fftSetup: FFTSetup, log2n: vDSP_Length, halfCount: Int, outputCount: Int) -> [Float] {
        let scaleFactor: Float = 1.0 // Python `rfft` 默认不缩放

        var realParts = [Float](repeating: 0, count: halfCount)
        var imagParts = [Float](repeating: 0, count: halfCount)
        var dspSplitComplex = DSPSplitComplex(realp: &realParts, imagp: &imagParts)

        var rowCopy = row
        rowCopy.withUnsafeMutableBufferPointer { buffer in
            buffer.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfCount) { complexPointer in
                vDSP_ctoz(complexPointer, 2, &dspSplitComplex, 1, vDSP_Length(halfCount))
            }
        }

        vDSP_fft_zrip(fftSetup, &dspSplitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

        var magnitudes = [Float](repeating: 0, count: outputCount)
        magnitudes[0] = abs(dspSplitComplex.realp[0]) * scaleFactor / 2.0 // python的值小了一半，人工减半
        magnitudes[halfCount] = abs(dspSplitComplex.imagp[0]) * scaleFactor

        for i in 1..<halfCount {
            let real = dspSplitComplex.realp[i]
            let imag = dspSplitComplex.imagp[i]
            let value = sqrt(real * real + imag * imag) * scaleFactor
            magnitudes[i] = value / 2.0 // python的值小了一半，人工减半
        }

        return magnitudes
    }

    func computeFFTConcurrent(inputData: [[Float]]) -> [[Float]] {
        guard !inputData.isEmpty, let rowLength = inputData.first?.count else {
            return []
        }

        guard (rowLength & (rowLength - 1)) == 0 else {
            fatalError("输入数据的每行长度必须是 2 的幂")
        }

        let log2n = vDSP_Length(log2(Float(rowLength)))
        let halfCount = rowLength / 2
        let outputCount = halfCount + 1

        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("FFT setup 失败")
        }

        let rowCount = inputData.count
        var outputData = [[Float]](repeating: [], count: rowCount)
        let lock = NSLock() // 加锁避免并发写入导致崩溃

        DispatchQueue.concurrentPerform(iterations: rowCount) { i in
            let row = inputData[i]
            let scaleFactor: Float = 1.0 // Python `rfft` 默认不缩放

            var realParts = [Float](repeating: 0, count: halfCount)
            var imagParts = [Float](repeating: 0, count: halfCount)
            var dspSplitComplex = DSPSplitComplex(realp: &realParts, imagp: &imagParts)

            var rowCopy = row
            rowCopy.withUnsafeMutableBufferPointer { buffer in
                buffer.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfCount) { complexPointer in
                    vDSP_ctoz(complexPointer, 2, &dspSplitComplex, 1, vDSP_Length(halfCount))
                }
            }

            vDSP_fft_zrip(fftSetup, &dspSplitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

            var magnitudes = [Float](repeating: 0, count: outputCount)
            magnitudes[0] = abs(dspSplitComplex.realp[0]) * scaleFactor / 2.0 // python的值小了一半，人工减半
            magnitudes[halfCount] = abs(dspSplitComplex.imagp[0]) * scaleFactor

            for i in 1..<halfCount {
                let real = dspSplitComplex.realp[i]
                let imag = dspSplitComplex.imagp[i]
                let value = sqrt(real * real + imag * imag) * scaleFactor
                magnitudes[i] = value / 2.0 // python的值小了一半，人工减半
            }

            lock.lock()
            outputData[i] = magnitudes
            lock.unlock()
        }

        vDSP_destroy_fftsetup(fftSetup)

        return outputData
    }

    /// 计算功率谱
    private func powerSpectrum(_ matrix: [[Float]]) -> [[Float]] {
        return matrix.map { vDSP.multiply($0, $0) }
    }
    
    private func applyMelFilter(_ spectrum: [[Float]], melBasis: [[Float]]) -> [[Float]] {
        self.log_____(data: spectrum)
        let result = self.applyMelFilterParallel(spectrum, melBasis: melBasis)
        self.log_____(data: result)
        return result
    }
    
    private func applyMelFilterParallel(_ spectrum: [[Float]], melBasis: [[Float]]) -> [[Float]] {
        let numFrames = spectrum.count
        let numMelFilters = melBasis.count
        let fftSize = spectrum[0].count

        // 把 melBasis 转为扁平数组以供 vDSP 使用
        let flatMel = melBasis.flatMap { $0 }

        // 创建扁平输出数组
        var flatOutput = [Float](repeating: 0.0, count: numFrames * numMelFilters)

        // 多线程并行处理
        flatOutput.withUnsafeMutableBufferPointer { outPtr in
            flatMel.withUnsafeBufferPointer { melPtr in
                DispatchQueue.concurrentPerform(iterations: numFrames) { i in
                    let inputFrame = spectrum[i]
                    inputFrame.withUnsafeBufferPointer { inputPtr in
                        let resultPtr = outPtr.baseAddress! + i * numMelFilters

                        vDSP_mmul(
                            melPtr.baseAddress!, 1,
                            inputPtr.baseAddress!, 1,
                            resultPtr, 1,
                            vDSP_Length(numMelFilters),
                            1,
                            vDSP_Length(fftSize)
                        )
                    }
                }
            }
        }

        // 将扁平数组转换为二维数组
        var melEnergies = [[Float]](repeating: [], count: numFrames)
        for i in 0..<numFrames {
            let start = i * numMelFilters
            let end = start + numMelFilters
            melEnergies[i] = Array(flatOutput[start..<end])
        }

        return melEnergies
    }

    /// 高性能 Mel 滤波器计算：spectrum [frame][bin], melBasis [mel][bin]
    private func applyMelFilterAccelerated(_ spectrum: [[Float]], melBasis: [[Float]]) -> [[Float]] {
        let numFrames = spectrum.count
        let numMelFilters = melBasis.count
        let numBins = spectrum[0].count

        // 转置 melBasis: [mel][bin] -> [bin][mel] for vDSP_mmul
        var melBasisT = [[Float]](repeating: [Float](repeating: 0, count: numMelFilters), count: numBins)
        for i in 0..<numMelFilters {
            for j in 0..<numBins {
                melBasisT[j][i] = melBasis[i][j]
            }
        }

        // 展平数据
        let spectrumFlat = spectrum.flatMap { $0 }         // [frame * bin]
        let melBasisFlat = melBasisT.flatMap { $0 }        // [bin * mel]
        var output = [Float](repeating: 0, count: numFrames * numMelFilters)

        // 批量矩阵乘法：[frame × bin] x [bin × mel] = [frame × mel]
        vDSP_mmul(
            spectrumFlat, 1,
            melBasisFlat, 1,
            &output, 1,
            vDSP_Length(numFrames),
            vDSP_Length(numMelFilters),
            vDSP_Length(numBins)
        )

        // 重组为 [[Float]]
        var melEnergies = [[Float]]()
        for i in 0..<numFrames {
            let start = i * numMelFilters
            let end = start + numMelFilters
            melEnergies.append(Array(output[start..<end]))
        }

        return melEnergies
    }

    /// 高性能 log 变换，使用 vDSP + vForce 优化
    private func logTransform(_ matrix: inout [[Float]]) {
        let minValue: Float = 1.1921e-07

        for i in 0..<matrix.count {
            var row = matrix[i]
            var clipped = [Float](repeating: 0.0, count: row.count)

            // 裁剪小值，避免 log(0)
            var threshold = minValue
            vDSP_vthr(row, 1, &threshold, &clipped, 1, vDSP_Length(row.count))

            // 申请内存用于 log 输出
            var logged = [Float](repeating: 0.0, count: row.count)
            // 使用 vvlogf 批量计算 log（底层 SIMD 向量优化）
            logged.withUnsafeMutableBufferPointer { logPtr in
                clipped.withUnsafeBufferPointer { clipPtr in
                    vvlogf(logPtr.baseAddress!, clipPtr.baseAddress!, [Int32(row.count)])
                }
            }

            matrix[i] = logged
        }
    }

    /// 矩阵-向量乘法
    private func matrixMultiply(_ matrix: [[Float]], _ vector: [Float]) -> [Float] {
        let rows = matrix.count
        let cols = vector.count
        var result = [Float](repeating: 0, count: rows)

        for i in 0..<rows {
            result[i] = vDSP.dot(matrix[i], vector)
        }

        return result
    }
    
    func log_____(data:[Float]?) {
//        NSLog("data: \(NumiOS.sum(data ?? [0]) as (Float,Float)),count = \(NumiOS.shape(data ?? []))")
    }
    func log_____(data:[[Float]]?) {
//        NSLog("data: \(NumiOS.sum(data ?? [0]) as (Float,Float)),count = \(NumiOS.shape(data ?? []))")
    }
}
