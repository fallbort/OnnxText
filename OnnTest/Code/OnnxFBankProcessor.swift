//
//  OnnxFBankProcessor.swift
//  OnnTest
//
//  Created by xfb on 2025/3/25.
//

import Accelerate

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

        // 2. **去均值**
        meanNormalize(&stridedInput)

        // 3. **Pre-emphasis 预加重**
        preemphasis(&stridedInput)

        // 4. **乘窗函数**
        applyWindow(&stridedInput, window: window)

        // 5. **填充**
        padInput(&stridedInput, targetSize: paddedWindowSize)

        // 6. **FFT 幅值谱**
        var spectrum = computeMagnitudeFFT(stridedInput, paddedWindowSize: paddedWindowSize)

        // 7. **功率谱**
        if usePower {
            spectrum = powerSpectrum(spectrum)
        }

        // 8. **Mel 滤波**
        var melEnergies = applyMelFilter(spectrum, melBasis: melBasis)

        // 9. **取 Log**
        logTransform(&melEnergies)

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
                var mean = mean // 需要可变变量
                vDSP_vsadd(buffer.baseAddress!, 1, &mean, buffer.baseAddress!, 1, vDSP_Length(buffer.count))
            }
        }
    }

    /// 预加重
    private func preemphasis(_ matrix: inout [[Float]]) {
        for i in 0..<matrix.count {
            for j in stride(from: matrix[i].count - 1, to: 0, by: -1) {
                matrix[i][j] -= preemphasisCoefficient * matrix[i][j - 1]
            }
        }
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

    /// 计算 FFT 幅值谱
    private func computeMagnitudeFFT(_ matrix: [[Float]], paddedWindowSize: Int) -> [[Float]] {
        let fftSize = paddedWindowSize / 2 + 1
        var output = [[Float]](repeating: [Float](repeating: 0, count: fftSize), count: matrix.count)

        let log2Size = vDSP_Length(log2(Float(paddedWindowSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2Size, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup")
        }

        for i in 0..<matrix.count {
            var real = matrix[i]
            var imaginary = [Float](repeating: 0, count: real.count)
            var splitComplex = DSPSplitComplex(realp: &real, imagp: &imaginary)

            // 执行 FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2Size, FFTDirection(kFFTDirection_Forward))

            // 计算功率谱（模长的平方）
            var powerSpectrum = [Float](repeating: 0, count: fftSize)
            vDSP_zvmags(&splitComplex, 1, &powerSpectrum, 1, vDSP_Length(fftSize))

            // 计算幅值谱（模长）
            var count = Int32(fftSize)
            powerSpectrum.withUnsafeMutableBufferPointer { ptr in
                output[i].withUnsafeMutableBufferPointer { outPtr in
                    vvsqrtf(outPtr.baseAddress!, ptr.baseAddress!, &count)
                }
            }
        }

        vDSP_destroy_fftsetup(fftSetup)
        return output
    }

    /// 计算功率谱
    private func powerSpectrum(_ matrix: [[Float]]) -> [[Float]] {
        return matrix.map { vDSP.multiply($0, $0) }
    }

    /// 应用 Mel 滤波器
    private func applyMelFilter(_ spectrum: [[Float]], melBasis: [[Float]]) -> [[Float]] {
        let numFrames = spectrum.count
        let numMelFilters = melBasis.count
        var melEnergies = [[Float]](repeating: [Float](repeating: 0, count: numMelFilters), count: numFrames)

        for i in 0..<numFrames {
            melEnergies[i] = matrixMultiply(melBasis, spectrum[i])
        }

        return melEnergies
    }

    /// 计算 log
    private func logTransform(_ matrix: inout [[Float]]) {
        let minValue: Float = 1.1921e-07
        for i in 0..<matrix.count {
            for j in 0..<matrix[i].count {
                matrix[i][j] = log(max(matrix[i][j], minValue))
            }
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
}
