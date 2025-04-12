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
        let rows = matrix.count
        let cols = matrix[0].count
        var result = matrix

        for i in 0..<rows {
            // **填充第一列**（复制第一列的值）
            var paddedRow = [matrix[i][0]] + matrix[i]

            // 计算预加重：output[i] = input[i] - coefficient * input[i-1]
            for j in 0..<cols {
                result[i][j] = paddedRow[j + 1] - coefficient * paddedRow[j]
            }
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
        guard !inputData.isEmpty, let rowLength = inputData.first?.count else {
            return []
        }

        // 确保每行的数据长度是 2 的幂
        guard (rowLength & (rowLength - 1)) == 0 else {
            fatalError("输入数据的每行长度必须是 2 的幂")
        }

        let log2n = vDSP_Length(log2(Float(rowLength)))
        let halfCount = rowLength / 2
        let outputCount = halfCount + 1  // Python rfft 返回 N/2+1 个点

        // **去掉 scaleFactor 归一化**
        let scaleFactor: Float = 1.0 // Python `rfft` 默认不缩放

        // 创建 FFT 设置
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("FFT setup 失败")
        }

        var outputData: [[Float]] = []

        for row in inputData {
            var realParts = [Float](repeating: 0, count: halfCount)
            var imagParts = [Float](repeating: 0, count: halfCount)
            var dspSplitComplex = DSPSplitComplex(realp: &realParts, imagp: &imagParts)

            // 复制输入数据，并转换为复数格式
            var rowCopy = row
            rowCopy.withUnsafeMutableBufferPointer { buffer in
                buffer.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfCount) { complexPointer in
                    vDSP_ctoz(complexPointer, 2, &dspSplitComplex, 1, vDSP_Length(halfCount))
                }
            }

            // 执行 FFT
            vDSP_fft_zrip(fftSetup, &dspSplitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

            // 计算幅度谱（模值）
            var magnitudes = [Float](repeating: 0, count: outputCount)

            // 计算 DC 分量（第 0 个点）
            let valueZero = abs(dspSplitComplex.realp[0]) * scaleFactor
            magnitudes[0] = valueZero / 2.0 //python的值小了一半，人工减半

            // 计算 Nyquist 频率分量（最后一个点）
            magnitudes[halfCount] = abs(dspSplitComplex.imagp[0]) * scaleFactor

            // 计算 1 到 halfCount-1 的频率分量
            for i in 1..<halfCount {
                let real = dspSplitComplex.realp[i]
                let imag = dspSplitComplex.imagp[i]
                let value = sqrt(real * real + imag * imag) * scaleFactor
                magnitudes[i] = value / 2.0 //python的值小了一半，人工减半
            }

            outputData.append(magnitudes)
        }

        // 释放 FFT 资源
        vDSP_destroy_fftsetup(fftSetup)

        return outputData
    }
    
    func computeFFT(inputData: [Float]) -> [Float] {
        let count = inputData.count
        guard count > 1, (count & (count - 1)) == 0 else {
            fatalError("输入数据的长度必须是 2 的幂")
        }
        
        let log2n = vDSP_Length(log2(Float(count)))
        let halfCount = count / 2 + 1

        // 创建 FFT 设置
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("FFT setup 失败")
        }
        
        // 创建输入缓冲区，并应用窗口函数（例如汉宁窗）
        var windowedInput = inputData
        var window = [Float](repeating: 0, count: count)
        vDSP_hann_window(&window, vDSP_Length(count), Int32(0))
        vDSP_vmul(inputData, 1, window, 1, &windowedInput, 1, vDSP_Length(count))

        // 将输入数据转换为复数格式
        var realParts = [Float](repeating: 0, count: halfCount)
        var imagParts = [Float](repeating: 0, count: halfCount)
        var dspSplitComplex = DSPSplitComplex(realp: &realParts, imagp: &imagParts)

        windowedInput.withUnsafeBufferPointer { buffer in
            buffer.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfCount) { complexPointer in
                vDSP_ctoz(complexPointer, 2, &dspSplitComplex, 1, vDSP_Length(halfCount))
            }
        }

        // 执行 FFT
        vDSP_fft_zrip(fftSetup, &dspSplitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

        // 归一化 FFT 结果
        var scale: Float = 1.0 / Float(count)
        vDSP_vsmul(dspSplitComplex.realp, 1, &scale, dspSplitComplex.realp, 1, vDSP_Length(halfCount))
        vDSP_vsmul(dspSplitComplex.imagp, 1, &scale, dspSplitComplex.imagp, 1, vDSP_Length(halfCount))

        // 计算幅度谱
        var magnitudes = [Float](repeating: 0, count: halfCount)
        vDSP_zvmags(&dspSplitComplex, 1, &magnitudes, 1, vDSP_Length(halfCount))

        // 释放 FFT 设置
        vDSP_destroy_fftsetup(fftSetup)

        return magnitudes
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
    
    func log_____(data:[Float]?) {
//        NSLog("data: \(NumiOS.sum(data ?? [0]) as (Float,Float)),count = \(NumiOS.shape(data ?? []))")
    }
    func log_____(data:[[Float]]?) {
//        NSLog("data: \(NumiOS.sum(data ?? [0]) as (Float,Float)),count = \(NumiOS.shape(data ?? []))")
    }
}
