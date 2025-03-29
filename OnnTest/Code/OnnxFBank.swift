//
//  OnnxFBank.swift
//  OnnTest
//
//  Created by xfb on 2025/3/27.
//

import Accelerate

struct OnnxFBank {
    let sampleRate: Double
    let frameLength: Int // 通常25ms
    let frameShift: Int  // 通常10ms
    let numFilters: Int  // 滤波器数量，通常40
    let windowType: WindowType
    let useEnergy: Bool
    let useLog: Bool
    
    enum WindowType {
        case hamming
        case hanning
    }
    
    init(sampleRate: Double = 16000.0,
         frameLength: Int = 400,  // 16000 * 0.025 = 400
         frameShift: Int = 160,   // 16000 * 0.010 = 160
         numFilters: Int = 40,
         windowType: WindowType = .hamming,
         useEnergy: Bool = true,
         useLog: Bool = true) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.frameShift = frameShift
        self.numFilters = numFilters
        self.windowType = windowType
        self.useEnergy = useEnergy
        self.useLog = useLog
    }
    
    // 提取FBANK特征
    func extract(from signal: [Float]) -> [[Float]] {
        // 1. 预加重
        let preEmphasized = preEmphasis(signal: signal, coefficient: 0.97)
        
        // 2. 分帧
        let frames = framing(signal: preEmphasized)
        
        // 3. 加窗
        let windowedFrames = windowing(frames: frames)
        
        // 4. 计算功率谱
        let powerSpectrum = computePowerSpectrum(frames: windowedFrames)
        
        // 5. 应用梅尔滤波器组
        let filterBanks = applyMelFilterBank(powerSpectrum: powerSpectrum)
        
        return filterBanks
    }
    
    // 预加重
    private func preEmphasis(signal: [Float], coefficient: Float) -> [Float] {
        var result = [Float](repeating: 0.0, count: signal.count)
        result[0] = signal[0]
        
        for i in 1..<signal.count {
            result[i] = signal[i] - coefficient * signal[i-1]
        }
        
        return result
    }
    
    // 分帧
    private func framing(signal: [Float]) -> [[Float]] {
        let numFrames = 1 + (signal.count - frameLength) / frameShift
        var frames = [[Float]]()
        
        for i in 0..<numFrames {
            let start = i * frameShift
            let end = start + frameLength
            if end <= signal.count {
                let frame = Array(signal[start..<end])
                frames.append(frame)
            }
        }
        
        return frames
    }
    
    // 加窗
    private func windowing(frames: [[Float]]) -> [[Float]] {
        var windowedFrames = [[Float]]()
        let window = createWindow(length: frameLength)
        
        for frame in frames {
            var windowedFrame = [Float](repeating: 0.0, count: frame.count)
            vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(frame.count))
            windowedFrames.append(windowedFrame)
        }
        
        return windowedFrames
    }
    
    // 创建窗函数
    private func createWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0.0, count: length)
        let twoPi = 2 * Float.pi
        
        for i in 0..<length {
            switch windowType {
            case .hamming:
                window[i] = 0.54 - 0.46 * cos(twoPi * Float(i) / Float(length - 1))
            case .hanning:
                window[i] = 0.5 - 0.5 * cos(twoPi * Float(i) / Float(length - 1))
            }
        }
        
        return window
    }
    
    // 计算功率谱
    private func computePowerSpectrum(frames: [[Float]]) -> [[Float]] {
        let fftLength = frameLength.nextPowerOfTwo()
        let halfLength = fftLength / 2 + 1
        
        var powerSpectrum = [[Float]]()
        
        for frame in frames {
            // 零填充
            var paddedFrame = [Float](repeating: 0.0, count: fftLength)
            for i in 0..<min(frame.count, fftLength) {
                paddedFrame[i] = frame[i]
            }
            
            // 执行FFT
            var real = [Float](paddedFrame)
            var imaginary = [Float](repeating: 0.0, count: fftLength)
            var splitComplex = DSPSplitComplex(realp: &real, imagp: &imaginary)
            
            let log2n = vDSP_Length(log2(Float(fftLength)))
            let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(FFT_RADIX2))!
            
            vDSP_fft_zip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
            
            vDSP_destroy_fftsetup(fftSetup)
            
            // 计算功率谱
            var magnitudes = [Float](repeating: 0.0, count: halfLength)
            vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfLength))
            
            powerSpectrum.append(magnitudes)
        }
        
        return powerSpectrum
    }
    
    // 创建梅尔滤波器组
    private func createMelFilterBank() -> [[Float]] {
        let fftLength = frameLength.nextPowerOfTwo()
        let halfLength = fftLength / 2 + 1
        
        let minMel = hzToMel(0)
        let maxMel = hzToMel(Float(sampleRate / 2))
        
        let melPoints = linspace(min: minMel, max: maxMel, num: numFilters + 2)
        let hzPoints = melPoints.map { melToHz($0) }
        
        var bin = hzPoints.map { $0 * Float(fftLength) / Float(sampleRate) }
        bin = bin.map { min($0, Float(halfLength - 1)) }
        
        var filterBank = [[Float]](repeating: [Float](repeating: 0.0, count: halfLength), count: numFilters)
        
        for i in 1...numFilters {
            let left = Int(bin[i-1])
            let center = Int(bin[i])
            let right = Int(bin[i+1])
            
            // 上升斜坡
            for k in left..<center {
                filterBank[i-1][k] = Float(k - left) / Float(center - left)
            }
            
            // 下降斜坡
            for k in center..<right {
                filterBank[i-1][k] = Float(right - k) / Float(right - center)
            }
        }
        
        return filterBank
    }
    
    // 应用梅尔滤波器组
    private func applyMelFilterBank(powerSpectrum: [[Float]]) -> [[Float]] {
        let filterBank = createMelFilterBank()
        var filterBanks = [[Float]]()
        
        for spectrum in powerSpectrum {
            var fb = [Float](repeating: 0.0, count: numFilters)
            
            for i in 0..<numFilters {
                var sum: Float = 0.0
                for j in 0..<spectrum.count {
                    sum += spectrum[j] * filterBank[i][j]
                }
                fb[i] = sum
            }
            
            // 对数压缩
            if useLog {
                fb = fb.map { log($0 + Float.leastNormalMagnitude) }
            }
            
            // 如果需要，添加能量作为第一个特征
            if useEnergy {
                let energy = spectrum.reduce(0, +)
                fb.insert(energy, at: 0)
            }
            
            filterBanks.append(fb)
        }
        
        return filterBanks
    }
    
    // 辅助函数: Hz转Mel
    private func hzToMel(_ hz: Float) -> Float {
        return 2595 * log10(1 + hz / 700)
    }
    
    // 辅助函数: Mel转Hz
    private func melToHz(_ mel: Float) -> Float {
        return 700 * (pow(10, mel / 2595) - 1)
    }
    
    // 辅助函数: 线性空间
    private func linspace(min: Float, max: Float, num: Int) -> [Float] {
        let step = (max - min) / Float(num - 1)
        return (0..<num).map { min + Float($0) * step }
    }
}

extension Int {
    // 计算下一个2的幂
    func nextPowerOfTwo() -> Int {
        var value = self - 1
        value |= value >> 1
        value |= value >> 2
        value |= value >> 4
        value |= value >> 8
        value |= value >> 16
        return value + 1
    }
}
