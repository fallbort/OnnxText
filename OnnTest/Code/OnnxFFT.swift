//
//  OnnxFFT.swift
//  OnnTest
//
//  Created by xfb on 2025/3/27.
//

import Foundation
import Accelerate

class OnnxFFT{
    
    private var mSpectrumAnalysis: FFTSetup?
    private var mDspSplitComplex: DSPSplitComplex
    private var mFFTNormFactor: Float32
    private var mFFTLength: vDSP_Length
    private var mLog2N: vDSP_Length
    
    
    private final var kAdjust0DB: Float32 = 1.5849e-13
    
    init(maxFramesPerSlice inMaxFramesPerSlice: Int) {
        mSpectrumAnalysis = nil
        mFFTNormFactor = 1.0/Float32(2*inMaxFramesPerSlice)
        mFFTLength = vDSP_Length(inMaxFramesPerSlice)/2
        //log2を求めている、leadingZeroBitCountは先頭から0の数を数える,1引いたものから数えて32から引くと、ビット的にlog2がわかる
        mLog2N = vDSP_Length(32 - UInt32((UInt32(inMaxFramesPerSlice) - 1).leadingZeroBitCount))
        mDspSplitComplex = DSPSplitComplex(
            realp: UnsafeMutablePointer.allocate(capacity: Int(mFFTLength)),
            imagp: UnsafeMutablePointer.allocate(capacity: Int(mFFTLength))
        )
        mSpectrumAnalysis = vDSP_create_fftsetup(mLog2N, FFTRadix(kFFTRadix2))
    }
    
    
    deinit {
        vDSP_destroy_fftsetup(mSpectrumAnalysis)
        mDspSplitComplex.realp.deallocate()
        mDspSplitComplex.imagp.deallocate()
    }
    
    
    //inAudioDataをFFTしたデータをoutFFTDataに格納
    func computeFFT(_ inAudioData: UnsafePointer<Float32>?, outFFTData: UnsafeMutablePointer<Float32>?) {
        guard
            let inAudioData = inAudioData,
            let outFFTData = outFFTData
        else { return }
        
        //make window (fft size)
        let mFFTFulLength: vDSP_Length = mFFTLength * 2
        typealias FloatPointer = UnsafeMutablePointer<Float32>
        let window = FloatPointer.allocate(capacity: Int(mFFTFulLength))
        
        //blackman window
        vDSP_blkman_window(window, mFFTFulLength, 0)
        
        //windowing
        var windowAudioData = UnsafeMutablePointer<Float32>.allocate(capacity: Int(mFFTFulLength))
        
        vDSP_vmul(inAudioData, 1, window, 1, windowAudioData, 1, mFFTFulLength)
        
        //Complex型に変換
        windowAudioData.withMemoryRebound(to: DSPComplex.self, capacity: Int(mFFTLength)) {inAudioDataPtr in
            vDSP_ctoz(inAudioDataPtr, 2, &mDspSplitComplex, 1, mFFTLength)
        }

        
        //Take the fft and scale appropriately
        //fft
        vDSP_fft_zrip(mSpectrumAnalysis!, &mDspSplitComplex, 1, mLog2N, FFTDirection(kFFTDirection_Forward))
        //実数と虚数にmFFTNormFactorをかける
        vDSP_vsmul(mDspSplitComplex.realp, 1, &mFFTNormFactor, mDspSplitComplex.realp, 1, mFFTLength)
        vDSP_vsmul(mDspSplitComplex.imagp, 1, &mFFTNormFactor, mDspSplitComplex.imagp, 1, mFFTLength)
        
        //Zero out the nyquist value
        mDspSplitComplex.imagp[0] = 0.0
        
        //Convert the fft data to dB  ルートの実数^2 + 虚数^2
        vDSP_zvmags(&mDspSplitComplex, 1, outFFTData, 1, mFFTLength)
        

    }
    
}
