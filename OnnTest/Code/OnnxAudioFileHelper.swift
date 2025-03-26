//
//  OnnxAudioFileHelper.swift
//  OnnTest
//
//  Created by xfb on 2025/3/25.
//

import AVFoundation

class OnnxAudioFileHelper {
    static func loadAudioFile(url: URL) -> [Float]? {
        do {
            let audioFile = try AVAudioFile(forReading: url)
            
            let format = audioFile.processingFormat
            let frameCount = UInt32(audioFile.length)
            
            // 读取音频数据到buffer
            let audioBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            try audioFile.read(into: audioBuffer!)
            
            // 获取音频数据（原始PCM数据）
            let channelCount = Int(format.channelCount)
            let sampleRate = format.sampleRate
            let numberOfFrames = Int(audioBuffer!.frameLength)
            
            var audioData = [Float]()
            
            // 如果是立体声，我们可以将左右声道合并为单声道
            for frame in 0..<numberOfFrames {
                var monoSample: Float = 0.0
                for channel in 0..<channelCount {
                    monoSample += audioBuffer!.floatChannelData![channel][frame]
                }
                audioData.append(monoSample / Float(channelCount))  // 合并为单声道（平均）
            }
            
            return audioData
        } catch {
            print("Error loading audio file: \(error)")
            return nil
        }
    }
}
