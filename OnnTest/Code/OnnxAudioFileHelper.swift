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
    
    static func loadAudioFileWithResampling(url: URL, targetSampleRate: Float = 16000) -> [Float]? {
        
        do {
            // 1. Load audio file
            let audioFile = try AVAudioFile(forReading: url)
            
            // 2. Set up the audio engine for resampling
            let audioEngine = AVAudioEngine()
            let audioPlayerNode = AVAudioPlayerNode()
            audioEngine.attach(audioPlayerNode)
            
            // 3. Set the target format (desired sample rate)
            let format = AVAudioFormat(standardFormatWithSampleRate: Double(targetSampleRate), channels: audioFile.processingFormat.channelCount)
            let outputNode = audioEngine.outputNode
            audioEngine.connect(audioPlayerNode, to: outputNode, format: format)
            
            // 4. Read the audio file into an AVAudioPCMBuffer
            let frameCount = UInt32(audioFile.length)
            let audioBuffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount)
            try audioFile.read(into: audioBuffer!)
            
            // 5. Prepare the player node and start the engine
            audioPlayerNode.scheduleBuffer(audioBuffer!, at: nil, options: .loops, completionHandler: nil)
            try audioEngine.start()
            
            // 6. Extract audio data from buffer
            let channelData = audioBuffer?.floatChannelData
            let frameLength = Int(audioBuffer?.frameLength ?? 0)
            
            // 7. Convert to [Float] array
            var audioData = [Float]()
            for i in 0..<frameLength {
                audioData.append(channelData?[0][i] ?? 0)
            }
            
            return audioData
            
        } catch {
            print("Error loading or resampling audio file: \(error)")
            return nil
        }
    }
}
