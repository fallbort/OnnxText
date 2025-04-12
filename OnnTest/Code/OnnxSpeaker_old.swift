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
