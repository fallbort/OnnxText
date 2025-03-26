//
//  AppDelegate.swift
//  OnnTest
//
//  Created by xfb on 2025/3/24.
//

import UIKit
import NumiOS

@main
class AppDelegate: UIResponder, UIApplicationDelegate {



    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
        
        let speaker = OnnxSpeaker()
//        if let url = Bundle.main.url(forResource: "test", withExtension: "WAV"),
//           let floats = OnnxAudioFileHelper.loadAudioFileWithResampling(url: url,targetSampleRate: 16000) {
//            let shape = NumiOS.shape(floats)
//            let sum:(Float,Float) = NumiOS.sum(floats)
//            let ret = speaker.run(wav: floats)
//            NSLog("ret=\(ret)")
//        }
        
        if let floats = OnnxSpeaker.readJSONOne(fileName: "aaa_first_400") {
            let shape = NumiOS.shape(floats)
            let sum:(Float,Float) = NumiOS.sum(floats)
            let ret = speaker.run(wav: floats)
            NSLog("ret=\(ret)")
        }
        return true
    }

    // MARK: UISceneSession Lifecycle

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session.
        // If any sessions were discarded while the application was not running, this will be called shortly after application:didFinishLaunchingWithOptions.
        // Use this method to release any resources that were specific to the discarded scenes, as they will not return.
    }


}

