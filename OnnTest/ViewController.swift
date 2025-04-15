//
//  ViewController.swift
//  OnnTest
//
//  Created by xfb on 2025/3/24.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        self.view.backgroundColor = .red
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            let alert = UIAlertController(title: "test", message: "result:\(OnnxLogHelper.result),time passed=\(OnnxLogHelper.endTime - OnnxLogHelper.beginTime)", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
            self.present(alert, animated: true, completion: nil)
        }
    }


}

