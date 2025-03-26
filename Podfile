platform:ios,'14.6'
#source 'https://github.com/CocoaPods/Specs.git'
target 'OnnTest' do
use_frameworks!

pod 'onnxruntime-objc'
pod 'NumiOS'

end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['BUILD_LIBRARY_FOR_DISTRIBUTION'] = 'YES'
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '14.6'
    end
  end
end
