# yolov5_trt_plugin
 This project used to create tensorrt yolov5 plugin which is adapted to ultralytics/yolov5 and references to https://github.com/wang-xinyu/tensorrtx.
 
 The format of result is [box_num, xmin, xmax, class, score, ...].
 
 You can use function Yolov5LayerPlugin::add_yolov5_layer() to add this plugin to your network.
 net_w and net_h equal input dimension w and h respectively.
 det0, det1 and det2 are outputs of onnx model exported from ultralytics/yolov5 model without detect mode.
 
 Some other functions, such as ByteStreamReader which is used to do serialization and deserialization.
