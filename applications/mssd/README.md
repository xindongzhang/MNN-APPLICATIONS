# MNN_MSSD
# 1. Training MSSD model 
trainning your own MSSD model by Google Object Detection API [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Here we suppose that you have already finished the training phase.
# 2. Exporting tflite model
For more details of exporting tflite model, please refer to [export tflite](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). And we List two important steps as follows:

``` 
object_detection/export_tflite_ssd_graph.py  \
--pipeline_config_path=$CONFIG_FILE          \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR               \
--add_postprocessing_op=false
```

After this step, you got a pb file. **If you would like to use tensorflow pb model for this MNN project, you can straightly jump the /tensorflow/jni, there are codes and details for it.**

If you would like to use tflite for this MNN project, please be patient. You still need to go throught the rest of this document.

``` 
bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb            \
--output_file=$OUTPUT_DIR/detect.tflite             \
--input_shapes=1,224,224,3                          \
--input_arrays=normalized_input_image_tensor        \
--output_arrays='concat','concat_1'                 \
--inference_type=FLOAT                              \
--change_concat_input_ranges=false                  
```

# 3. Compile MNN for android 
MNN is a lightweight deep neural network inference engine [MNN](https://github.com/alibaba/MNN). You can follow this [link](https://github.com/alibaba/MNN/blob/master/doc/Benchmark_EN.md) for compilation and benckmark.

# 4. Convert tflite model to mnn model 
For details, please follow [link](https://github.com/alibaba/MNN/blob/master/tools/converter/README.md). By only simple command line as follow you can get MNN model for MSSD, if you do it right. The following command is only for example, you need to replace items in --modelFile and --MNNModel with yours respectively.

``` 
./MNNConvert -f TFLITE --modelFile ~/Desktop/DMS-data/facedet/trained_model/detect.tflite --MNNModel ~/Desktop/face_det.mnn --bizCode MNN
```

And for tensorflow pb files, type

``` 
./MNNConvert -f TF --modelFile ~/Desktop/DMS-data/facedet/trained_model/detect.tflite --MNNModel ~/Desktop/face_det.mnn --bizCode MNN
```


# 5. Compile and execute demo
This demo depends on [OpenCV on Android](https://sourceforge.net/projects/opencvlibrary/files/4.1.0/opencv-4.1.0-android-sdk.zip/download) and [MNN](https://github.com/alibaba/MNN), please download them respectively. First, you need to revise **OpenCV_BASE** and **MNN_BASE** in jni/Android.mk acording to your desktop environment. And remember to install NDK, I use **android-ndk-r17c** for compilation in my desktop.

Then, by following simple command, you should get libs and executable file

if you have done things right.

``` 
cd jni
ndk-build
```

In order to test it, we push them to the android devices. For example, here we suppose you are in jni folder.

``` 
adb push ../libs/arm64-v8a/* /data/local/tmp
adb push image.jpg /data/local/tmp
adb push face_det.mnn /data/local/tmp
```

Then we need to set up the enviroment for testing in android devices, like
``` 
adb shell   
cd /data/local/tmp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp
```

then, in /data/local/tmp, run the executable file **mssd**

``` 
./mssd
```

The last step, close the adb shell session, and pull the output result back to your desktop.
``` 
adb pull /data/local/tmp/output.jpg ./
```

# 6. Discussion
For more discussion, you can contact me via Wechat: **zxd675816777**
