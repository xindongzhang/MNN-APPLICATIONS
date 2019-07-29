### **mxnet与mnn的部署工具链整合**

```
mxnet版本：1.4.1
onnx版本：1.3.0
```

首先mxnet到mnn之间需要有一个可以直接进行解析的模型，可选的有caffe/onnx，这里我们选择的模型为onnx。大体的流程如下，

1. 训练mxnet模型
执行train文件夹中的train_mnist.py

2. 导出onnx模型
执行train文件夹中的export_onnx.py

3. 导出mnn模型
使用MNN提供的模型转换工具

```
./MNNConvert -f ONNX --modelFile mnist.onnx --MNNModel mnist.mnn --bizCode MNN
```

4. 编写jni业务代码
可参考jni文件夹目录中的代码


#### 注意
选取的样例为简单的mnist，里面没有动态op，部署流程基本没有坑，**后面我们会选择一些含动态op + 静态部署的案例**，尽量还原真实项目情况。

#### 参考
1. https://github.com/alibaba/MNN
2. https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/mnist.html
3. https://mxnet.incubator.apache.org/versions/master/tutorials/onnx/export_mxnet_to_onnx.html?highlight=onnx