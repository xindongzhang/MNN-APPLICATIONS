import onnxruntime as rt
import numpy
import cv2

session = rt.InferenceSession("./graphs/pfld-lite.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

image = cv2.imread("./image-02.jpg")
image = cv2.resize(image, (96, 96))

data = numpy.array(image, numpy.float32)
data = (data - 123.0) / 58.0

data = numpy.transpose(data, (2,0,1))
data = numpy.reshape(data, (1, 3, 96, 96))

pred_onnx = session.run([output_name], {input_name: data})[0]

for i in range(0, 98):
    x = pred_onnx[0, i*2 + 0]
    y = pred_onnx[0, i*2 + 1]
    x, y = int(x), int(y)
    cv2.circle(image, (x, y), 2, (0,0,255), -1)

cv2.imwrite("./onnx_output.jpg", image)


