#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "revertMNNModel.hpp"

#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

int main(void)
{
    std::string image_name = "./image-02.jpg";
    std::string model_name = "./pfld-lite.mnn";
    int forward = MNN_FORWARD_CPU;

    int threads    = 1;
    int INPUT_SIZE = 96;

    cv::Mat raw_image    = cv::imread(image_name.c_str());
    cv::resize(raw_image, raw_image, cv::Size(INPUT_SIZE, INPUT_SIZE)); 
    int raw_image_height = raw_image.rows;
    int raw_image_width  = raw_image.cols; 
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));

    // load and config mnn model
    auto revertor = std::unique_ptr<Revert>(new Revert(model_name.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    MNN::ScheduleConfig config;
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    config.backendConfig = &backendConfig;
    
    auto session = net->createSession(config);
    net->releaseModel();
    
 

    clock_t start = clock();
    // preprocessing
    image.convertTo(image, CV_32FC3);
    image = (image - 123.0) / 58.0;

    // wrapping input tensor, convert nhwc to nchw    
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "data";
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = "conv5_fwd";

    MNN::Tensor *tensor_lmks  = net->getSessionOutput(session, output_tensor_name0.c_str());

    MNN::Tensor tensor_lmks_host(tensor_lmks, tensor_lmks->getDimensionType());
    
    tensor_lmks->copyToHostTensor(&tensor_lmks_host);

    int batch = tensor_lmks->batch();
    int channel = tensor_lmks->channel();
    int height = tensor_lmks->height();
    int width  = tensor_lmks->width();
    int type   = tensor_lmks->getDimensionType();
    printf("%d, %d, %d, %d, %d\n", batch, channel, height, width, type);

    // post processing steps
    auto lmks_dataPtr  = tensor_lmks_host.host<float>();

    int num_of_pts = 98;
    for (int i = 0; i < num_of_pts; ++i)
    {
        int x = (int) (lmks_dataPtr[i*2 + 0]);
        int y = (int) (lmks_dataPtr[i*2 + 1]);
        cv::circle(raw_image, cv::Point(x, y), 2, cv::Scalar(0,0,255), -1);
    }

    cv::imwrite("./output.jpg", raw_image);
    return 0;
}
