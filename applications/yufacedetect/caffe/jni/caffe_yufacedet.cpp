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
    std::string image_name = "./demo.jpg";
    std::string model_name = "./yufacedet.mnn";
    int forward = MNN_FORWARD_CPU;

    int precision = 2;
    int power     = 0;
    int memory    = 0;
    int threads   = 1;

    int INPUT_W = 320;
    int INPUT_H = 240;
    int OUTPUT_NUMS = 50;

    float score_threshold = 0.25f;
    int mean[3] = { 104,117,123 };

    cv::Mat raw_image    = cv::imread(image_name.c_str());
    int raw_image_height = raw_image.rows;
    int raw_image_width  = raw_image.cols; 
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(INPUT_W, INPUT_H));

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
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;
    
    auto session = net->createSession(config);
    net->releaseModel();
    
 

    clock_t start = clock();
    // preprocessing
    image.convertTo(image, CV_32FC3);
    int hw = INPUT_H * INPUT_W;
    for (int h = 0; h < INPUT_H; ++h)
    {
        for (int w = 0; w < INPUT_W; ++w)
        {
            for (int c = 0; c < 3; ++c)
            {
                float val = image.data[c * hw + h * INPUT_W + w];
                image.data[c * hw + h * INPUT_W + w] = (float) (val - mean[c]);
            }
        }
    }


    // wrapping input tensor, convert nhwc to nchw    
    std::vector<int> dims{1, INPUT_H, INPUT_W, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "data";
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    int type = inputTensor->getDimensionType();

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = "detection_out";

    MNN::Tensor *tensor_faces  = net->getSessionOutput(session, output_tensor_name0.c_str());

    MNN::Tensor tensor_faces_host(tensor_faces, tensor_faces->getDimensionType());
    
    tensor_faces->copyToHostTensor(&tensor_faces_host);

    // post processing steps
    auto faces_dataPtr  = tensor_faces_host.host<float>();
    std::vector<cv::Rect> faces;
    for (int i = 0; i < OUTPUT_NUMS; ++i)
    {
        float label = faces_dataPtr[i*6 + 0];
        float score = faces_dataPtr[i*6 + 1];
        float x0    = faces_dataPtr[i*6 + 2] * raw_image_width;
        float y0    = faces_dataPtr[i*6 + 3] * raw_image_height;
        float x1    = faces_dataPtr[i*6 + 4] * raw_image_width;
        float y1    = faces_dataPtr[i*6 + 5] * raw_image_height;
        if (score > score_threshold)
        {
            cv::Rect face;
            face.x = (int) (x0);
            face.y = (int) (y0);
            face.width  = (int)(x1 - x0);
            face.height = (int)(y1 - y0); 
            faces.push_back(face);
        }
    }
    clock_t end = clock();
    float duration = (float)(end - start) / CLOCKS_PER_SEC;
    printf("duration: %f\n", duration);
    // visualize result
    for (auto face: faces)
    {
        cv::rectangle(raw_image, face, cv::Scalar(0,0,255), 2);
    }    
    cv::imwrite("./output.jpg", raw_image);

    return 0;
}
