#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "revertMNNModel.hpp"

#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#include "config.h"
#include "anchor_generator.h"
#include "tools.h"



int main(void)
{
    std::string image_name = "./test.jpg";
    std::string model_name = "./mnet-128x128.mnn";
    int forward = MNN_FORWARD_CPU;

    int precision = 2;
    int threads   = 1;

    int INPUT_W = 128;
    int INPUT_H = 128;

    cv::Mat raw_image    = cv::imread(image_name.c_str());
    cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGB);

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
    config.backendConfig = &backendConfig;
    
    auto session = net->createSession(config);

    // preprocessing
    image.convertTo(image, CV_32FC3);

    // wrapping input tensor, convert nhwc to nchw    
    std::vector<int> dims{1, INPUT_H, INPUT_W, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "data";
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    clock_t start = clock();
    net->runSession(session);
    clock_t end = clock();
    float duration = (float)(end - start) / CLOCKS_PER_SEC;
    printf("duration: %f\n", duration);
    // get output data
    std::string tensor_box_s32_name = "face_rpn_bbox_pred_stride32";
    std::string tensor_lmk_s32_name = "face_rpn_landmark_pred_stride32";
    std::string tensor_cls_s32_name = "face_rpn_cls_prob_reshape_stride32";
    
    std::string tensor_box_s16_name = "face_rpn_bbox_pred_stride16";
    std::string tensor_lmk_s16_name = "face_rpn_landmark_pred_stride16";
    std::string tensor_cls_s16_name = "face_rpn_cls_prob_reshape_stride16";

    std::string tensor_box_s8_name  = "face_rpn_bbox_pred_stride8";
    std::string tensor_lmk_s8_name  = "face_rpn_landmark_pred_stride8";
    std::string tensor_cls_s8_name  = "face_rpn_cls_prob_reshape_stride8";



    MNN::Tensor *tensor_box_s32  = net->getSessionOutput(session, tensor_box_s32_name.c_str());
    MNN::Tensor *tensor_lmk_s32  = net->getSessionOutput(session, tensor_lmk_s32_name.c_str());
    MNN::Tensor *tensor_cls_s32  = net->getSessionOutput(session, tensor_cls_s32_name.c_str());

    MNN::Tensor *tensor_box_s16  = net->getSessionOutput(session, tensor_box_s16_name.c_str());
    MNN::Tensor *tensor_lmk_s16  = net->getSessionOutput(session, tensor_lmk_s16_name.c_str());
    MNN::Tensor *tensor_cls_s16  = net->getSessionOutput(session, tensor_cls_s16_name.c_str());

    MNN::Tensor *tensor_box_s8   = net->getSessionOutput(session, tensor_box_s8_name.c_str());
    MNN::Tensor *tensor_lmk_s8   = net->getSessionOutput(session, tensor_lmk_s8_name.c_str());
    MNN::Tensor *tensor_cls_s8   = net->getSessionOutput(session, tensor_cls_s8_name.c_str());

    MNN::Tensor tensor_box_s32_host(tensor_box_s32, tensor_box_s32->getDimensionType());
    MNN::Tensor tensor_lmk_s32_host(tensor_lmk_s32, tensor_lmk_s32->getDimensionType());
    MNN::Tensor tensor_cls_s32_host(tensor_cls_s32, tensor_cls_s32->getDimensionType());
    
    MNN::Tensor tensor_box_s16_host(tensor_box_s16, tensor_box_s16->getDimensionType());
    MNN::Tensor tensor_lmk_s16_host(tensor_lmk_s16, tensor_lmk_s16->getDimensionType());
    MNN::Tensor tensor_cls_s16_host(tensor_cls_s16, tensor_cls_s16->getDimensionType());

    MNN::Tensor tensor_box_s8_host (tensor_box_s8 , tensor_box_s8->getDimensionType());
    MNN::Tensor tensor_lmk_s8_host (tensor_lmk_s8 , tensor_lmk_s8->getDimensionType());
    MNN::Tensor tensor_cls_s8_host (tensor_cls_s8 , tensor_cls_s8->getDimensionType());
    
    tensor_box_s32->copyToHostTensor(&tensor_box_s32_host);
    tensor_lmk_s32->copyToHostTensor(&tensor_lmk_s32_host);
    tensor_cls_s32->copyToHostTensor(&tensor_cls_s32_host);

    tensor_box_s16->copyToHostTensor(&tensor_box_s16_host);
    tensor_lmk_s16->copyToHostTensor(&tensor_lmk_s16_host);
    tensor_cls_s16->copyToHostTensor(&tensor_cls_s16_host);

    tensor_box_s8->copyToHostTensor(&tensor_box_s8_host);
    tensor_lmk_s8->copyToHostTensor(&tensor_lmk_s8_host);
    tensor_cls_s8->copyToHostTensor(&tensor_cls_s8_host);

    // post processing steps
    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;

    ac[0].FilterAnchor(tensor_cls_s32_host, tensor_box_s32_host, tensor_lmk_s32_host, proposals);
    ac[1].FilterAnchor(tensor_cls_s16_host, tensor_box_s16_host, tensor_lmk_s16_host, proposals);
    ac[2].FilterAnchor(tensor_cls_s8_host, tensor_box_s8_host, tensor_lmk_s8_host, proposals);
    
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);
    printf("final result %d\n", result.size());

    cv::cvtColor(raw_image, raw_image, cv::COLOR_RGB2BGR);
    for(int i = 0; i < result.size(); i ++)
    {
        float h_scale = (float) raw_image_height / INPUT_H;
        float w_scale = (float) raw_image_width  / INPUT_W;
        cv::rectangle (
            raw_image, 
            cv::Point((int)result[i].finalbox.x * w_scale, (int)result[i].finalbox.y * h_scale), 
            cv::Point((int)result[i].finalbox.width * w_scale, (int)result[i].finalbox.height * h_scale), 
            cv::Scalar(0, 255, 255), 2, 8, 0
        );
        for (int j = 0; j < result[i].pts.size(); ++j) {
        	cv::circle(
                raw_image, 
                cv::Point((int)result[i].pts[j].x * w_scale, (int)result[i].pts[j].y * h_scale), 
                1, cv::Scalar(225, 0, 225), 2, 8
            );
        }
    }
 
    // visualize result
    cv::imwrite("./output.jpg", raw_image);

    return 0;
}
