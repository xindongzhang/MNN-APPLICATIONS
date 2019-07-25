#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "revertMNNModel.hpp"

#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

float iou(cv::Rect box0, cv::Rect box1) 
{
    float xmin0 = box0.x;
    float ymin0 = box0.y;
    float xmax0 = box0.x + box0.width;
    float ymax0 = box0.y + box0.height;
    
    float xmin1 = box1.x;
    float ymin1 = box1.y;
    float xmax1 = box1.x + box1.width;
    float ymax1 = box1.y + box1.height;

    float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
    
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    
    if (u <= 0.0) return 0.0f;
    else          return i/u;
}



int main(void)
{
    std::string image_name = "./nopluz_0.jpg";
    std::string model_name = "./bsd_224x224.mnn";
    int forward = MNN_FORWARD_CPU;
    // int forward = MNN_FORWARD_OPENCL;

    int precision = 2;
    int power     = 0;
    int memory    = 0;
    int threads   = 1;

    // int INPUT_SIZE = 300;
    // int OUTPUT_NUM = 1917; // for 300x300

    int INPUT_SIZE = 224;
    int OUTPUT_NUM = 1014; // for 224x224
    float X_SCALE    = 10.0;
    float Y_SCALE    = 10.0;   
    float H_SCALE    = 5.0;  
    float W_SCALE    = 5.0;
    float score_threshold = 0.5f;
    float nms_threshold   = 0.45f;

    cv::Mat raw_image    = cv::imread(image_name.c_str());
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
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;
    
    auto session = net->createSession(config);
    net->releaseModel();
    
 

    clock_t start = clock();
    // preprocessing
    image.convertTo(image, CV_32FC3);
    image = (image * 2 / 255.0f) - 1;

    // wrapping input tensor, convert nhwc to nchw    
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "normalized_input_image_tensor";
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = "convert_scores";
    std::string output_tensor_name1 = "Squeeze";
    std::string output_tensor_name2 = "anchors";

    MNN::Tensor *tensor_scores  = net->getSessionOutput(session, output_tensor_name0.c_str());
    MNN::Tensor *tensor_boxes   = net->getSessionOutput(session, output_tensor_name1.c_str());
    MNN::Tensor *tensor_anchors = net->getSessionOutput(session, output_tensor_name2.c_str());

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    MNN::Tensor tensor_anchors_host(tensor_anchors, tensor_anchors->getDimensionType());
    
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    tensor_anchors->copyToHostTensor(&tensor_anchors_host);

    // post processing steps
    auto scores_dataPtr  = tensor_scores_host.host<float>();
    auto boxes_dataPtr   = tensor_boxes_host.host<float>();
    auto anchors_dataPtr = tensor_anchors_host.host<float>();


    int batch = tensor_scores->batch();
    int channel = tensor_scores->channel();
    int height  = tensor_scores->height();
    int width   = tensor_scores->width();
    printf("%d, %d, %d, %d\n", batch, height, width, channel);
    

    // location and score decoding
    std::vector<cv::Rect> tmp_faces;
    for(int i = 0; i < OUTPUT_NUM; ++i)
    {
        // location decoding
        float ycenter =     boxes_dataPtr[i*4 + 0] / Y_SCALE  * anchors_dataPtr[i*4 + 2] + anchors_dataPtr[i*4 + 0];
        float xcenter =     boxes_dataPtr[i*4 + 1] / X_SCALE  * anchors_dataPtr[i*4 + 3] + anchors_dataPtr[i*4 + 1];
        float h       = exp(boxes_dataPtr[i*4 + 2] / H_SCALE) * anchors_dataPtr[i*4 + 2];
        float w       = exp(boxes_dataPtr[i*4 + 3] / W_SCALE) * anchors_dataPtr[i*4 + 3];

        float ymin    = ( ycenter - h * 0.5 ) * raw_image_height;
        float xmin    = ( xcenter - w * 0.5 ) * raw_image_width;
        float ymax    = ( ycenter + h * 0.5 ) * raw_image_height;
        float xmax    = ( xcenter + w * 0.5 ) * raw_image_width;

        // probability decoding, softmax
        float nonface_prob = exp(scores_dataPtr[i*2 + 0]);
        float face_prob    = exp(scores_dataPtr[i*2 + 1]);

        float ss           = nonface_prob + face_prob;
        nonface_prob       /= ss;
        face_prob          /= ss;

        if (face_prob > score_threshold) {
            cv::Rect tmp_face;
            tmp_face.x = xmin;
            tmp_face.y = ymin;
            tmp_face.width  = xmax - xmin;
            tmp_face.height = ymax - ymin;
            tmp_faces.push_back(tmp_face); 
        }
    }
    
    // perform NMS
    int N = tmp_faces.size();
    std::vector<int> labels(N, -1); 
    for(int i = 0; i < N-1; ++i)
    {
        for (int j = i+1; j < N; ++j)
        {
            cv::Rect pre_box = tmp_faces[i];
            cv::Rect cur_box = tmp_faces[j];
            float iou_ = iou(pre_box, cur_box);
            if (iou_ > nms_threshold) {
                labels[j] = 0;
            }
        }
    }

    std::vector<cv::Rect> faces;
    for (int i = 0; i < N; ++i)
    {
        if (labels[i] == -1)
            faces.push_back(tmp_faces[i]);
    }
    clock_t end = clock();
    float duration = float(end - start)/CLOCKS_PER_SEC;
    printf("duration: %f \n", duration);

    // visualize
    for (cv::Rect face: faces) 
    {
        cv::Rect vis_box;
        vis_box.x = (int) face.x;
        vis_box.y = (int) face.y;
        vis_box.width  = (int) face.width;
        vis_box.height = (int) face.height;
        cv::rectangle(raw_image, vis_box, cv::Scalar(0,0,255), 2);
    }

    cv::imwrite("./output.jpg", raw_image);

    return 0;
}
