LOCAL_PATH := $(call my-dir)

OpenCV_BASE = ../../../../libraries/opencv-4.1
MNN_BASE    = ../../../../libraries/mnn

include $(CLEAR_VARS)
LOCAL_MODULE := MNN
LOCAL_SRC_FILES := $(MNN_BASE)/libs/libMNN.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := MNN_CL
LOCAL_SRC_FILES := $(MNN_BASE)/libs/libMNN_CL.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := MNN_Vulkan
LOCAL_SRC_FILES := $(MNN_BASE)/libs/libMNN_Vulkan.so
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS)

OpenCV_INSTALL_MODULES := on
OPENCV_LIB_TYPE := STATIC
include $(OpenCV_BASE)/sdk/native/jni/OpenCV.mk
LOCAL_MODULE := onnx_mnist

LOCAL_C_INCLUDES += $(OPENCV_INCLUDE_DIR)
LOCAL_C_INCLUDES += $(MNN_BASE)/include/include
LOCAL_C_INCLUDES += $(MNN_BASE)/include/tools
LOCAL_C_INCLUDES += $(MNN_BASE)/include/tools/cpp
LOCAL_C_INCLUDES += $(MNN_BASE)/include/source
LOCAL_C_INCLUDES += $(MNN_BASE)/include/source/backend
LOCAL_C_INCLUDES += $(MNN_BASE)/include/source/core
LOCAL_C_INCLUDES += $(MNN_BASE)/include/source/cv
LOCAL_C_INCLUDES += $(MNN_BASE)/include/source/math
LOCAL_C_INCLUDES += $(MNN_BASE)/include/source/shape

LOCAL_SRC_FILES := \
                onnx_mnist.cpp \
				$(MNN_BASE)/include/tools/cpp/revertMNNModel.cpp


LOCAL_LDLIBS := -landroid -llog -ldl -lz 
LOCAL_CFLAGS   := -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -ftree-vectorize -fPIC -Ofast -ffast-math -w -std=c++14
LOCAL_CPPFLAGS := -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fPIC -Ofast -ffast-math -std=c++14
LOCAL_LDFLAGS  += -Wl,--gc-sections
LOCAL_CFLAGS   += -fopenmp
LOCAL_CPPFLAGS += -fopenmp
LOCAL_LDFLAGS  += -fopenmp
LOCAL_ARM_NEON := true

APP_ALLOW_MISSING_DEPS = true

LOCAL_SHARED_LIBRARIES :=                             \
                        MNN                           \
					    MNN_CL                        \
						MNN_Vulkan                    

include $(BUILD_EXECUTABLE)
