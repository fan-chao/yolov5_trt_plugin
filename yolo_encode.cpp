/*
* 集成yolo kernel，从网络中获取bbox结果
* 计算bbox位置和分类信息
*/

#include "yolo_encode.h"
#include "yolo_kernel.h"

#include "cuda.h"
#include "cuda_utils_sdk.h"

namespace yolo{

    template <typename Dtype>
    YoloEncode<Dtype>::YoloEncode(nvinfer1::Dims const& encode_dims
        , int net_w
        , int net_h
        , int classes
        , float objThresh
        , YoloKernel const& yolo_kernel) {

        m_net_w = net_w;
        m_net_h = net_h;

        m_classes = classes;
        m_objThresh = objThresh;

        m_priorNum = encode_dims.d[0];
        m_h = encode_dims.d[1];
        m_w = encode_dims.d[2];

        m_boxNum = m_w * m_h * m_priorNum;

        int anchor_size = m_priorNum * 2 * sizeof(float);
        cuAllocMapped((void**)&m_anchors, anchor_size);
        memcpy(m_anchors, yolo_kernel.anchors.data(), anchor_size);

        cuAllocMapped((void**)&m_normBox, m_boxNum * 6 * sizeof(float));
    }

    template <typename Dtype>
    YoloEncode<Dtype>::~YoloEncode() {
        if (nullptr != m_anchors){
            cuFreeMapped(m_anchors);
            m_anchors = nullptr;
        }

        if (nullptr != m_normBox){
            cuFreeMapped(m_normBox);
            m_normBox = nullptr;
        }
    }

    template <typename Dtype>
    int YoloEncode<Dtype>::encode(Dtype *data, cudaStream_t stream) {
        memset(m_normBox, 0, m_boxNum * 6 * sizeof(float));
        yolo_detection(m_w, m_h, m_priorNum, m_classes, m_objThresh, m_net_w, m_net_h, data, m_anchors, m_normBox, stream);
        return m_boxNum;
    }

    template class YoloEncode<float>;
}