#ifndef YOLO_ENCODE_H
#define YOLO_ENCODE_H

#include "plugins/layerparams_tool.h"
#include "NvInfer.h"

namespace yolo{
    typedef struct _NormalizedBBox{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float clas;
        float score;
        bool operator <(const _NormalizedBBox &tmp) const{
            return score < tmp.score;
        }
    } NormalizedBBox;

    template <typename Dtype>
    class YoloEncode{
    public:
        YoloEncode(nvinfer1::Dims const& encode_dims
            , int net_w
            , int net_h
            , int classes
            , float objThresh
            , YoloKernel const& yolo_kernel);

        ~YoloEncode();

        int encode(Dtype *data, cudaStream_t stream);

        float *normBoxData(){
            return m_normBox;
        }

    private:
        int m_net_w{};
        int m_net_h{};

        //yolo kernel w,h,c
        int m_w{};
        int m_h{};
        int m_c{};

        int m_priorNum{};
        int m_boxNum{};
        int m_classes{};
        float m_objThresh{};

        Dtype *m_anchors{};
        float *m_normBox{};
    };
}
#endif