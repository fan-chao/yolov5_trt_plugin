#ifndef YOLO_KERNEL_H
#define YOLO_KERNEL_H

namespace yolo {
    template <typename Dtype>
    void yolo_detection(int w, int h, int priorn, int classn, float objThresh, int net_w, int net_h
        , Dtype *input, Dtype *anchors, float *norm_box, cudaStream_t ss);
}

#endif