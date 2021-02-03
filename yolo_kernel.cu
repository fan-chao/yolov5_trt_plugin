#include "yolo_kernel.h"

#include <iostream>

namespace yolo {

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

    static const int YOLO_CUDA_NUM_THREADS = 512;

    // CUDA: number of blocks for threads.
    static inline int YOLO_GET_BLOCKS(const int N) {
        return (N + YOLO_CUDA_NUM_THREADS - 1) / YOLO_CUDA_NUM_THREADS;
    }

    __device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

    template <typename Dtype>
    __global__ void CalDetection(const int num, const int w, const int h, const int priorn, int net_w, int net_h, const int classsn, const float objThresh
        , const Dtype *input, const float *anchors, float *norm_box) {
        CUDA_KERNEL_LOOP(index, num) {
            int pix = w * h;
            int _c = index / pix;
            int _h = (index % pix) / w;
            int _w = (index % pix) % w;

            int box_index = _c * pix + _h * w + _w;

            int elem_size = 5 + classsn;
            const Dtype* cur_box = input + box_index * elem_size;

            //get box prob
            float box_prob = Logist(cur_box[4]);
            if (box_prob < objThresh) continue;

            //get box class
            int class_id = -1;
            float score = 0.01;
            for (int i = 5; i < elem_size; ++i) {
                float p = Logist(cur_box[i]) * box_prob;
                if (p > score) {
                    score = p;
                    class_id = i - 5;
                }
            }

            if (class_id >= 0) {
                //get box location
                float box_x = (_w + Logist(cur_box[0]) * 2 - 0.5f) * net_w / w;
                float box_y = (_h + Logist(cur_box[1]) * 2 - 0.5f) * net_h / h;

                float box_w = powf(Logist(cur_box[2]) * 2, 2) * anchors[2 * _c];
                float box_h = powf(Logist(cur_box[3]) * 2, 2) * anchors[2 * _c + 1];

                norm_box[box_index * 6 + 0] = (box_x - box_w / 2) / net_w;
                norm_box[box_index * 6 + 1] = (box_y - box_h / 2) / net_h;
                norm_box[box_index * 6 + 2] = (box_x + box_w / 2) / net_w;
                norm_box[box_index * 6 + 3] = (box_y + box_h / 2) / net_h;
                norm_box[box_index * 6 + 4] = class_id;
                norm_box[box_index * 6 + 5] = score;

                //printf("box id=%d, xmin=%.02f, ymin=%.02f, xmax=%.02f, ymax=%.02f, clas=%f, score=%.06f, score0=%.06f, score1=%.06f, score2=%.06f, boxprob=%.06f\n", box_index
                //    , norm_box[box_index * 6 + 0], norm_box[box_index * 6 + 1]
                //    , norm_box[box_index * 6 + 2], norm_box[box_index * 6 + 3]
                //    , norm_box[box_index * 6 + 4], norm_box[box_index * 6 + 5]
                //    , Logist(cur_box[5]), Logist(cur_box[6]), Logist(cur_box[7])
                //    , box_prob);
            }
        }
    }

    template <typename Dtype>
    void yolo_detection(int w, int h, int priorn, int classn, float objThresh, int net_w, int net_h
        , Dtype *input, Dtype *anchors, float *norm_box, cudaStream_t ss) {
        int num = w * h * priorn;
        CalDetection << <YOLO_GET_BLOCKS(num), YOLO_CUDA_NUM_THREADS, 0, ss >> >
            (num, w, h, priorn, net_w, net_h, classn, objThresh, input, anchors, norm_box);
    }

    template void yolo_detection(int w, int h, int priorn, int classn, float objThresh, int net_w, int net_h
        , float *input, float *anchors, float *norm_box, cudaStream_t ss);
}