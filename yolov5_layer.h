#ifndef _YOLOV5_LAYER_PLUGIN
#define _YOLOV5_LAYER_PLUGIN

#include "utils/yolo_encode.h"
#include "layerparams_tool.h"
#include "NvInfer.h"

namespace nvinfer1
{
    class Yolov5LayerPlugin : public IPluginV2IOExt
    {
    public:
        explicit Yolov5LayerPlugin(yolo::Yolov5Param yolov5_param);
        // create the plugin at runtime from a byte stream
        Yolov5LayerPlugin(const void* data, size_t length);
        Yolov5LayerPlugin() = delete;
        virtual ~Yolov5LayerPlugin();

        virtual int getNbOutputs() const override;

        virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        virtual int initialize() override;

        virtual void terminate() override;

        virtual size_t getWorkspaceSize(int maxBatchSize) const override;

        virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() const override;

        virtual void serialize(void* buffer) const override;

        // 这个函数可以获取到数据类型和输入输出的维度信息，如果有需要用到的可以在这里将相关信息取出来
        virtual void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

        // pos索引到的input/output的数据格式（format）和数据类型（datatype）如果都支持则返回true
        virtual bool supportsFormatCombination(int pos, const PluginTensorDesc* in_out, int num_inputs, int num_outputs) const override;

        // 返回输出的数据类型，如何输入相同，可以直接 return input_types[0]；
        virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* input_types, int num_inputs) const override;

        // 返回自定义类型，如这里是：return Upsample
        virtual const char* getPluginType() const override;

        // 返回plugin version，没啥说的
        virtual const char* getPluginVersion() const override;

        // 销毁对象
        virtual void destroy() override;

        // 在这里new一个该自定义类型并返回
        virtual nvinfer1::IPluginV2Ext* clone() const override;

        // 设置命名空间，用来在网络中查找和创建plugin
        virtual void setPluginNamespace(const char* lib_namespace) override;

        // 返回plugin对象的命名空间
        virtual const char* getPluginNamespace() const override;

        virtual bool isOutputBroadcastAcrossBatch(int output_index, const bool* input_is_broadcasted, int num_inputs) const override;

        virtual bool canBroadcastInputAcrossBatch(int input_index) const override;

        static int add_yolov5_layer(INetworkDefinition *network, int net_w, int net_h, int max_batch_size
            , ITensor* det0, ITensor* det1, ITensor* det2);

    private:
        void forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize = 1);

        void getBox(yolo::YoloEncode<float> *boxEncodor, int num
            , std::vector<std::vector<yolo::NormalizedBBox>> &result_array
            , std::vector<int> &size_array
            , std::vector<int> &count_arrray
            , const std::vector<float> &class_thresh);

        void nms(std::vector<yolo::NormalizedBBox> &input_boxes, float nms_thresh);

    private:
        std::vector<Dims> mInputDims;
        Dims mOutputDim;

        DataType mDataType{ DataType::kFLOAT };

        yolo::Yolov5Param mYolov5Param;
        const char* mPluginNamespace;

        std::vector<yolo::YoloEncode<float> *> mBox_encodes_float;

        float* topdata{};
    };

    class Yolov5PluginCreator : public nvinfer1::IPluginCreator {
    public:
        Yolov5PluginCreator();
        ~Yolov5PluginCreator() override = default;
        const char* getPluginName() const override;
        const char* getPluginVersion() const override;
        const PluginFieldCollection* getFieldNames() override;
        // 创建自定义层pluin的对象并返回
        nvinfer1::IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;
        // 创建自定义层pluin的对象并返回，反序列化用到
        nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serial_data, size_t serial_length) override;
        void setPluginNamespace(const char* lib_namespace) override;
        const char* getPluginNamespace() const override;

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
};

#endif