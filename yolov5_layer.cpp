#include "yolov5_layer.h"
#include "utils/yolo_encode.h"
#include "utils/Utils.h"
#include "PluginFactory.h"
#include "cuda_utils_sdk.h"

#include <glog/logging.h>

using namespace Tn; // Utils.h

const int OBJECT_TOPK = 200;
const std::vector<float> s_class_thresh{ 0.5, 0.5, 0.5 };
const std::vector<float> s_nms_thresh{ 0.5, 0.5, 0.5 };

namespace nvinfer1 {
    namespace {
        const char* YOLOV5_LAYER_PLUGIN_VERSION = "1";
        const char* YOLOV5_LAYER_PLUGIN_TYPE = "Yolov5LayerPlugin";
        const char* YOLOV5_LAYER_PLUGIN_NAMESPACE = "_TRT";
        const char* YOLOV5_LAYER_PLUGIN_NAME = "Yolov5LayerPlugin_TRT";
    }

    Yolov5LayerPlugin::Yolov5LayerPlugin(yolo::Yolov5Param yolov5_param) {
        mYolov5Param = std::move(yolov5_param);
    }

    // create the plugin at runtime from a byte stream
    Yolov5LayerPlugin::Yolov5LayerPlugin(const void* data, size_t length) {
        ByteStreamReader br(data, length);
        br >> mInputDims >> mOutputDim >> mDataType >> mYolov5Param;
        assert(br.left_bytes() == 0);
    }

    Yolov5LayerPlugin::~Yolov5LayerPlugin() {}

    int Yolov5LayerPlugin::getNbOutputs() const {
        return 1;
    }

    Dims Yolov5LayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims){
        assert(nbInputDims == mYolov5Param.yolo_kernels.size());

        mInputDims.clear();
        for (int i = 0; i < nbInputDims; i++){
            mInputDims.push_back(inputs[i]);
        }

        mOutputDim = Dims3(OBJECT_TOPK, 6, 1);
        return mOutputDim;
    }

    int Yolov5LayerPlugin::initialize() {
        if (mInputDims.empty()) return 0;

        int kernel_num = mYolov5Param.yolo_kernels.size();
        mBox_encodes_float.resize(kernel_num);
        for (int i = 0; i < kernel_num; i++){
            if (mDataType == DataType::kFLOAT){
                yolo::YoloEncode<float> *encoder = new yolo::YoloEncode<float>(mInputDims[i]
                    , mYolov5Param.net_w
                    , mYolov5Param.net_h
                    , mYolov5Param.classes
                    , mYolov5Param.obj_thresh
                    , mYolov5Param.yolo_kernels[i]);

                mBox_encodes_float[i] = encoder;
            }
        }

        if (cuAllocMapped((void**)&topdata, OBJECT_TOPK * 6 * sizeof(float)) != 0) {
            return -1;
        }

        return 0;
    }

    void Yolov5LayerPlugin::terminate(){
        if (topdata){
            cuFreeMapped(topdata);
            topdata = nullptr;
        }

        for (int i = 0; i < mBox_encodes_float.size(); ++i) {
            if (mBox_encodes_float[i] != nullptr) {
                delete mBox_encodes_float[i];
                mBox_encodes_float[i] = nullptr;
            }
        }
        mBox_encodes_float.clear();
    }

    size_t Yolov5LayerPlugin::getWorkspaceSize(int maxBatchSize) const {
        return 0;
    }

    int Yolov5LayerPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream){
        for (int batch = 0; batch < batchSize; ++batch) {
            int kernel_num = mYolov5Param.yolo_kernels.size();
            vector<int> box_num_array;
            box_num_array.resize(kernel_num);
            for (int i = 0; i < kernel_num; i++){
                yolo::YoloEncode<float> *encoder = mBox_encodes_float[i];
                int num = encoder->encode((float *)inputs[i] + batch * mInputDims[i].d[0] * mInputDims[i].d[1] * mInputDims[i].d[2] * mInputDims[i].d[3], stream);
                box_num_array[i] = num;
            }

            int classes = mYolov5Param.classes;
            vector<int> size_array(classes, 0);
            vector<int> count_array(classes, 500);
            vector<vector<yolo::NormalizedBBox>> result_array;
            result_array.resize(classes);
            for (int i = 0; i < classes; i++)
                result_array[i].resize(500);

            cudaStreamSynchronize(stream);

            for (int i = 0; i < kernel_num; i++){
                getBox(mBox_encodes_float[i], box_num_array[i], result_array, size_array, count_array, s_class_thresh);
            }

            int box_num = 0;
            for (int i = 0; i < classes; i++){
                result_array[i].resize(size_array[i]);
                std::sort(result_array[i].rbegin(), result_array[i].rend());
                nms(result_array[i], s_nms_thresh[i]);
                int size = result_array[i].size();
                box_num += size;
            }

            if (cuArrayFillValue(topdata, -1, OBJECT_TOPK * 6, stream) != 0) {
                for (int i = 0; i < OBJECT_TOPK * 6; ++i)
                    topdata[i] = -1;
            }

            cudaStreamSynchronize(stream);

            int k = 1;
            for (int i = 0; i < classes; i++){
                int n = result_array[i].size();
                bool finished{};
                for (int j = 0; j < n; j++){
                    topdata[k * 6 + 0] = result_array[i][j].xmin;
                    topdata[k * 6 + 1] = result_array[i][j].ymin;
                    topdata[k * 6 + 2] = result_array[i][j].xmax;
                    topdata[k * 6 + 3] = result_array[i][j].ymax;
                    topdata[k * 6 + 4] = result_array[i][j].clas;
                    topdata[k * 6 + 5] = result_array[i][j].score;
                    k++;

                    if (k > OBJECT_TOPK - 1) {
                        finished = true;
                        break;
                    }
                }
                if (finished) break;
            }

            topdata[0] = k - 1;

            int data_size = OBJECT_TOPK * 6 * sizeof(float);
            cudaMemcpyAsync((void*)outputs[0] + batch * data_size, topdata, data_size, cudaMemcpyDeviceToDevice, stream);
        }
        return 0;
    }

    size_t Yolov5LayerPlugin::getSerializationSize() const {
        ByteStreamSizeWriter bw;
        bw << mInputDims << mOutputDim << mDataType << mYolov5Param;
        return bw.offset();
    }

    void Yolov5LayerPlugin::serialize(void* buffer) const {
        ByteStreamWriter1 bw((char*)buffer);
        bw << mInputDims << mOutputDim << mDataType << mYolov5Param;
    }

    void Yolov5LayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) {
        assert(in && nbInput == mYolov5Param.yolo_kernels.size());
        mInputDims.resize(nbInput);
        for (int i = 0; i < nbInput; i++){
            mInputDims[i] = in[i].dims;
        }

        assert(out && nbOutput == 1);
        mOutputDim = out[0].dims;

        assert(in[0].type == out[0].type && in[0].type == DataType::kFLOAT);
        assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);

        mDataType = in[0].type;
    }

    //! The combination of kLINEAR + kFLOAT is supported.
    bool Yolov5LayerPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const {
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    }

    DataType Yolov5LayerPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const {
        assert(index == 0);
        assert(nbInputs == mYolov5Param.yolo_kernels.size());
        return DataType::kFLOAT;
    }

    const char* Yolov5LayerPlugin::getPluginType() const {
        return YOLOV5_LAYER_PLUGIN_TYPE;
    }

    const char* Yolov5LayerPlugin::getPluginVersion() const {
        return YOLOV5_LAYER_PLUGIN_VERSION;
    }

    void Yolov5LayerPlugin::destroy() {
        terminate();
        delete this;
    }

    IPluginV2Ext* Yolov5LayerPlugin::clone() const {
        auto* plugin = new Yolov5LayerPlugin(mYolov5Param);
        plugin->mInputDims = mInputDims;
        plugin->mOutputDim = mOutputDim;
        plugin->mDataType = mDataType;
        plugin->setPluginNamespace(mPluginNamespace);
        plugin->initialize();

        return plugin;
    }

    void Yolov5LayerPlugin::setPluginNamespace(const char* libNamespace) {
        mPluginNamespace = libNamespace;
    }

    const char* Yolov5LayerPlugin::getPluginNamespace() const {
        return mPluginNamespace;
    }

    bool Yolov5LayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {
        return false;
    }

    bool Yolov5LayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const {
        return false;
    }

    void Yolov5LayerPlugin::getBox(yolo::YoloEncode<float> *boxEncodor, int num
        , vector<vector<yolo::NormalizedBBox>> &result_array
        , vector<int> &size_array
        , vector<int> &count_arrray
        , const vector<float> &class_thresh){
        yolo::NormalizedBBox  *prob = (yolo::NormalizedBBox *)(boxEncodor->normBoxData());
        int prob_size = num;
        int classes = class_thresh.size();
        for (int i = 0; i<prob_size; i++){
            if (prob[i].score < 0.01 || static_cast<int>(prob[i].clas) == -1)
                continue;
            for (int j = 0; j<classes; j++){
                if (j == static_cast<int>(prob[i].clas) && prob[i].score >= class_thresh[j]){
                    result_array[j][size_array[j]] = prob[i];
                    size_array[j] = size_array[j] + 1;
                    if (size_array[j] == count_arrray[j]){
                        count_arrray[j] = count_arrray[j] * 2;
                        result_array[j].resize(count_arrray[j]);
                    }
                    break;
                }
            }
        }
    }

    void Yolov5LayerPlugin::nms(vector<yolo::NormalizedBBox> &input_boxes, float nms_thresh) {
        std::vector<float>vArea(input_boxes.size());
        for (int i = 0; i < input_boxes.size(); ++i){
            float w = input_boxes.at(i).xmax - input_boxes.at(i).xmin;
            float h = input_boxes.at(i).ymax - input_boxes.at(i).ymin;
            vArea[i] = (w*h);
        }
        for (int i = 0; i < input_boxes.size(); ++i){
            for (int j = i + 1; j < input_boxes.size();){
                float xx1 = std::max(input_boxes[i].xmin, input_boxes[j].xmin);
                float yy1 = std::max(input_boxes[i].ymin, input_boxes[j].ymin);
                float xx2 = std::min(input_boxes[i].xmax, input_boxes[j].xmax);
                float yy2 = std::min(input_boxes[i].ymax, input_boxes[j].ymax);
                float w = std::max(float(0), xx2 - xx1);
                float h = std::max(float(0), yy2 - yy1);
                float inter = w * h;
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr > nms_thresh){
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else
                    j++;
            }
        }
    }

    int Yolov5LayerPlugin::add_yolov5_layer(INetworkDefinition* network, int net_w, int net_h, int max_batch_size
        , ITensor* det0, ITensor* det1, ITensor* det2) {
        yolo::Yolov5Param param;
        param.net_w = net_w;
        param.net_h = net_h;
        param.classes = 3;
        param.obj_thresh = 0.5;
        param.max_batch_size = max_batch_size;
        param.nms_thresh = { 0.3, 0.3, 0.3 };
        param.class_thresh = { 0.5, 0.5, 0.5 };

        yolo::YoloKernel yk1;
        yk1.anchors = { 10, 13, 16, 30, 33, 23 };
        yolo::YoloKernel yk2;
        yk2.anchors = { 30, 61, 62, 45, 59, 119 };
        yolo::YoloKernel yk3;
        yk3.anchors = { 116, 90, 156, 198, 373, 326 };
        param.yolo_kernels = { yk1, yk2, yk3 };

        ByteStreamWriter0 bw(64);
        bw << param;

        auto creator = getPluginRegistry()->getPluginCreator(YOLOV5_LAYER_PLUGIN_TYPE
            , YOLOV5_LAYER_PLUGIN_VERSION
            , YOLOV5_LAYER_PLUGIN_NAMESPACE);
        if (creator == nullptr) return -1;

        PluginField plugin_filed[1];
        plugin_filed[0].data = bw.ptr();
        plugin_filed[0].length = bw.offset();
        plugin_filed[0].name = "yolov5data";

        PluginFieldCollection plugin_data;
        plugin_data.nbFields = 1;
        plugin_data.fields = plugin_filed;
        IPluginV2 *pluginObj = creator->createPlugin("Yolov5LayerPlugin", &plugin_data);
        ITensor* inputTensors_yolo[] = { det0, det1, det2 };
        auto yolov5 = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
        yolov5->getOutput(0)->setName("output");
        network->markOutput(*yolov5->getOutput(0));
        network->unmarkOutput(*det0);
        network->unmarkOutput(*det1);
        network->unmarkOutput(*det2);
        return 0;
    }

    /*
    ** plugin creator
    */
    PluginFieldCollection Yolov5PluginCreator::mFC{};
    std::vector<PluginField> Yolov5PluginCreator::mPluginAttributes;


    Yolov5PluginCreator::Yolov5PluginCreator() {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();

        mNamespace = YOLOV5_LAYER_PLUGIN_NAMESPACE;
    }

    const char* Yolov5PluginCreator::getPluginName() const {
        return YOLOV5_LAYER_PLUGIN_NAME;
    }

    const char* Yolov5PluginCreator::getPluginVersion() const {
        return YOLOV5_LAYER_PLUGIN_VERSION;
    }

    const PluginFieldCollection* Yolov5PluginCreator::getFieldNames() {
        return &mFC;
    }

    IPluginV2* Yolov5PluginCreator::createPlugin(const char *layerName, const PluginFieldCollection* fc) {
        yolo::Yolov5Param yolov5_param;

        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; i++) {
            if (strcmp(fields[i].name, "yolov5data") == 0) {
                ByteStreamReader br(fields[i].data, fields[i].length);
                br >> yolov5_param;
            }
        }
        Yolov5LayerPlugin* obj = new Yolov5LayerPlugin(std::move(yolov5_param));
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    // deserialization plugin implementation
    IPluginV2* Yolov5PluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength) {
        // This object will be deleted when the network is destroyed, which will
        // call Yolov5LayerPlugin::destroy()
        Yolov5LayerPlugin* obj = new Yolov5LayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    void Yolov5PluginCreator::setPluginNamespace(const char* pluginNamespace) {
        mNamespace = pluginNamespace;
    }

    const char* Yolov5PluginCreator::getPluginNamespace() const {
        return mNamespace.c_str();
    }

    REGISTER_TENSORRT_PLUGIN(Yolov5PluginCreator);
}
