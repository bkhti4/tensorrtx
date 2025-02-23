#include "yololayer.h"

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin()
    {
        mClassCount = CLASS_NUM;
        mYoloKernel.clear();
        mYoloKernel.push_back(yolo1);
        mYoloKernel.push_back(yolo2);
        mYoloKernel.push_back(yolo3);
        mKernelCount = mYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t anchorLen = sizeof(float) * CHECK_COUNT * 2;
        for (int i = 0; i < mKernelCount; i++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[i], anchorLen));
            const auto& yolo = mYoloKernel[i];
            CUDA_CHECK(cudaMemcpy(mAnchor[i], yolo.anchors, anchorLen, cudaMemcpyHostToDevice));
        }
    }

    YoloLayerPlugin::~YoloLayerPlugin()
    {
        for (int i = 0; i < mKernelCount; i++)
        {
            CUDA_CHECK(cudaFree(mAnchor[i]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        assert(d == a + length);

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t anchorLen = sizeof(float) * CHECK_COUNT * 2;
        for (int i = 0; i < mKernelCount; i++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[i], anchorLen));
            const auto& yolo = mYoloKernel[i];
            CUDA_CHECK(cudaMemcpy(mAnchor[i], yolo.anchors, anchorLen, cudaMemcpyHostToDevice));
        }
    }

    void YoloLayerPlugin::serialize(void* buffer) const noexcept
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const noexcept
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(Yolo::YoloKernel) * mYoloKernel.size();
    }

    int YoloLayerPlugin::initialize() noexcept
    {
        return 0;
    }

    DimsExprs YoloLayerPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
    {
        //output the result to channel
        int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);
        DimsExprs de;
        de.nbDims = 2;
        de.d[0] = exprBuilder.constant(inputs[0].d[0]->getConstantValue());  // batchsize
        de.d[1] = exprBuilder.constant(totalsize + 1);  // outputsize
        return de;
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const noexcept
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
    {
        return DataType::kFLOAT;
    }

    void YoloLayerPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() noexcept {}

    const char* YoloLayerPlugin::getPluginType() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const noexcept
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() noexcept
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2DynamicExt* YoloLayerPlugin::clone() const noexcept
    {
        YoloLayerPlugin *p = new YoloLayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output, int noElements,
            int yoloWidth, int yoloHeight, int yoloStride, const float anchors[CHECK_COUNT * 2], int classes, int outputElem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

            float *res_count = output + bnIdx * outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;
            char* data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
            Detection* det = (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * yoloStride;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * yoloStride;
            det->bbox[2] = expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2 * k];
            det->bbox[3] = expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2 * k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);
        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (size_t i = 0; i < mYoloKernel.size(); ++i) {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width * yolo.height * batchSize;
            CalDetection<<<(yolo.width * yolo.height * batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i], output, numElem, yolo.width, yolo.height, yolo.stride, (float*)mAnchor[i], mClassCount, outputElem);
        }
    }

    int YoloLayerPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        int batchSize = inputDesc[0].dims.d[0];
        for (size_t i = 0; i < mYoloKernel.size(); ++i) {
            mYoloKernel[i].width = inputDesc[i].dims.d[3];
            mYoloKernel[i].height = inputDesc[i].dims.d[2];
        }
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const noexcept
    {
            return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const noexcept
    {
            return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames() noexcept
    {
            return &mFC;
    }

    IPluginV2DynamicExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2DynamicExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
