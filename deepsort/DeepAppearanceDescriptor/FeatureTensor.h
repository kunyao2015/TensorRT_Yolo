#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "model.h"
#include "dataType.h"

#include "NvInfer.h"
#include "logger.h"
#include "common.h"
#include <cudnn.h>
#include <cuda_runtime_api.h>

typedef unsigned char uint8;

static const int INPUT_C = 3;
static const int INPUT_H = 128;
static const int INPUT_W = 64;
static const int DIM = 512;

class FeatureTensor
{
public:
	static FeatureTensor* getInstance(const int gpuId, const std::string enginePath);
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);

private:
	FeatureTensor();
	FeatureTensor(const int gpuId, const std::string enginePath);
	FeatureTensor(const FeatureTensor&);
	FeatureTensor& operator = (const FeatureTensor&);
	static FeatureTensor* m_instance;
	bool init(const int gpuId, const std::string enginePath);
	~FeatureTensor();

	void tobuffer(const std::vector<cv::Mat> &imgs, float *buf);
	bool set_gpu_id(const int id = 0);
	void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize);

	int m_featureDim;
    
    nvinfer1::IRuntime* m_runtime;
	nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IExecutionContext* m_context;
public:
	void test();
};
