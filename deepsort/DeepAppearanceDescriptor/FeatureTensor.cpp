/*
 * FeatureTensor.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: zy
 */

#include "FeatureTensor.h"
#include <iostream>



FeatureTensor *FeatureTensor::m_instance = NULL;

FeatureTensor *FeatureTensor::getInstance(const int gpuId, const std::string enginePath) {
	if(m_instance == NULL) {
		m_instance = new FeatureTensor(gpuId,enginePath);
	}
	return m_instance;
}

FeatureTensor::FeatureTensor() {}

FeatureTensor::FeatureTensor(const int gpuId, const std::string enginePath) {
	//prepare model:
	bool status = init(gpuId, enginePath);
	if(status == false)
	  {
	    std::cout<<"init failed"<<std::endl;
	    exit(1);
	  }
	else {
	    std::cout<<"init succeed"<<std::endl;
	  }
}

FeatureTensor::~FeatureTensor() {
	if(!m_context)
		m_context->destroy();
	if(!m_runtime)
		m_runtime->destroy();
	if(!m_engine)
		m_engine->destroy();

}

bool FeatureTensor::init(const int gpuId, const std::string enginePath) {

	//if(!this->set_gpu_id(gpuId)){
	//	return false;
	//}

	std::vector<char> trtModelStream;
	size_t size{0};
	std::ifstream file(enginePath, std::ios::binary);
	if (file.good())
	{
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream.resize(size);
		file.read(trtModelStream.data(), size);
		file.close();
	} else {
		printf("deepsort读取解析file 失败\n");
		return false;
	}

	// deserialize the engine
	m_runtime = nvinfer1::createInferRuntime(gLogger);
	if(m_runtime == nullptr){
		printf("m_runtime 创建失败\n");
		return false;
	}

	//m_engine = m_runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
	m_engine = m_runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
	if (m_engine == nullptr){
		printf("m_engine 创建失败");
		return false;
	}
	//trtModelStream->destroy();

	m_context = m_engine->createExecutionContext();
	if(m_context == nullptr){
		printf("m_context 创建失败");
		return false;
	}
	m_featureDim = DIM; //128


	return true;
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
	std::vector<cv::Mat> mats;
	for(DETECTION_ROW& dbox : d) {
		cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
				int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
		rc.width = rc.height * 0.5;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img.cols? rc.width: (img.cols-rc.x));
		rc.height = (rc.y + rc.height <= img.rows? rc.height:(img.rows - rc.y));

		cv::Mat mattmp = img(rc).clone();
		//cv::Mat mattmp = img.clone();
		cv::resize(mattmp, mattmp, cv::Size(INPUT_W, INPUT_H));
		mats.push_back(mattmp);
	}
	int count = mats.size();

	// float mean[3] = {0.485,0.456,0.406}; 
	// float std[3] = {0.229,0.224,0.225}; 

	float* data = new float[count * INPUT_C * INPUT_H * INPUT_W];
	this->tobuffer(mats, data);
   
    // run inference
    float tensor_buffer[count * this->m_featureDim];

    this->doInference(*m_context, data, tensor_buffer, count);

    delete[] data;

	int i = 0;
	for(DETECTION_ROW& dbox : d) {
		for(int j = 0; j < m_featureDim; j++)
			dbox.feature[j] = tensor_buffer[i*m_featureDim+j];
		i++;
	}
	return true;
}

bool FeatureTensor::set_gpu_id(const int id)
{
	cudaError_t status = cudaSetDevice(id);
	if (status != cudaSuccess)
	{
		std::cout << "gpu id :" + std::to_string(id) + " not exist !" << std::endl;
		return false;
	}
	return true;
}

void FeatureTensor::tobuffer(const std::vector<cv::Mat> &imgs, float *buf) {
	int pos = 0;
	for(const cv::Mat& img : imgs) {
		int Lenth = img.rows * img.cols * 3;
		int nr = img.rows;
		int nc = img.cols;
		if(img.isContinuous()) {
			nr = 1;
			nc = Lenth;
		}
		for(int i = 0; i < nr; i++) {
			const uchar* inData = img.ptr<uchar>(i);
			for(int j = 0; j < nc; j++) {
				buf[pos] = (*inData++)/255.0;
				pos++;
			}
		}//end for
	}//end imgs;
}

void FeatureTensor::doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize)
{
	/*for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; ++i){
	        	std::cout << input[i] << std::endl;
	}*/
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    //std::cout<< "getMaxBatchSize:" << engine.getMaxBatchSize() << std::endl;
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    //std::cout << engine.getNbBindings() << std::endl;
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = 0, outputIndex = 1;


    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * DIM * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * DIM * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}
void FeatureTensor::test()
{
return;
}
