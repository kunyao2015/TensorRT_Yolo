#ifndef CLASS_YOLO_DETECTOR_HPP_
#define CLASS_YOLO_DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolov2.h"
#include "yolov3.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <chrono>
#include <stdio.h>  /* defines FILENAME_MAX */
#include<algorithm>

#include "class_detector.h"
#include "sort.h"
//struct Result
//{
//	int		 id = -1;
//	float	 prob = 0.f;
//	cv::Rect rect;
//};
//
//enum ModelType
//{
//	YOLOV2 = 0,
//	YOLOV3,
//	YOLOV2_TINY,
//	YOLOV3_TINY
//};
//
//enum Precision
//{
//	INT8 = 0,
//	FP16,
//	FP32
//};

//struct Config
//{
//	std::string file_model_cfg						= "configs/yolov3.cfg";
//
//	std::string file_model_weights					= "configs/yolov3.weights";
//
//	float detect_thresh								= 0.9;
//
//	ModelType	net_type							= YOLOV3;
//
//	Precision	inference_precison					= INT8;
//
//	std::string calibration_image_list_file_txt     = "configs/calibration_images.txt";
//};

class YoloDectector
{
public:
	YoloDectector()
	{

	}
	~YoloDectector()
	{

	}

	void init(const Config &config)
	{
		_config = config;
		//_sort = new SortTracker();

		this->set_gpu_id(_config.gpu_id);

		this->parse_config();

		this->build_net();
		
	}
/*
	void detect(const cv::Mat		&mat_image,
				std::vector<Result> &vec_result)
	{
		std::vector<DsImage> vec_ds_images;
		vec_result.clear();
		vec_ds_images.emplace_back(mat_image, _p_net->getInputH(), _p_net->getInputW());
		cv::Mat trtInput = blobFromDsImages(vec_ds_images, _p_net->getInputH(),_p_net->getInputW());
		_p_net->doInference(trtInput.data, vec_ds_images.size());
		for (uint32_t i = 0; i < vec_ds_images.size(); ++i)
		{
			auto curImage = vec_ds_images.at(i);
			auto binfo = _p_net->decodeDetections(i, curImage.getImageHeight(), curImage.getImageWidth());
			auto remaining = nmsAllClasses(_p_net->getNMSThresh(), binfo, _p_net->getNumClasses());
			for (const auto &b : remaining)
			{
				Result res;
				res.id = b.label;
				res.prob = b.prob;
				const int x = b.box.x1;
				const int y = b.box.y1;
				const int w = b.box.x2 - b.box.x1;
				const int h = b.box.y2 - b.box.y1;
				res.rect = cv::Rect(x, y, w, h);
				res.class_name = _p_net->getClassName(b.label);
				std::cout << "Label:" << b.label <<std::endl;
				std::cout << "className:" << _p_net->getClassName(b.label) << std::endl;
				vec_result.push_back(res);
			}
		}
	}
*/
	void detect(unsigned char* imgdata, int width, int height,int channel, int frameNo,
				std::vector<TrackingBox> &vec_result)
	{
		
		cv::Mat image(height,width,CV_8UC3); 
		image.data = imgdata;
		//std::cout << image.size() << std::endl;
		int inputH = _p_net->getInputH();
		int inputW = _p_net->getInputW();
		//std::cout<< "width:"<<width << ";height:" << height << ";channel:" <<channel <<std::endl;
		//std::cout << "inputH:" << inputH << ";inputW:" << inputW <<std::endl;
		cv::Mat imageRGB;
		cv::resize(image,image,cv::Size(inputW,inputH),0,0,cv::INTER_LINEAR);
		cv::cvtColor(image,imageRGB,cv::COLOR_BGR2RGB);
		//cv::resize(image,image,cv::Size(inputW,inputH),0,0,cv::INTER_LINEAR);

		//float mean[3] = {0.485,0.456,0.406}; 
		//float std[3] = {0.229,0.224,0.225}; 
        /*
		float* inputdata = new float[1 * channel * inputH * inputW];
		for (int c=0; c < channel; ++c){
			for (unsigned j=0, volChl = inputW*inputW; j<volChl; ++j) {
				inputdata[c*volChl+j] = (float(image.data[j*channel +2 -c])/255.0 - mean[c])/std[c];
				//std::cout<< data[c*volChl+j] << std::endl;
			}
		}
		*/
		// std::cout << imageRGB.size() << std::endl;
		// cv::Mat ch[3];
		// cv::split(imageRGB,ch);
		// std::cout << ch[0].size() << std::endl;
		
		float* inputdata = new float[1 * channel * inputH * inputW];
		for (int c=0; c < channel; ++c){
			for (int j=0, volChl = inputW*inputW; j<volChl; ++j) {
				inputdata[c*volChl+j] = image.data[j*channel +2 -c];
				//inputdata[c*volChl+j] = image.data[j*channel + c];
				//std::cout<< data[c*volChl+j] << std::endl;
			}
		}
		
		// imageRGB.convertTo(imageRGB,CV_32F);
		cv::Scalar mean = cv::Scalar(0.0, 0.0, 0.0);
		imageRGB -= mean;
		imageRGB *= 1.0;
		//std::cout<<"******1111*********"<<std::endl;
		
		int sz[] = { 1, 3, inputH, inputW };
		cv::Mat InputBlob = cv::Mat(4, sz, CV_32F,inputdata);
        //InputBlob.create(4, sz, CV_32F,inputdata);

        // cv::Mat ch[4];
		// std::cout<<"******1112*********"<<std::endl;

		// const cv::Mat& img = imageRGB;

		// int nch = img.channels();
		// std::cout<<nch<<std::endl;
        // std::cout<<"******1113*********"<<std::endl;
		// for( int j = 0; j < nch; j++ )
		// 	ch[j] = cv::Mat(inputH, inputW, CV_32F, InputBlob.ptr(0, j));
		// if(true)
		// 	std::swap(ch[0], ch[2]);
		// split(img, ch);
		// std::cout<<"******1114*********"<<std::endl;


		_p_net->doInference(InputBlob.data, 1);
		delete[] inputdata;
		
		auto binfo = _p_net->decodeDetections(0, height, width);
		auto remaining = nmsAllClasses(_p_net->getNMSThresh(), binfo, _p_net->getNumClasses());
		std::vector<std::string> use_name = {
			"person",        "bicycle",       "car",    
		    "motorbike",     "bus",           "truck"
		};
		std::vector<DetectBox> detect_result;
		for (const auto &b : remaining)
		{
			if (find(use_name.begin(),use_name.end(),_p_net->getClassName(b.label)) == use_name.end()){
				continue;
			}
			DetectBox res;
			res.class_id = b.label;
			res.prob = b.prob;
			const int x = b.box.x1;
			const int y = b.box.y1;
			const int w = b.box.x2 - b.box.x1;
			const int h = b.box.y2 - b.box.y1;
			res.rect = cv::Rect(x, y, w, h);
			res.class_name = _p_net->getClassName(b.label);
			//std::cout << "Label:" << b.label <<std::endl;
			//std::cout << "className:" << _p_net->getClassName(b.label) << std::endl;
			detect_result.push_back(res);
		}
		for (const auto &r : detect_result)
		{
			std::cout << "id:" << r.class_id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
		}

		if (false)
		{
			for (const auto &dr : detect_result)
			{
				TrackingBox res;
	            res.box = dr.rect;
	            res.id = -1;
	            res.prob = dr.prob;
	            res.frame = frameNo;
	            res.class_name = dr.class_name;
	            vec_result.push_back(res);
				
			}

		} 
		else 
		{
            //KalmanTracker::kf_count = 0;
			bool bsort = _sort.SORT(detect_result, frameNo, vec_result);
		    if (!bsort)
		    {
		        return;
		    }
		}
	   //  for i,object in enumerate(objects):
    //     x, y, w, h = object[2]
    //     x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    //     iou_list.append(cal_iou(item[:4],[x1,y1,x2,y2]))
    // max_index = iou_list.index(max(iou_list))
	    /*
	    for (const auto &vr : vec_result)
		{
			std::vector<double> iouValue;
			for (const auto &dr : detect_result)
			{
				double iou = _sort.GetIOU(vr.box, dr.rect);
				iouValue.push_back(iou);
			}
			int maxPosition = std::max_element(iouValue.begin(),iouValue.end()) - iouValue.begin();
			float prob = detect_result[maxPosition].prob;
			std::string class_name = detect_result[maxPosition].class_name;
			std::cout<< prob << std::endl;
			std::cout<< class_name << std::endl;
			vr.prob = prob;
			vr.class_name = class_name;

			//std::cout << "id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
			//cv::rectangle(frame, r.box, cv::Scalar(255, 0, 0), 2);
			//cv::putText(frame, r.class_name.c_str(), cv::Point(r.box.x + 20, r.box.y + 20),
	        //            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 255, 0), 1, CV_AA);
		}*/
		
	}

private:

	void set_gpu_id(const int id = 0)
	{
		cudaError_t status = cudaSetDevice(id);
		if (status != cudaSuccess)
		{
			std::cout << "gpu id :" + std::to_string(id) + " not exist !" << std::endl;
			assert(0);
		}
	}

	void parse_config()
	{
		_yolo_info.networkType = _vec_net_type[_config.net_type];
		_yolo_info.configFilePath = _config.file_model_cfg;
		_yolo_info.wtsFilePath = _config.file_model_weights;
		_yolo_info.precision = _vec_precision[_config.inference_precison];
		_yolo_info.deviceType = "kGPU";
		int npos = _yolo_info.wtsFilePath.find(".weights");
		assert(npos != std::string::npos
			&& "wts file file not recognised. File needs to be of '.weights' format");
		std::string dataPath = _yolo_info.wtsFilePath.substr(0, npos);
		_yolo_info.calibrationTablePath = dataPath + "-calibration.table";
		_yolo_info.enginePath = dataPath + "-" + _yolo_info.precision + ".engine";
		_yolo_info.inputBlobName = "data";

		_infer_param.printPerfInfo = false;
		_infer_param.printPredictionInfo = false;
		_infer_param.calibImages = _config.calibration_image_list_file_txt;
		_infer_param.calibImagesPath = "";
		_infer_param.probThresh = _config.detect_thresh;
		_infer_param.nmsThresh = 0.4;
	}

	void build_net()
	{
		if ((_config.net_type == YOLOV2) || (_config.net_type == YOLOV2_TINY))
		{
			_p_net = std::unique_ptr<Yolo>{ new YoloV2(1, _yolo_info, _infer_param) };
		}
		else if ((_config.net_type == YOLOV3) || (_config.net_type == YOLOV3_TINY))
		{
			_p_net = std::unique_ptr<Yolo>{ new YoloV3(1, _yolo_info, _infer_param) };
		}
		else
		{
			assert(false && "Unrecognised network_type. Network Type has to be one among the following : yolov2, yolov2-tiny, yolov3 and yolov3-tiny");
		}
	}

private:
	Config _config;
	NetworkInfo _yolo_info;
	InferParams _infer_param;
	SortTracker _sort;

	std::vector<std::string> _vec_net_type{ "yolov2","yolov3","yolov2-tiny","yolov3-tiny" };
	std::vector<std::string> _vec_precision{ "kINT8","kHALF","kFLOAT" };
	std::unique_ptr<Yolo> _p_net = nullptr;
};


#endif
