#ifndef CLASS_DETECTOR_H_
#define CLASS_DETECTOR_H_

#include "API.h"
#include <iostream>
#include <opencv2/opencv.hpp>

typedef struct DetectBox
{
	int		 class_id = -1;
	float	 prob	= 0.f;
	std::string class_name;
	cv::Rect rect;
}DetectResult;

typedef struct TrackingBox
{
    int frame = 0;
    int id = -1;
    float	 prob	= 0.f;
    std::string class_name;
    cv::Rect_<float> box;
}TrackingBox;

enum ModelType
{
	YOLOV2 = 0,
	YOLOV3,
	YOLOV2_TINY,
	YOLOV3_TINY
};

enum Precision
{
	INT8 = 0,
	FP16,
	FP32
};

enum TrackType
{
	SORT = 0,
	DEEPSORT
};

struct Config
{
	std::string file_model_cfg					= "configs/yolov3.cfg";

	std::string file_model_weights				= "configs/yolov3.weights";

	float detect_thresh							= 0.9;

	ModelType	net_type						= YOLOV3;

	Precision	inference_precison				= FP32;
	
	int	gpu_id									= 0;

	std::string calibration_image_list_file_txt = "";

	TrackType track_type                        = SORT;

	std::string deepsort_file                   = "configs/deepsort.engine";

};

class API Detector
{
public:
	explicit Detector();

	~Detector();

	void init(const Config &config);

	//void detect(const cv::Mat &mat_image, std::vector<Result> &vec_result);
	void detect(unsigned char* imgdata, int width, int height,int channel, int frameNo, std::vector<TrackingBox> &vec_result);

private:
	
	Detector(const Detector &);
	const Detector &operator =(const Detector &);
	class Impl;
	Impl *_impl;
};

#endif // !CLASS_QH_DETECTOR_H_