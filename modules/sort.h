#ifndef __SORT_H__
#define __SORT_H__

#include <opencv2/opencv.hpp>

#include "Hungarian.h"
#include "KalmanTracker.h"
#include "class_detector.h"
using namespace cv;



class SortTracker
{
public:
	SortTracker();
	double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
    bool SORT(const std::vector<DetectBox> &detectResults, int frame_count, std::vector<TrackingBox> &trackingResults);
private:
	// sort 跟踪器
    std::vector<KalmanTracker> m_trackers;
};




#endif