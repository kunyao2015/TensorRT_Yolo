#include <time.h>
#include "class_detector.h"

int main()
{
	Detector detector;
	Config config;
	config.net_type = YOLOV3;
	config.detect_thresh = 0.5;
	config.file_model_cfg = "../configs/yolov3.cfg";
	config.file_model_weights = "../configs/yolov3.weights";
	//config.calibration_image_list_file_txt = "";
	//config.inference_precison = FP32;
	detector.init(config);

    /***************************** Single Video *********************/
    cv::VideoCapture capture;
    cv::Mat frame;
    frame = capture.open("/home/aipc/code-YK/datasets/jiegouhua.avi");
    if(!capture.isOpened())
    {
        std::cout << "can not open ..." << std::endl;
        return -1;
    }
    int frameNo = 0;
    while (capture.read(frame))
    {
    	frameNo++;
    	std::vector<TrackingBox> res;
		std::clock_t t_strat = std::clock();
		//detector.detect(mat_image, res);
		detector.detect(frame.data, frame.cols, frame.rows,frame.channels(),frameNo,res);
		std::cout << "detect time = " << double(std::clock() - t_strat) / CLOCKS_PER_SEC << "s" << std::endl;
		//fps  = ( fps + (1./(time.time()-t1)) ) / 2

		for (const auto &r : res)
		{
			std::cout << "id:" << r.id << " prob:" << r.prob << " rect:" << r.box << std::endl;
			cv::rectangle(frame, r.box, cv::Scalar(255, 0, 0), 2);
			std::string content = r.class_name + "  ID:" + std::to_string(r.id);
			cv::putText(frame, content.c_str(), cv::Point(r.box.x + 20, r.box.y + 20),
	                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 255, 0), 1, CV_AA);
		}
		cv::imshow("image", frame);
        cv::waitKey(10);
        
    }
    capture.release();
    return 0;
	
	/****************************************************************/
    
    /***************************** Single Image *********************/
    /*
	cv::Mat mat_image = cv::imread("../configs/dog.jpg", cv::IMREAD_UNCHANGED);
	std::vector<Result> res;
	for (int i = 0; i < 3; ++i) {
		std::clock_t t_strat = std::clock();
		//detector.detect(mat_image, res);
		detector.detect(mat_image.data, mat_image.cols, mat_image.rows,3,res);
		std::cout << "detect time = " << double(std::clock() - t_strat) / CLOCKS_PER_SEC << "s" << std::endl;

	}

	
	for (const auto &r : res)
	{
		std::cout << "id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
		cv::rectangle(mat_image, r.rect, cv::Scalar(255, 0, 0), 2);
		cv::putText(mat_image, r.class_name.c_str(), cv::Point(r.rect.x + 20, r.rect.y + 20),
        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 255, 0), 1, CV_AA);
	}
	cv::imshow("image", mat_image);
	return 0;
	*/
	//cv::waitKey();
	//std::cin.get();
}
