#include "sort.h"


SortTracker::SortTracker() 
{
    // sort 跟踪器
    KalmanTracker::kf_count = 0;
    std::cout<<"KalmanTracker 初始化：" << KalmanTracker::kf_count<<std::endl;
}

// Computes IOU between two bounding boxes
double SortTracker::GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

bool SortTracker::SORT(const std::vector<DetectBox> &detectResults, int frame_count, std::vector<TrackingBox> &trackingResult)
{
    // update across frames
    // int max_age = 3; // 1
    // int min_hits = 0; //3
    // double iouThreshold = 0.3;
    
    int max_age = 5; // 1
    int min_hits = 0; //3
    double iouThreshold = 0.3;


    // variables used in the for-loop
    std::vector<cv::Rect_<float>> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix;
    std::vector<int> assignment;
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    //vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    /////////////////////////////////////////////
    std::cout << "sort frame_count:" << frame_count << std::endl;
    std::cout << "m_trackers size:" << m_trackers.size() << std::endl;
    if (m_trackers.size() == 0) // the first frame met
    {
        // initialize kalman trackers using first detections.
        for (unsigned int i = 0; i < detectResults.size(); i++)
        {
            float x = (float)detectResults[i].rect.x;
            float y = (float)detectResults[i].rect.y;
            float width = (float)detectResults[i].rect.width;
            float height = (float)detectResults[i].rect.height;
            StateType box(x,y,width,height);
            KalmanTracker trk = KalmanTracker(box);
            m_trackers.push_back(trk);
        }

        return false; // 可以更改,暂时第一帧放弃处理，仅作tracker的初始化
    }

    ///////////////////////////////////////
    //  get predicted locations from existing trackers.
    predictedBoxes.clear();

    for (auto it = m_trackers.begin(); it != m_trackers.end();)
    {
        Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            it++;
        }
        else
        {
            std::cout <<frame_count << ":" <<pBox.x  << " " << pBox.y<< std::endl;
            it = m_trackers.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }

    ///////////////////////////////////////
    // associate detections to tracked object (both represented as bounding boxes)
    // dets : detFrameData[fi]
    trkNum = predictedBoxes.size();
    detNum = detectResults.size();

    std::cout << "trkNum:" << trkNum << std::endl;
    std::cout << "detNum:" << detNum <<std::endl;

    if((0 == trkNum) && (0 == detNum))
    {
        std::cout<<m_trackers.size()<<std::endl;
        std::cout<<"trkNum = 0; detNum = 0"<<std::endl;
        return false;
    }

    iouMatrix.clear();
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            float x = (float)detectResults[j].rect.x;
            float y = (float)detectResults[j].rect.y;
            float width = (float)detectResults[j].rect.width;
            float height = (float)detectResults[j].rect.height;
            StateType box(x,y,width,height);
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    // find matches, unmatched_detections and unmatched_predictions
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();

    if (detNum > trkNum) //	there are unmatched detections
    {
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
            matchedItems.begin(), matchedItems.end(),
            insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }

    // filter out matched with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }

    ///////////////////////////////////////
    //  updating trackers

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        float x = (float)detectResults[detIdx].rect.x;
        float y = (float)detectResults[detIdx].rect.y;
        float width = (float)detectResults[detIdx].rect.width;
        float height = (float)detectResults[detIdx].rect.height;
        StateType box(x,y,width,height);
        m_trackers[trkIdx].update(box);
    }
    std::cout<< "matchedPairs size:" << matchedPairs.size() << std::endl;
    std::cout<< "unmatchedDetections size:" << unmatchedDetections.size() << std::endl;

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        float x = (float)detectResults[umd].rect.x;
        float y = (float)detectResults[umd].rect.y;
        float width = (float)detectResults[umd].rect.width;
        float height = (float)detectResults[umd].rect.height;
        StateType box(x,y,width,height);
        KalmanTracker tracker = KalmanTracker(box);
        m_trackers.push_back(tracker);
    }

    // get trackers' output

    trackingResult.clear();

    for (auto it = m_trackers.begin(); it != m_trackers.end();)
    {
//           Change
// if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
// to
// if ((trk.time_since_update <= self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
        if (((*it).m_time_since_update < 1 ) &&
            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
        //if (((*it).m_time_since_update < max_age ) &&
        //    ((*it).m_hits >= min_hits || frame_count <= min_hits))

        {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = frame_count;
            trackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != m_trackers.end() && (*it).m_time_since_update > max_age )
            it = m_trackers.erase(it);
    }
    
    // for (auto &vr : trackingResult)
    // {
    //     std::vector<double> iouValue;
    //     for (const auto &dr : detectResults)
    //     {
    //         double iou = GetIOU(vr.box, dr.rect);
    //         iouValue.push_back(iou);
    //     }
    //     int maxPosition = std::max_element(iouValue.begin(),iouValue.end()) - iouValue.begin();
    //     float prob = detectResults[maxPosition].prob;
    //     std::string class_name = detectResults[maxPosition].class_name;
    //     vr.prob = prob;
    //     vr.class_name = class_name;
    // }


    // for (auto tb : frameTrackingResult)
    // 	resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
    return true;


}