#ifndef LIB_INCLUDE_IBOW_LC_DETECTOR_MULTI_CENTRALIZED_H_
#define LIB_INCLUDE_IBOW_LC_DETECTOR_MULTI_CENTRALIZED_H_
#include "obindex2/binary_index.h"
#include "ibow-lcd/Agent.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/xfeatures2d.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

class Agent;

class LCDetectorMultiCentralized
{
  public:
    LCDetectorMultiCentralized(unsigned agents,
                    std::vector<std::string> &imageFiles,
                    obindex2::ImageIndex* centralOb);
    void process();
    unsigned displayImages();

  private:
    unsigned imagesPerAgent_;
    unsigned agents_;
    std::vector<std::vector<std::string>> filesPerAgent_;
    std::vector<boost::thread *> agentObjects_;
    std::vector<std::pair<unsigned, cv::Mat>> outP_;//for debugging porpuses
    unsigned totalNumberOfImages_;
    boost::mutex locker_;
    obindex2::ImageIndex* centralOb_;
};
#endif