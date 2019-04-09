#ifndef LIB_INCLUDE_IBOW_LCD_AGENT_H_
#define LIB_INCLUDE_IBOW_LCD_AGENT_H_

// #include "obindex2/binary_index.h"
#include "ibow-lcd/LCDetectorMultiCentralized.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/xfeatures2d.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

class LCDetectorMultiCentralized;

class Agent
{
public:
  Agent(LCDetectorMultiCentralized *centralCerv,
        std::vector<std::string> &fileNames,
        boost::mutex *locker,
        unsigned agentId,
        unsigned firstImageId);
  unsigned getId();
  void run();

private:
  LCDetectorMultiCentralized *centr_;       //central server
  std::vector<std::string> fileNames_;      //files which contain the images waiting to be processed
  std::vector<cv::KeyPoint> prevKeyPoints_; //keypoints seen on the previous image
  cv::Mat previousImage_;
  unsigned agentId_;
  unsigned nImages_;
  cv::Mat prevDescriptors_;
  boost::mutex *locker_;
  unsigned gImageId_;
};
#endif