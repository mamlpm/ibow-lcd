#ifndef LIB_INCLUDE_IBOW_LC_DETECTOR_MULTI_CENTRALIZED_H_
#define LIB_INCLUDE_IBOW_LC_DETECTOR_MULTI_CENTRALIZED_H_
#include "obindex2/binary_index.h"
#include "ibow-lcd/island.h"
#include "ibow-lcd/Agent.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
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
                             obindex2::ImageIndex *centralOb,
                             unsigned p,
                             double mScore);
  void process(std::vector<std::string> &imageFiles);
  void processImage(unsigned agentN,
                    unsigned imageId,
                    unsigned gImageId,
                    const cv::Mat &descs,
                    std::vector<cv::KeyPoint> keyPoints,
                    std::vector<cv::KeyPoint> stableKeyPoints,
                    bool lookForLoop);
  void filterMatches(std::vector<std::vector<cv::DMatch>> &candidatesToFilter,
                     std::vector<cv::DMatch> *filteredCandidates);
  void filterCandidates(
      const std::vector<obindex2::ImageMatch> &image_matches,
      std::vector<obindex2::ImageMatch> *image_matches_filt);

static bool compareByScore (const obindex2::ImageMatch& a, const obindex2::ImageMatch& b);

private:
  unsigned imagesPerAgent_;
  unsigned agents_;
  std::vector<std::vector<std::string>> filesPerAgent_;
  boost::thread_group agentSim_;
  unsigned totalNumberOfImages_;
  boost::mutex locker_;
  obindex2::ImageIndex *centralOb_;
  unsigned p_;
  double min_score_;
};
#endif