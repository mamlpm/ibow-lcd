#ifndef LIB_INCLUDE_IBOW_LC_DETECTOR_MULTI_CENTRALIZED_H_
#define LIB_INCLUDE_IBOW_LC_DETECTOR_MULTI_CENTRALIZED_H_
#include "obindex2/binary_index.h"
#include "ibow-lcd/IslanDistributed.h"
#include "ibow-lcd/Agent.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <opencv2/xfeatures2d.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

enum LCDetectorMultiCentralizedStatus
{
  LC_DETECTED,
  LC_NOT_DETECTED,
  LC_NOT_ENOUGH_IMAGES,
  LC_NOT_ENOUGH_ISLANDS,
  LC_NOT_ENOUGH_INLIERS,
  LC_TRANSITION
};

struct LCDetectorMultiCentralizedResult
{
  LCDetectorMultiCentralizedResult() : status(LC_NOT_DETECTED),
                                       QagentId(0),
                                       TagentId(0),
                                       query_id(1),
                                       train_id(-1) {}

  inline bool isLoop()
  {
    return status == LC_DETECTED;
  }

  LCDetectorMultiCentralizedStatus status;
  unsigned QagentId;
  unsigned TagentId;
  unsigned query_id;
  unsigned train_id;
  unsigned inliers;
};

class Agent;
// class IslanDistributed;

class LCDetectorMultiCentralized
{
public:
  LCDetectorMultiCentralized(unsigned agents,
                             obindex2::ImageIndex *centralOb,
                             unsigned p,
                             double mScore,
                             std::unordered_map<unsigned, std::vector<std::pair<unsigned, obindex2::ImageMatch>>> *fReslt,
                             unsigned island_size,
                             int min_consecutive_loops,
                             unsigned min_inliers,
                             float nndr_bf,
                             double epDist,
                             double confProb);

  void process(std::vector<std::string> &imageFiles);

  void processImage(unsigned agentN,
                    unsigned imageId,
                    unsigned gImageId,
                    const cv::Mat &descs,
                    const cv::Mat &stableDescs,
                    std::vector<cv::KeyPoint> keyPoints,
                    std::vector<cv::KeyPoint> stableKeyPoints,
                    bool lookForLoop,
                    std::vector<std::pair<unsigned, obindex2::ImageMatch>> *result);

  void filterMatches(std::vector<std::vector<cv::DMatch>> &candidatesToFilter,
                     std::vector<cv::DMatch> *filteredCandidates);

  void filterCandidates(
      const std::vector<obindex2::ImageMatch> &image_matches,
      std::vector<obindex2::ImageMatch> *image_matches_filt);

  void addImage(const unsigned image_id,
                const unsigned globalImgId,
                const unsigned agentId,
                const std::vector<cv::KeyPoint> &kps,
                const cv::Mat &descs);

  static bool compareByScore(const obindex2::ImageMatch &a, const obindex2::ImageMatch &b);

  void buildIslands(const std::vector<obindex2::ImageMatch> &image_matches,
                    std::vector<ibow_lcd::IslanDistributed> *islands);

  void getPriorIslands(const ibow_lcd::IslanDistributed &island,
                       const std::vector<ibow_lcd::IslanDistributed> &islands,
                       std::vector<ibow_lcd::IslanDistributed> *p_islands);

  void ratioMatchingBF(const cv::Mat &query,
                       const cv::Mat &train,
                       std::vector<cv::DMatch> *matches);

  void convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                     const std::vector<cv::KeyPoint> &train_kps,
                     const std::vector<cv::DMatch> &matches,
                     std::vector<cv::Point2f> *query,
                     std::vector<cv::Point2f> *train);

  unsigned checkEpipolarGeometry(const std::vector<cv::Point2f> &query,
                                 const std::vector<cv::Point2f> &train);

private:
  unsigned* globalImagePointer_;
  unsigned imagesPerAgent_;
  unsigned agents_;
  std::vector<std::vector<std::string>> filesPerAgent_;
  boost::thread_group agentSim_;
  unsigned totalNumberOfImages_;
  boost::mutex locker_;
  obindex2::ImageIndex *centralOb_;
  unsigned p_;
  double minScore_;
  unsigned islandOffset_;
  unsigned islandSize_;
  std::vector<unsigned> currentImagePerAgent_;
  std::unordered_map<unsigned, std::vector<std::pair<unsigned, obindex2::ImageMatch>>> *fReslt_;
  LCDetectorMultiCentralizedResult lastLcResult_;
  std::vector<ibow_lcd::IslanDistributed> lastLcIsland_;
  int consecutiveLoops_;
  int minConsecutiveLoops_;
  std::vector<std::vector<cv::Mat>> prevDescs_;
  std::vector<std::vector<std::vector<cv::KeyPoint>>> prevKps_;
  unsigned minInliers_;
  float nndrBf_;
  double epDist_;
  double confProb_;
};
#endif