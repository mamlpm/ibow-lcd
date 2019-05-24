#ifndef LIB_INCLUDE_IBOW_LCD_AGENT_DISTRIBUTED_H_
#define LIB_INCLUDE_IBOW_LCD_AGENT_DISTRIBUTED_H_

#include "obindex2/binary_index.h"
#include "ibow-lcd/IslanDistributed.h"
#include "ibow-lcd/LCDetectorMultiCentralized.h"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/xfeatures2d.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// enum AgentDistributedStatus
// {
//   LC_DETECTED,
//   LC_NOT_DETECTED,
//   LC_NOT_ENOUGH_IMAGES,
//   LC_NOT_ENOUGH_ISLANDS,
//   LC_NOT_ENOUGH_INLIERS,
//   LC_TRANSITION
// };

struct AgentDistributedResult
{
  AgentDistributedResult() : status(LC_NOT_DETECTED),
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

class middleLayer;

class AgentDistributed
{
public:
  AgentDistributed(std::vector<obindex2::ImageIndex> *trees,
                   unsigned agents,
                   std::vector<std::string> &flNames,
                   unsigned agentId,
                   unsigned firstImageId,
                   std::vector<unsigned> *currImPAgent,
                   bool filter,
                   bool original,
                   unsigned p,
                   std::vector<std::vector<std::vector<cv::KeyPoint>>> *prevKps,
                   std::vector<std::vector<cv::Mat>> *prevDescs,
                   double mScore,
                   std::vector<std::vector<int>> *fReslt,
                   unsigned island_size,
                   int min_consecutive_loops,
                   unsigned min_inliers,
                   float nndr_bf,
                   double epDist,
                   double confProb,
                   unsigned* globalImagePointer,
                   boost::mutex* locker);

  unsigned getId();

  void run();

  void processImage(unsigned agentN,
                    unsigned agents,
                    unsigned imageId,
                    unsigned gImageId,
                    const cv::Mat &descs,
                    const cv::Mat &stableDescs,
                    std::vector<cv::KeyPoint> keyPoints,
                    std::vector<cv::KeyPoint> stableKeyPoints,
                    bool lookForLoop,
                    bool aImage);

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
  std::vector<std::string> fileNames_;      //files which contain the images waiting to be processed
  std::vector<cv::KeyPoint> prevKeyPoints_; //keypoints seen on the previous image
  cv::Mat previousImage_;
  std::vector<obindex2::ImageIndex> *trees_;
  unsigned agentId_;
  unsigned nImages_;
  cv::Mat prevDescriptors_;
  boost::mutex* locker_;
  unsigned gImageId_;
  std::vector<unsigned> *currImPAgent_;
  bool filter_;
  bool original_;
  unsigned agents_;
  std::vector<std::vector<std::vector<cv::KeyPoint>>> *prevKps_;
  unsigned p_;
  std::vector<std::vector<int>> *fReslt_;
  AgentDistributedResult lastLcResult_;
  ibow_lcd::IslanDistributed lastLcIsland_;
  std::vector<unsigned>* currentImagePerAgent_;
  int consecutiveLoops_;
  int minConsecutiveLoops_;
  std::vector<std::vector<cv::Mat>> *prevDescs_;
  unsigned minInliers_;
  float nndrBf_;
  double epDist_;
  double confProb_;
  double minScore_;
  unsigned islandOffset_;
  unsigned islandSize_;
  unsigned* globalImagePointer_;
};
#endif