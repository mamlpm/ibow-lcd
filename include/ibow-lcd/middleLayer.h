#ifndef LIB_INCLUDE_MIDDLE_LAYER_H_
#define LIB_INCLUDE_MIDDLE_LAYER_H_

#include "obindex2/binary_index.h"
#include "ibow-lcd/AgentDistributed.h"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/xfeatures2d.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

class AgentDistributed;

class middleLayer
{
public:
  middleLayer(unsigned agents,
              bool purge,
              bool filter,
              bool original,
              unsigned p,
              double mScore,
              std::vector<std::vector<int>> *fResult,
              unsigned islandSize,
              unsigned minConsecutiveLoops,
              unsigned minInliers,
              float nndrBf,
              double epDist,
              double confProb);

  void process(std::vector<std::string> &imageFiles);

private:
  std::vector<obindex2::ImageIndex> agenTrees_;
  boost::thread_group agentSim_;
  unsigned *globalImagePointer_;
  unsigned imagesPerAgent_;
  unsigned agents_;
  std::vector<std::vector<std::string>> filesPerAgent_;
  unsigned totalNumberOfImages_;
  boost::mutex locker_;
  bool purge_;
  std::vector<std::vector<cv::Mat>> prevDescs_;
  std::vector<std::vector<std::vector<cv::KeyPoint>>> prevKps_;
  std::vector<unsigned> currentImagePerAgent_;
  unsigned p_;
  double mScore_;
  std::vector<std::vector<int>> *fResult_;
  unsigned islandSize_;
  unsigned minConsecutiveLoops_;
  unsigned minInliers_;
  float nndrBf_;
  double epDist_;
  double confProb_;
  bool filter_;
  bool original_;
};
#endif