#include "ibow-lcd/LCDetectorMultiCentralized.h"

LCDetectorMultiCentralized::LCDetectorMultiCentralized(unsigned agents,
                                                       obindex2::ImageIndex *centralOb,
                                                       const unsigned p)
{
    agents_ = agents;
    centralOb_ = centralOb;
    p_ = p;
}

void LCDetectorMultiCentralized::process(std::vector<std::string> &imageFiles)
{
    totalNumberOfImages_ = imageFiles.size();
    imagesPerAgent_ = totalNumberOfImages_ / agents_;
    unsigned firstImage[agents_];

    unsigned aux = 0;
    for (unsigned i = 0; i < agents_; i++)
    {
        firstImage[i] = i * imagesPerAgent_;
        std::vector<std::string> fiAgent;
        for (unsigned f = 0; f < imagesPerAgent_; f++)
        {
            fiAgent.push_back(imageFiles[aux]);
            aux++;
        }

        if (i == agents_ - 1)
        {
            while (aux < totalNumberOfImages_)
            {
                fiAgent.push_back(imageFiles[aux]);
                aux++;
            }
        }
        filesPerAgent_.push_back(fiAgent);

        std::cout << "I'm agent " << i << " and I have to process " << filesPerAgent_[i].size() << " Images" << std::endl;
    }

    for (unsigned i = 0; i < filesPerAgent_.size(); i++)
    {
        Agent *a = new Agent(this, filesPerAgent_[i], &locker_, i, firstImage[i]);
        agentSim_.create_thread(boost::bind(&Agent::run, a));
    }

    agentSim_.join_all();
}

void LCDetectorMultiCentralized::processImage(unsigned agentN,
                                              unsigned imageId,
                                              unsigned gImageID,
                                              const cv::Mat &descs,
                                              std::vector<cv::KeyPoint> keyPoints,
                                              std::vector<cv::KeyPoint> stableKeyPoints,
                                              bool lookForLoop)
{
    std::cout << "This is central server "
              << "and now I should be processing agent's " << agentN << " "
              << imageId << " which is the " << gImageID << " processed image" << std::endl;
    std::vector<std::vector<cv::DMatch>> mtchs;
    centralOb_->searchDescriptors(descs,
                                  &mtchs);
    std::vector<cv::DMatch> usableMatches;
    filterCandidates(mtchs, &usableMatches);
    std::unordered_map <unsigned, centralOb_ -> ImageMatch> iMatch;
    centralOb_ -> searchImagesRestrictive(descs, usableMatches, &iMatch, agentN, 0, p_);
}

void LCDetectorMultiCentralized::filterCandidates(std::vector<std::vector<cv::DMatch>> &candidatesToFilter,
                                                  std::vector<cv::DMatch> *filteredCandidates)
{
    filteredCandidates->clear();
    for (unsigned i = 0; i < candidatesToFilter.size(); i++)
    {
        if (candidatesToFilter[i][0].distance < 0.8 * candidatesToFilter[i][1].distance)
        {
            filteredCandidates->push_back(candidatesToFilter[i][0]);
        }
    }
}

// unsigned LCDetectorMultiCentralized::displayImages()
// {
//     unsigned e = 1;
//     for (unsigned i = 0; i < outP_.size(); i++)
//     {
//         cv::imshow(std::to_string(outP_[i].first), outP_[i].second);
//         cv::waitKey(5);
//         usleep(100000);
//         // e++;
//         // std::cout << e << std::endl;
//     }
//     e = outP_.size();
//     outP_.clear();
//     return e;
// }