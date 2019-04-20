#include "ibow-lcd/LCDetectorMultiCentralized.h"

LCDetectorMultiCentralized::LCDetectorMultiCentralized(unsigned agents,
                                                       obindex2::ImageIndex *centralOb,
                                                       unsigned p,
                                                       double mScore,
                                                       std::unordered_map <unsigned, std::vector<std::pair<unsigned, obindex2::ImageMatch>>>* fReslt)
{
    agents_ = agents;
    centralOb_ = centralOb;
    p_ = p;
    min_score_ = mScore;
    fReslt_ = fReslt;
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
        Agent *a = new Agent(this, filesPerAgent_[i], i, firstImage[i], fReslt_);
        agentSim_.create_thread(boost::bind(&Agent::run, a));
    }

    agentSim_.join_all();
}

void LCDetectorMultiCentralized::processImage(unsigned agentN,
                                              unsigned imageId,
                                              unsigned gImageID,
                                              const cv::Mat &descs,
                                              const cv::Mat &stableDescs,
                                              std::vector<cv::KeyPoint> keyPoints,
                                              std::vector<cv::KeyPoint> stableKeyPoints,
                                              bool lookForLoop,
                                              std::vector<std::pair<unsigned, obindex2::ImageMatch>> *result)
{
    locker_.lock();
    // Searching for loops
    std::cout << "Processing image " << imageId << " seen by agent " << agentN << std::endl;
    if (lookForLoop)
    {
        std::vector<std::vector<cv::DMatch>> mtchs;
        centralOb_->searchDescriptors(descs,
                                      &mtchs);
        std::vector<cv::DMatch> usableMatches;
        filterMatches(mtchs, &usableMatches);
        std::unordered_map<unsigned, obindex2::ImageMatch> iMatch;
        std::vector<obindex2::ImageMatch> iMatchVect;
        centralOb_->searchImagesRestrictive(descs, usableMatches, &iMatch, agentN, p_, imageId);
        for (auto it = iMatch.begin(); it != iMatch.end(); it++)
        {
            iMatchVect.push_back(it->second);
        }
        sort(iMatchVect.begin(), iMatchVect.end(), compareByScore);
        if (iMatchVect.size() > 2)
        {
            std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[0].agentId << " image " << iMatchVect[0].image_id << " (" << iMatchVect[0].agentId*imagesPerAgent_ + iMatchVect[0].image_id << ") scoring " << iMatchVect[0].score << std::endl;
            std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[1].agentId << " image " << iMatchVect[1].image_id << " (" << iMatchVect[1].agentId*imagesPerAgent_ + iMatchVect[1].image_id << ") scoring " << iMatchVect[1].score << std::endl;
            std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[2].agentId << " image " << iMatchVect[2].image_id << " (" << iMatchVect[2].agentId*imagesPerAgent_ + iMatchVect[2].image_id << ") scoring " << iMatchVect[2].score << std::endl;
        }
        else
        {
            for (unsigned i = 0; i < iMatchVect.size(); i++)
            {
                std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[i].agentId << " image " << iMatchVect[i].image_id << " (" << iMatchVect[i].agentId*imagesPerAgent_ + iMatchVect[i].image_id << ") scoring " << iMatchVect[i].score << std::endl;
            }
        }
        std::cout << "---" << std::endl;

        std::vector<obindex2::ImageMatch> iMatchFilt;
        filterCandidates(iMatchVect, &iMatchFilt);
        if (iMatchVect.size() > 0)
        {
            result->resize(result->size() + 1);
            result->at(result->size() - 1).first = imageId;
            result->at(result->size() - 1).second = iMatchVect[0];
        }
    }

    // Adding new image to the index
    addImage(imageId, gImageID, agentN, stableKeyPoints, stableDescs);
    locker_.unlock();
}

void LCDetectorMultiCentralized::filterMatches(std::vector<std::vector<cv::DMatch>> &candidatesToFilter,
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

void LCDetectorMultiCentralized::filterCandidates(
    const std::vector<obindex2::ImageMatch> &image_matches,
    std::vector<obindex2::ImageMatch> *image_matches_filt)
{
    if (image_matches.size() > 0)
    {
        image_matches_filt->clear();
        double max_score = image_matches[0].score;
        double min_score = image_matches[image_matches.size() - 1].score;

        for (unsigned i = 0; i < image_matches.size(); i++)
        {
            // Computing the new score
            double new_score = (image_matches[i].score - min_score) /
                               (max_score - min_score);
            // Assessing if this image match is higher than a threshold
            if (new_score > min_score_)
            {
                obindex2::ImageMatch match = image_matches[i];
                match.score = new_score;
                image_matches_filt->push_back(match);
            }
            else
            {
                break;
            }
        }
    }
}

void LCDetectorMultiCentralized::addImage(const unsigned image_id,
                                          const unsigned globalImgId,
                                          const unsigned agentId,
                                          const std::vector<cv::KeyPoint> &kps,
                                          const cv::Mat &descs)
{
    if (centralOb_->numImages() == 0)
    {
        // This is the first image that is inserted into the index
        centralOb_->addvWords(image_id, globalImgId, agentId, kps, descs);
    }
    else
    {
        // We have to search the descriptor and filter them before adding descs
        // Matching the descriptors
        std::vector<std::vector<cv::DMatch>> matches_feats;

        // Searching the query descriptors against the features
        centralOb_->searchDescriptors(descs, &matches_feats, 2, 64);

        // Filtering matches according to the ratio test
        std::vector<cv::DMatch> matches;
        filterMatches(matches_feats, &matches);

        // Finally, we add the image taking into account the correct matchings
        centralOb_->addvWords(image_id, globalImgId, agentId, kps, descs, matches);
    }
}

bool LCDetectorMultiCentralized::compareByScore(const obindex2::ImageMatch &a,
                                                const obindex2::ImageMatch &b)
{
    return a.score > b.score;
}