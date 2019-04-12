#include "ibow-lcd/LCDetectorMultiCentralized.h"

LCDetectorMultiCentralized::LCDetectorMultiCentralized(unsigned agents,
                                                       obindex2::ImageIndex *centralOb,
                                                       unsigned p,
                                                       double mScore)
{
    agents_ = agents;
    centralOb_ = centralOb;
    p_ = p;
    min_score_ = mScore;
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
        Agent *a = new Agent(this, filesPerAgent_[i], i, firstImage[i]);
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
    locker_.lock();
    std::cout << "This is central server "
              << "and now I should be processing agent's " << agentN << " "
              << imageId << " which is the " << gImageID << " processed image" << std::endl;
    if (lookForLoop)
    {
        std::vector<std::vector<cv::DMatch>> mtchs;
        centralOb_->searchDescriptors(descs,
                                      &mtchs);
        std::vector<cv::DMatch> usableMatches;
        std::cout << "Hola " << std::endl;
        filterMatches(mtchs, &usableMatches);
        std::unordered_map<unsigned, obindex2::ImageMatch> iMatch;
        centralOb_->searchImagesRestrictive(descs, usableMatches, &iMatch, agentN, p_, imageId);
        std::vector<obindex2::ImageMatch> iMatchVect;
        for (auto it = iMatch.begin(); it != iMatch.end(); it++)
        {
            iMatchVect.push_back(it->second);
        }
        sort(iMatchVect.begin(), iMatchVect.end(), compareByScore);
        std::vector<obindex2::ImageMatch> iMatchFilt;
        filterCandidates(iMatchVect, &iMatchFilt);
    }
    //Llamar metodo addImage (local)
    locker_.unlock();
}

void LCDetectorMultiCentralized::filterMatches(std::vector<std::vector<cv::DMatch>> &candidatesToFilter,
                                               std::vector<cv::DMatch> *filteredCandidates)
{
    // std::cout << "Hola 1" << std::endl;
    filteredCandidates->clear();
    // std::cout << "Hola 2" << std::endl;
    for (unsigned i = 0; i < candidatesToFilter.size(); i++)
    {
        // std::cout << " size: " << candidatesToFilter.size() <<  std::endl;

        // std::cout << "Hola 3" << std::endl;
        if (candidatesToFilter[i][0].distance < 0.8 * candidatesToFilter[i][1].distance)
        {
            // std::cout << "Hola 4" << std::endl;
            filteredCandidates->push_back(candidatesToFilter[i][0]);
        }
    }
    // std::cout << "Hola 5" << std::endl;
}

void LCDetectorMultiCentralized::filterCandidates(
    const std::vector<obindex2::ImageMatch> &image_matches,
    std::vector<obindex2::ImageMatch> *image_matches_filt)
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

bool LCDetectorMultiCentralized::compareByScore(const obindex2::ImageMatch &a,
                                                const obindex2::ImageMatch &b)
{
    return a.score > b.score;
}

void LCDetectorMultiCentralized::addImage(const unsigned image_id,
                                          const unsigned globalImgId,
                                          const unsigned agentId,
                                          const std::vector<cv::KeyPoint> &kps,
                                          const cv::Mat &descs,
                                          const std::vector<cv::DMatch> mtchs)
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