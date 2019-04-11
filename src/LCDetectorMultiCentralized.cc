#include "ibow-lcd/LCDetectorMultiCentralized.h"

LCDetectorMultiCentralized::LCDetectorMultiCentralized(unsigned agents,
                                                       obindex2::ImageIndex *centralOb,
                                                       unsigned p)
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
    filterMatches(mtchs, &usableMatches);
    std::unordered_map<unsigned, obindex2::ImageMatch> iMatch;
    centralOb_->searchImagesRestrictive(descs, usableMatches, &iMatch, agentN, 0, p_);
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
    const std::unordered_map<unsigned, obindex2::ImageMatch> &image_matches,
    std::unordered_map<unsigned, obindex2::ImageMatch> *image_matches_filt)
{
    image_matches_filt->clear();

    std::vector <obindex2::ImageMatch> aux;
    std::vector <double> aux1;
    for (auto it = image_matches.begin(); it != image_matches.end(); it++)
    {
        aux.push_back(it->second);
        aux1.push_back(aux[aux.size()-1].score);
    }

    auto aux2 = std::max_element(aux1.begin(), aux1.end());
    double max_score = *aux2;

    aux2 = std::min_element(aux1.begin(), aux1.end());
    double min_score = *aux2;

    for (auto i = image_matches.begin(); i != image_matches.end(); i++)
    {
        obindex2::ImageMatch match = i -> second;
        // Computing the new score
        double new_score = (match.score - min_score) /
                           (max_score - min_score);
        // Assessing if this image match is higher than a threshold
        if (new_score > min_score_)
        {
            //obindex2::ImageMatch match = image_matches[i];
            // match.score = new_score;
            // obindex2::ImageMatch aux3 = i -> second;
            // match.image_id = aux3.image_id;
            // match.score = aux3.score;
            image_matches_filt->insert({match.image_id, match});
        }
        else
        {
            break;
        }
    }
}