#include "ibow-lcd/LCDetectorMultiCentralized.h"

LCDetectorMultiCentralized::LCDetectorMultiCentralized(unsigned agents,
                                                       obindex2::ImageIndex *centralOb,
                                                       unsigned p,
                                                       double mScore,
                                                       std::unordered_map<unsigned, std::vector<std::pair<unsigned, obindex2::ImageMatch>>> *fReslt,
                                                       unsigned island_size,
                                                       int min_consecutive_loops)
{
    agents_ = agents;
    centralOb_ = centralOb;
    p_ = p;
    min_score_ = mScore;
    fReslt_ = fReslt;
    island_size_ = island_size;
    island_offset_ = island_size / 2;
    currentImagePerAgent_.resize(agents);
    lastLcIsland_.resize(agents);
    consecutive_loops_ = 0;
    min_consecutive_loops_ = min_consecutive_loops;
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
        Agent *a = new Agent(this, filesPerAgent_[i], i, firstImage[i], fReslt_, &currentImagePerAgent_);
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
    //std::cout << "Processing image " << imageId << " seen by agent " << agentN << std::endl;
    LCDetectorMultiCentralizedResult rslt;
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
        // if (iMatchVect.size() > 2)
        // {
        //     std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[0].agentId << " image " << iMatchVect[0].image_id << " (" << iMatchVect[0].agentId * imagesPerAgent_ + iMatchVect[0].image_id << ") scoring " << iMatchVect[0].score << std::endl;
        //     std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[1].agentId << " image " << iMatchVect[1].image_id << " (" << iMatchVect[1].agentId * imagesPerAgent_ + iMatchVect[1].image_id << ") scoring " << iMatchVect[1].score << std::endl;
        //     std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[2].agentId << " image " << iMatchVect[2].image_id << " (" << iMatchVect[2].agentId * imagesPerAgent_ + iMatchVect[2].image_id << ") scoring " << iMatchVect[2].score << std::endl;
        // }
        // else
        // {
        //     for (unsigned i = 0; i < iMatchVect.size(); i++)
        //     {
        //         std::cout << "Agent " << agentN << " image " << imageId << " (" << agentN * imagesPerAgent_ + imageId << ") with agent " << iMatchVect[i].agentId << " image " << iMatchVect[i].image_id << " (" << iMatchVect[i].agentId * imagesPerAgent_ + iMatchVect[i].image_id << ") scoring " << iMatchVect[i].score << std::endl;
        //     }
        // }
        // std::cout << "---" << std::endl;

        std::vector<obindex2::ImageMatch> iMatchFilt;
        filterCandidates(iMatchVect, &iMatchFilt);
        if (iMatchVect.size() > 0)
        {
            result->resize(result->size() + 1);
            result->at(result->size() - 1).first = imageId;
            result->at(result->size() - 1).second = iMatchVect[0];
        }

        std::vector<ibow_lcd::IslanDistributed> islands;
        buildIslands(iMatchFilt, &islands);

        bool shouldGetOut = 0;
        if (!islands.size())
        {
            // No resulting islands
            rslt.status = LC_NOT_ENOUGH_ISLANDS;
            rslt.train_id = 0;
            rslt.inliers = 0;
            last_lc_result_.status = LC_NOT_ENOUGH_ISLANDS;
            shouldGetOut = 1;
        }

        if (!shouldGetOut)
        {
            ibow_lcd::IslanDistributed island = islands[0];
            std::vector<ibow_lcd::IslanDistributed> p_islands;
            getPriorIslands(lastLcIsland_[agentN], islands, &p_islands);
            if (p_islands.size())
            {
                island = p_islands[0];

                std::cout << "---" << std::endl;
                std::cout << "Prior island important number " << p_islands[0].img_id << " limits " << p_islands[0].min_img_id << " | "
                          << p_islands[0].max_img_id << " agent " << p_islands[0].agentId << std::endl;
                std::cout << "Last loop closure island important number " << lastLcIsland_[agentN].img_id << " limits "
                          << lastLcIsland_[agentN].min_img_id << " | " << lastLcIsland_[agentN].max_img_id << " agent "
                          << lastLcIsland_[agentN].agentId << std::endl;
            }
            lastLcIsland_[agentN] = island;
            bool overlap = p_islands.size() != 0;

            unsigned best_img = island.img_id;

            // // Assessing the loop
            // if (consecutive_loops_ > min_consecutive_loops_ && overlap)
            // {
            //     // LOOP can be considered as detected
            //     rslt.status = LC_DETECTED;
            //     rslt.train_id = best_img;
            //     rslt.inliers = 0;
            //     // Store the last result
            //     last_lc_result_ = *rslt;
            //     consecutive_loops_++;
            // }
            // else
            // {
            //     // We obtain the image matchings, since we need them for compute F
            //     std::vector<cv::DMatch> tmatches;
            //     std::vector<cv::Point2f> tquery;
            //     std::vector<cv::Point2f> ttrain;
            //     ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
            //     convertPoints(kps, prev_kps_[best_img], tmatches, &tquery, &ttrain);
            //     unsigned inliers = checkEpipolarGeometry(tquery, ttrain);

            //     if (inliers > min_inliers_)
            //     {
            //         // LOOP detected
            //         rslt.status = LC_DETECTED;
            //         rslt.train_id = best_img;
            //         rslt.inliers = inliers;
            //         // Store the last result
            //         last_lc_result_ = *rslt;
            //         consecutive_loops_++;
            //     }
            //     else
            //     {
            //         rslt.status = LC_NOT_ENOUGH_INLIERS;
            //         rslt.train_id = best_img;
            //         rslt.inliers = inliers;
            //         last_lc_result_.status = LC_NOT_ENOUGH_INLIERS;
            //         consecutive_loops_ = 0;
            //     }
            // }
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

void LCDetectorMultiCentralized::buildIslands(const std::vector<obindex2::ImageMatch> &image_matches,
                                              std::vector<ibow_lcd::IslanDistributed> *islands)
{
    islands->clear();

    // We process each of the resulting image matchings
    for (unsigned i = 0; i < image_matches.size(); i++)
    {
        // Getting information about this match
        unsigned curr_img_id = static_cast<unsigned>(image_matches[i].image_id);
        unsigned nAgent = static_cast<unsigned>(image_matches[i].agentId);
        double curr_score = image_matches[i].score;

        // Theoretical island limits
        unsigned min_id = static_cast<unsigned>(std::max((int)curr_img_id - (int)island_offset_,
                                                         0));
        unsigned max_id = static_cast<unsigned>(std::min(curr_img_id + island_offset_, currentImagePerAgent_[nAgent] - 1));

        // We search for the closest island
        bool found = false;
        for (unsigned j = 0; j < islands->size(); j++)
        {
            if (islands->at(j).fits(curr_img_id, nAgent))
            {
                islands->at(j).incrementScore(curr_score);
                found = true;
                break;
            }
            else
            {
                // We adjust the limits of a future island
                if (islands->at(j).agentId == nAgent)
                {
                    islands->at(j).adjustLimits(curr_img_id, &min_id, &max_id);
                }
            }
        }

        // Creating a new island if required
        if (!found)
        {
            ibow_lcd::IslanDistributed new_island(curr_img_id,
                                                  curr_score,
                                                  min_id,
                                                  max_id,
                                                  nAgent);
            islands->push_back(new_island);
        }
    }

    // Normalizing the final scores according to the number of images
    for (unsigned j = 0; j < islands->size(); j++)
    {
        islands->at(j).normalizeScore();
    }

    std::sort(islands->begin(), islands->end());
    std::cout << "---" << std::endl;
    std::cout << islands->size() << " islands have been created" << std::endl;
    for (unsigned a = 0; a < islands->size(); a++)
    {
        std::cout << "Agent " << islands->at(a).agentId << " new island important number: " << islands->at(a).img_id
                  << " limits: " << islands->at(a).min_img_id << " | " << islands->at(a).max_img_id << " with score: "
                  << islands->at(a).score << std::endl;
    }
}

void LCDetectorMultiCentralized::getPriorIslands(const ibow_lcd::IslanDistributed &island,
                                                 const std::vector<ibow_lcd::IslanDistributed> &islands,
                                                 std::vector<ibow_lcd::IslanDistributed> *p_islands)
{
    p_islands->clear();

    // We search for overlapping islands
    for (unsigned i = 0; i < islands.size(); i++)
    {
        ibow_lcd::IslanDistributed tisl = islands[i];
        if (island.overlaps(tisl))
        {
            p_islands->push_back(tisl);
        }
    }
}