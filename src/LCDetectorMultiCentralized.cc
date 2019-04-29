#include "ibow-lcd/LCDetectorMultiCentralized.h"

LCDetectorMultiCentralized::LCDetectorMultiCentralized(unsigned agents,
                                                       obindex2::ImageIndex *centralOb,
                                                       unsigned p,
                                                       double mScore,
                                                       std::unordered_map<unsigned, std::vector<std::pair<unsigned, obindex2::ImageMatch>>> *fReslt,
                                                       unsigned island_size,
                                                       int min_consecutive_loops,
                                                       unsigned min_inliers,
                                                       unsigned nndr_bf,
                                                       double epDist,
                                                       double confProb)
{
    agents_ = agents;
    centralOb_ = centralOb;
    p_ = p;
    minScore_ = mScore;
    fReslt_ = fReslt;
    islandSize_ = island_size;
    islandOffset_ = island_size / 2;
    currentImagePerAgent_.resize(agents);
    lastLcIsland_.resize(agents);
    prevDescs_.resize(agents);
    prevKps_.resize(agents);
    consecutiveLoops_ = 0;
    minConsecutiveLoops_ = min_consecutive_loops;
    minInliers_ = min_inliers;
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

    // std::cout << "Argument data: " << descs.rows << " | " << descs.cols << std::endl;
    prevDescs_[agentN].resize(imageId + 1);
    prevDescs_[agentN][imageId] = descs;
    // std::cout << descs.rows << " | " << descs.cols << std::endl;
    // std::cout << "HAgent " << agentN << " " << imageId << " | " << prevDescs_[agentN].size() << std::endl;
    prevKps_[agentN].resize(imageId + 1);
    prevKps_[agentN][imageId] = keyPoints;

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
            lastLcResult_.status = LC_NOT_ENOUGH_ISLANDS;
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

                // std::cout << "---" << std::endl;

                // std::cout << "Prior island important number " << p_islands[0].img_id << " limits " << p_islands[0].min_img_id << " | "
                //           << p_islands[0].max_img_id << " agent " << p_islands[0].agentId << std::endl;

                // std::cout << "Last loop closure island important number " << lastLcIsland_[agentN].img_id << " limits "
                //           << lastLcIsland_[agentN].min_img_id << " | " << lastLcIsland_[agentN].max_img_id << " agent "
                //           << lastLcIsland_[agentN].agentId << std::endl;
            }
            lastLcIsland_[agentN] = island;
            bool overlap = p_islands.size() != 0;

            unsigned best_img = island.img_id;
            unsigned bestAgentNum = island.agentId;
            // std::cout << "---" << std::endl
            //           << "Input data: " << prevDescs_[bestAgentNum][best_img].rows << " | "
            //           << prevDescs_[bestAgentNum][best_img].cols << std::endl
            //           << "Agent Nº: " << bestAgentNum << std::endl
            //           << "Best image Nº: " << best_img << std::endl
            //           << "Vector size: " << prevDescs_.size() << std::endl
            //           << "Vector of vector size: " << prevDescs_[bestAgentNum].size() << std::endl
            //           << "Number of images processed by this agent: " << imageId << std::endl
            //           << "---" << std::endl;
            // Assessing the loop
            if (consecutiveLoops_ > minConsecutiveLoops_ && overlap)
            {
                // std::cout << "Hola 1" << std::endl;
                // LOOP can be considered as detected
                rslt.status = LC_DETECTED;
                rslt.train_id = best_img;
                rslt.inliers = 0;
                // Store the last result
                lastLcResult_ = rslt;
                consecutiveLoops_++;
            }
            else
            {
                // std::cout << "Hola 2" << std::endl;
                // We obtain the image matchings, since we need them for compute F
                std::vector<cv::DMatch> tmatches;
                std::vector<cv::Point2f> tquery;
                std::vector<cv::Point2f> ttrain;
                ratioMatchingBF(descs, prevDescs_[bestAgentNum][best_img], &tmatches);
                // std::cout << "Hola 2.0" << std::endl;
                convertPoints(keyPoints, prevKps_[bestAgentNum][best_img], tmatches, &tquery, &ttrain);
                // std::cout << "Hola 2.1" << std::endl;
                unsigned inliers = checkEpipolarGeometry(tquery, ttrain);
                // std::cout << "Hola 2.2" << std::endl;

                if (inliers > minInliers_)
                {
                    // std::cout << "Hola 3" << std::endl;
                    // LOOP detected
                    rslt.status = LC_DETECTED;
                    rslt.train_id = best_img;
                    rslt.inliers = inliers;
                    // Store the last result
                    lastLcResult_ = rslt;
                    consecutiveLoops_++;
                }
                else
                {
                    // std::cout << "Hola 4" << std::endl;
                    rslt.status = LC_NOT_ENOUGH_INLIERS;
                    rslt.train_id = best_img;
                    rslt.inliers = inliers;
                    lastLcResult_.status = LC_NOT_ENOUGH_INLIERS;
                    consecutiveLoops_ = 0;
                }
            }
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
            if (new_score > minScore_)
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
        unsigned min_id = static_cast<unsigned>(std::max((int)curr_img_id - (int)islandOffset_,
                                                         0));
        unsigned max_id = static_cast<unsigned>(std::min(curr_img_id + islandOffset_, currentImagePerAgent_[nAgent] - 1));

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
    // std::cout << "---" << std::endl;
    // std::cout << islands->size() << " islands have been created" << std::endl;
    // for (unsigned a = 0; a < islands->size(); a++)
    // {
    //     std::cout << "Agent " << islands->at(a).agentId << " new island important number: " << islands->at(a).img_id
    //               << " limits: " << islands->at(a).min_img_id << " | " << islands->at(a).max_img_id << " with score: "
    //               << islands->at(a).score << std::endl;
    // }
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

void LCDetectorMultiCentralized::ratioMatchingBF(const cv::Mat &query,
                                                 const cv::Mat &train,
                                                 std::vector<cv::DMatch> *matches)
{
    // if (query.rows <= 0 || query.cols <= 0 || train.rows <= 0 || train.cols <= 0)
    // {
    //     std::cout << "Getting out from here "
    //               << "Query: " << query.rows << " | " << query.cols << std::endl
    //               << "Train: " << train.rows << " | " << train.cols << std::endl;
    //     return;
    // }
    // std::cout << "Hola 2.0.1" << std::endl;
    matches->clear();
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Matching descriptors
    std::vector<std::vector<cv::DMatch>> matches12;
    // std::cout << "Hola 2.0.2" << std::endl;

    // std::cout << "Query: " << query.rows << " | " << query.cols << std::endl;
    // std::cout << "Train: " << train.rows << " | " << train.cols << std::endl;

    matcher.knnMatch(query, train, matches12, 2);
    // std::cout << "Hola 2.0.3" << std::endl;

    // Filtering the resulting matchings according to the given ratio
    for (unsigned m = 0; m < matches12.size(); m++)
    {
        if (matches12[m][0].distance <= matches12[m][1].distance * nndrBf_)
        {
            matches->push_back(matches12[m][0]);
        }
    }
}

void LCDetectorMultiCentralized::convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                                               const std::vector<cv::KeyPoint> &train_kps,
                                               const std::vector<cv::DMatch> &matches,
                                               std::vector<cv::Point2f> *query,
                                               std::vector<cv::Point2f> *train)
{
    query->clear();
    train->clear();
    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        // Get the position of query keypoints
        float x = query_kps[it->queryIdx].pt.x;
        float y = query_kps[it->queryIdx].pt.y;
        query->push_back(cv::Point2f(x, y));

        // Get the position of train keypoints
        x = train_kps[it->trainIdx].pt.x;
        y = train_kps[it->trainIdx].pt.y;
        train->push_back(cv::Point2f(x, y));
    }
}

unsigned LCDetectorMultiCentralized::checkEpipolarGeometry(const std::vector<cv::Point2f> &query,
                                                           const std::vector<cv::Point2f> &train)
{
    std::vector<uchar> inliers(query.size(), 0);
    if (query.size() > 7)
    {
        cv::Mat F =
            cv::findFundamentalMat(
                cv::Mat(query), cv::Mat(train), // Matching points
                CV_FM_RANSAC,                   // RANSAC method
                epDist_,                        // Distance to epipolar line
                confProb_,                      // Confidence probability
                inliers);                       // Match status (inlier or outlier)
    }

    // Extract the surviving (inliers) matches
    auto it = inliers.begin();
    unsigned total_inliers = 0;
    for (; it != inliers.end(); it++)
    {
        if (*it)
            total_inliers++;
    }

    return total_inliers;
}