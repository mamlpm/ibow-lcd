#include "ibow-lcd/AgentDistributed.h"

AgentDistributed::AgentDistributed(std::vector<obindex2::ImageIndex> *trees,
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
                                   unsigned* globalImagePointer)
{
    agentId_ = agentId;
    nImages_ = flNames.size();
    gImageId_ = firstImageId;
    trees_ = trees;
    currImPAgent_ = currImPAgent;
    filter_ = filter;
    original_ = original;
    consecutiveLoops_ = 0;

    for (unsigned i = 0; i < flNames.size(); i++)
    {
        fileNames_.push_back(flNames[i]);
    }
    minConsecutiveLoops_ = min_consecutive_loops;
    agents_ = agents;
    p_ = p;
    prevKps_ = prevKps;
    prevDescs_ = prevDescs;
    minScore_ = mScore;
    fReslt_ = fReslt;
    islandSize_ = island_size;
    islandOffset_ = island_size / 2;
    minInliers_ = min_inliers;
    nndrBf_ = nndr_bf;
    epDist_ = epDist;
    confProb_ = confProb;
    globalImagePointer_ = globalImagePointer;
}

unsigned AgentDistributed::getId()
{
    return agentId_;
}

void AgentDistributed::run()
{
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1500); // Default params

    for (unsigned j = 0; j < fileNames_.size(); j++)
    {
        currImPAgent_->at(agentId_) = j;
        std::vector<cv::KeyPoint> kpoints;
        cv::Mat importedImage = cv::imread(fileNames_[j]); //import image to read
        cv::Mat descript;

        detector->detect(importedImage, kpoints);            //detect all key points
        cv::KeyPointsFilter::retainBest(kpoints, 1000);      //filter those key points
        detector->compute(importedImage, kpoints, descript); //descript those key points

        //Cheking if there is any previous image
        if (j == 0 || kpoints.size() == 0)
        {
            if (kpoints.size() > 0)
            {
                prevKeyPoints_.clear();         //clean the previous key points vector
                prevKeyPoints_ = kpoints;       //fill the previous key points vector
                previousImage_ = importedImage; //update the last seen image
                prevDescriptors_ = descript;    //Update previous seen descrpitors matrix

                processImage(agentId_, agents_, j, gImageId_, descript, descript, kpoints, kpoints, 0, 1);
            }
            else
            {
                std::cerr << std::endl
                          << "There are no KeyPoints found" << std::endl;
            }
        }
        else
        {
            cv::BFMatcher Mtch(cv::NORM_HAMMING, false);
            std::vector<std::vector<cv::DMatch>> foundMatches;
            Mtch.knnMatch(descript, prevDescriptors_, foundMatches, 2); //Match descriptors from previous
                                                                        //seen image with current seen image ones
            std::vector<cv::DMatch> matchedDescriptors;
            std::vector<cv::KeyPoint> matchedKeyPoints;

            for (unsigned e = 0; e < foundMatches.size(); e++)
            {
                if (foundMatches[e].size() > 1)
                {
                    if (foundMatches[e][0].distance < foundMatches[e][1].distance * 0.8)
                    {
                        matchedDescriptors.push_back(foundMatches[e][0]);
                        matchedKeyPoints.push_back(kpoints[matchedDescriptors[matchedDescriptors.size() - 1].queryIdx]);
                    }
                }
            }

            if (matchedDescriptors.size() > 0)
            {
                cv::Mat foundDescriptors = cv::Mat::zeros(matchedDescriptors.size(), descript.cols, CV_8U);
                for (unsigned i = 0; i < matchedDescriptors.size(); i++)
                {
                    descript.row(matchedDescriptors[i].queryIdx).copyTo(foundDescriptors.row(i));
                }

                if (filter_)
                {
                    std::cout << "---" << std::endl
                              << "Processing filtered images ";
                    if (original_)
                    {
                        std::cout << "(Original algorithm)" << std::endl;
                    }
                    else
                    {
                        std::cout << "(New algorithm)" << std::endl;
                    }
                    processImage(agentId_, agents_, j, gImageId_, descript, foundDescriptors, kpoints, matchedKeyPoints, 1, 1);
                }

                /*****************************************/
            }
            else if (filter_ && !original_)
            {
                processImage(agentId_, agents_, j, gImageId_, descript, descript, kpoints, kpoints, 1, 0);
            }
            if (!filter_)
            {
                std::cout << "---" << std::endl
                          << "Processing non filtered images" << std::endl;
                processImage(agentId_, agents_, j, gImageId_, descript, descript, kpoints, kpoints, 1, 1);
            }
            prevKeyPoints_.clear();         //clean the previous key points vector
            prevKeyPoints_ = kpoints;       //fill the previous key points vector
            previousImage_ = importedImage; //update the last seen image
            prevDescriptors_ = descript;    //Update previous seen descrpitors matrix
        }
        gImageId_++;
    }
}

void AgentDistributed::processImage(unsigned agentN,
                                    unsigned agents,
                                    unsigned imageId,
                                    unsigned gImageID,
                                    const cv::Mat &descs,
                                    const cv::Mat &stableDescs,
                                    std::vector<cv::KeyPoint> keyPoints,
                                    std::vector<cv::KeyPoint> stableKeyPoints,
                                    bool lookForLoop,
                                    bool aImage)
{
    locker_.lock();
    // Searching for loops
    prevDescs_->at(agentN).resize(imageId + 1);
    prevDescs_->at(agentN).at(imageId) = descs;
    prevKps_->at(agentN).resize(imageId + 1);
    prevKps_->at(agentN).at(imageId) = keyPoints;

    AgentDistributedResult rslt;
    rslt.query_id = imageId;
    rslt.QagentId = agentN;
    if (lookForLoop)
    {
        std::vector<obindex2::ImageMatch> iMatchVect;
        for (unsigned agn = 0; agn < agents; agn++)
        {
            std::vector<std::vector<cv::DMatch>> mtchs;
            trees_->at(agn).searchDescriptors(descs,
                                             &mtchs);
            std::vector<cv::DMatch> usableMatches;
            filterMatches(mtchs, &usableMatches);
            std::unordered_map<unsigned, obindex2::ImageMatch> iMatch;

            trees_->at(agn).searchImagesRestrictive(descs, usableMatches, &iMatch, agentN, p_, imageId);
            for (auto it = iMatch.begin(); it != iMatch.end(); it++)
            {
                iMatchVect.push_back(it->second);
            }
        }

        std::vector<obindex2::ImageMatch> iMatchFilt;
        sort(iMatchVect.begin(), iMatchVect.end(), compareByScore);
        filterCandidates(iMatchVect, &iMatchFilt);

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

        bool overlap = 0;

        if (!shouldGetOut)
        {
            ibow_lcd::IslanDistributed island = islands[0];
            std::vector<ibow_lcd::IslanDistributed> p_islands;
            getPriorIslands(lastLcIsland_, islands, &p_islands);

            if (p_islands.size())
            {
                island = p_islands[0];
            }

            lastLcIsland_ = island;
            overlap = p_islands.size() != 0;

            unsigned best_img = island.img_id;
            unsigned bestAgentNum = island.agentId;

            // Assessing the loop
            if (consecutiveLoops_ > minConsecutiveLoops_ && overlap)
            {
                // LOOP can be considered as detected
                rslt.status = LC_DETECTED;
                rslt.train_id = best_img;
                rslt.TagentId = bestAgentNum;
                rslt.inliers = 0;
                // Store the last result
                lastLcResult_ = rslt;
                consecutiveLoops_++;
            }
            else
            {
                // We obtain the image matchings, since we need them for compute F
                std::vector<cv::DMatch> tmatches;
                std::vector<cv::Point2f> tquery;
                std::vector<cv::Point2f> ttrain;

                ratioMatchingBF(descs, prevDescs_->at(bestAgentNum).at(best_img), &tmatches);

                convertPoints(keyPoints, prevKps_->at(bestAgentNum).at(best_img), tmatches, &tquery, &ttrain);

                unsigned inliers = checkEpipolarGeometry(tquery, ttrain);

                if (inliers > minInliers_)
                {
                    // LOOP detected
                    rslt.status = LC_DETECTED;
                    rslt.train_id = best_img;
                    rslt.TagentId = bestAgentNum;
                    rslt.inliers = inliers;
                    // Store the last result
                    lastLcResult_ = rslt;
                    consecutiveLoops_++;
                }
                else
                {
                    rslt.status = LC_NOT_ENOUGH_INLIERS;
                    rslt.train_id = best_img;
                    rslt.TagentId = bestAgentNum;
                    rslt.inliers = inliers;
                    lastLcResult_.status = LC_NOT_ENOUGH_INLIERS;
                    consecutiveLoops_ = 0;
                }
            }
        }

        unsigned globalQueryImage = globalImagePointer_[rslt.QagentId] + rslt.query_id;
        unsigned globalTrainImage = globalImagePointer_[rslt.TagentId] + rslt.train_id;

        switch (rslt.status)
        {
        case LC_NOT_DETECTED:
            std::cout << "No loop closure detected" << std::endl;
            break;
        case LC_NOT_ENOUGH_IMAGES:
            std::cout << "Not enough images to close loop" << std::endl;
            break;
        case LC_NOT_ENOUGH_ISLANDS:
            std::cout << "Not enough islands to close loop" << std::endl;
            break;
        case LC_NOT_ENOUGH_INLIERS:
            std::cout << "Not enough inliers to close a loop" << std::endl;
            break;
        case LC_TRANSITION:
            std::cout << "Currently in a transition" << std::endl;
            break;
        case LC_DETECTED:
            std::cout << "LOOP DETECTED!" << std::endl;
            std::cout << "Query image " << rslt.query_id << " of agent " << rslt.QagentId
                      << " (" << globalQueryImage
                      << ") with train image " << rslt.train_id << " of agent " << rslt.TagentId
                      << " (" << globalTrainImage << ")" << std::endl;
            break;
        }

        std::vector<int> rlt;
        rlt.push_back(static_cast<int>(globalTrainImage));
        rlt.push_back(static_cast<int>(rslt.inliers));
        rlt.push_back(static_cast<int>(overlap));
        rlt.push_back(static_cast<int>(agentN));
        fReslt_->at(globalQueryImage) = rlt;
    }

    // Adding new image to the index
    if (aImage)
    {
        addImage(imageId, gImageID, agentN, stableKeyPoints, stableDescs);
    }
    
    locker_.unlock();
}

void AgentDistributed::filterMatches(std::vector<std::vector<cv::DMatch>> &candidatesToFilter,
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

void AgentDistributed::filterCandidates(
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

void AgentDistributed::addImage(const unsigned image_id,
                                const unsigned globalImgId,
                                const unsigned agentId,
                                const std::vector<cv::KeyPoint> &kps,
                                const cv::Mat &descs)
{
    if (trees_->at(agentId_).numImages() == 0)
    {
        // This is the first image that is inserted into the index
        trees_->at(agentId_).addvWords(image_id, globalImgId, agentId, kps, descs);
    }
    else
    {
        // We have to search the descriptor and filter them before adding descs
        // Matching the descriptors
        std::vector<std::vector<cv::DMatch>> matches_feats;

        // Searching the query descriptors against the features
        trees_->at(agentId_).searchDescriptors(descs, &matches_feats, 2, 64);

        // Filtering matches according to the ratio test
        std::vector<cv::DMatch> matches;
        filterMatches(matches_feats, &matches);

        // Finally, we add the image taking into account the correct matchings
        trees_->at(agentId_).addvWords(image_id, globalImgId, agentId, kps, descs, matches);
    }
}

bool AgentDistributed::compareByScore(const obindex2::ImageMatch &a,
                                      const obindex2::ImageMatch &b)
{
    return a.score > b.score;
}

void AgentDistributed::buildIslands(const std::vector<obindex2::ImageMatch> &image_matches,
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
        unsigned max_id = static_cast<unsigned>(std::min(curr_img_id + islandOffset_, currImPAgent_->at(nAgent) - 1));

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
}

void AgentDistributed::getPriorIslands(const ibow_lcd::IslanDistributed &island,
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

void AgentDistributed::ratioMatchingBF(const cv::Mat &query,
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
    matches->clear();
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Matching descriptors
    std::vector<std::vector<cv::DMatch>> matches12;

    matcher.knnMatch(query, train, matches12, 2);

    //std::cout << "Matches found using brute force: " << matches12.size() << std::endl << std::endl;

    // Filtering the resulting matchings according to the given ratio
    //std::cout << nndrBf_ << std::endl;
    for (unsigned m = 0; m < matches12.size(); m++)
    {
        if (matches12[m][0].distance <= matches12[m][1].distance * nndrBf_)
        {
            //std::cout << "Hola caracola" << std::endl;
            matches->push_back(matches12[m][0]);
        }
    }
}

void AgentDistributed::convertPoints(const std::vector<cv::KeyPoint> &query_kps,
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

unsigned AgentDistributed::checkEpipolarGeometry(const std::vector<cv::Point2f> &query,
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