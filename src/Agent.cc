#include "ibow-lcd/Agent.h"

Agent::Agent(LCDetectorMultiCentralized *centralCerv,
             std::vector<std::string> &flNames,
             boost::mutex *locker,
             unsigned agentId,
             unsigned firstImageId)
{
    centr_ = centralCerv;
    agentId_ = agentId;
    nImages_ = flNames.size();
    locker_ = locker;
    gImageId_ = firstImageId;
    for (unsigned i = 0; i < flNames.size(); i++)
    {
        fileNames_.push_back(flNames[i]);
    }
}

unsigned Agent::getId()
{
    return agentId_;
}

void Agent::run()
{
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(); //create a features detector
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> descriptor =
        cv::xfeatures2d::BriefDescriptorExtractor::create(); //create a descriptor extractor
    for (unsigned j = 0; j < fileNames_.size(); j++)
    {

        std::vector<cv::KeyPoint> kpoints;
        cv::Mat importedImage = cv::imread(fileNames_[j]); //import image to read
        locker_->lock();
        std::cout << "This is agent " << agentId_ << " processing image " << fileNames_[j] << std::endl;
        locker_->unlock();
        cv::Mat descript;

        detector->detect(importedImage, kpoints);              //detect all key points
        cv::KeyPointsFilter::retainBest(kpoints, 1000);        //filter those key points
        descriptor->compute(importedImage, kpoints, descript); //descript those key points

        //Cheking if there is any previous image
        if (j == 0 || kpoints.size() == 0)
        {
            if (kpoints.size() > 0)
            {
                prevKeyPoints_.clear();         //clean the previous key points vector
                prevKeyPoints_ = kpoints;       //fill the previous key points vector
                previousImage_ = importedImage; //update the last seen image
                prevDescriptors_ = descript;    //Update previous seen descrpitors matrix
                
                std::vector<cv::KeyPoint> matchedKeyPoints;
                locker_->lock();
                centr_->processImage(agentId_, j, gImageId_, descript, kpoints, matchedKeyPoints, 0);
                locker_->unlock();

            
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

                /*****************Uncomment to display results ***********************/
                // locker_->lock();
                // cv::drawMatches(importedImage, kpoints, previousImage_, prevKeyPoints_, matchedDescriptors, outP);
                // outPut_->push_back(std::make_pair(agentId_, outP));
                // locker_->unlock();
                /***************************************/

                /*****************************************/
                locker_->lock();
                centr_->processImage(agentId_, j, gImageId_, descript, kpoints, matchedKeyPoints, 1);
                locker_->unlock();

                /*****************************************/
            }

            prevKeyPoints_.clear();         //clean the previous key points vector
            prevKeyPoints_ = kpoints;       //fill the previous key points vector
            previousImage_ = importedImage; //update the last seen image
            prevDescriptors_ = descript;    //Update previous seen descrpitors matrix
        }
        gImageId_++;
    }
}