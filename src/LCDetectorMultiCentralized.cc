#include "ibow-lcd/LCDetectorMultiCentralized.h"

LCDetectorMultiCentralized::LCDetectorMultiCentralized(unsigned agents,
                                 std::vector<std::string> &imageFiles,
                                 obindex2::ImageIndex *centralOb)
{

    totalNumberOfImages_ = imageFiles.size();
    agents_ = agents;
    imagesPerAgent_ = totalNumberOfImages_ / agents_;
    unsigned aux = 0;
    centralOb_ = centralOb;

    for (unsigned i = 0; i < agents_; i++)
    {
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
}

void LCDetectorMultiCentralized::process()
{

    // for (unsigned i = 0; i < agents_; i++)
    // {
    //     cv::namedWindow(std::to_string(i), cv::WINDOW_AUTOSIZE);
    // }

    for (unsigned i = 0; i < filesPerAgent_.size(); i++)
    {
        Agent *a = new Agent(this, filesPerAgent_[i], &locker_, &outP_, i);
        boost::thread *trd = new boost::thread(&Agent::run, a);
        agentObjects_.push_back(trd);
    }

    // unsigned a = 1;
    // while (a < totalNumberOfImages_)
    // {
    //     locker_.lock();
    //     std::cout << a << std::endl;
    //     a = a + displayImages();
    //     locker_.unlock();
    // }

    for (unsigned i = 0; i < agentObjects_.size(); i++)
    {
        agentObjects_[i]->join();
    }
}

unsigned LCDetectorMultiCentralized::displayImages()
{
    unsigned e = 1;
    for (unsigned i = 0; i < outP_.size(); i++)
    {
        cv::imshow(std::to_string(outP_[i].first), outP_[i].second);
        cv::waitKey(5);
        usleep(100000);
        // e++;
        // std::cout << e << std::endl;        
    }
    e = outP_.size();
    outP_.clear();
    return e;
}