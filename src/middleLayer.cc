#include "ibow-lcd/middleLayer.h"

middleLayer::middleLayer(unsigned agents,
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
                         double confProb)
{
    agents_ = agents;
    purge_ = purge;
    p_ = p;
    original_ = original;
    filter_ = filter;

    for (unsigned i = 0; i < agents_; i++)
    {
        obindex2::ImageIndex* I = new obindex2::ImageIndex(16, 150, 4, obindex2::MERGE_POLICY_NONE, purge_);
        agenTrees_.push_back(*I);
    }

    prevDescs_.resize(agents);
    prevKps_.resize(agents);
    currentImagePerAgent_.resize(agents);
    mScore_ = mScore;
    fResult_ = fResult;
    islandSize_ = islandSize;
    minConsecutiveLoops_ = minConsecutiveLoops;
    minInliers_ = minInliers;
    nndrBf_ = nndrBf;
    epDist_ = epDist;
    confProb_ = confProb;
}

void middleLayer::process(std::vector<std::string> &imageFiles)
{
    totalNumberOfImages_ = imageFiles.size();
    unsigned firstImage[agents_];

    if (original_)
    {
        imagesPerAgent_ = totalNumberOfImages_ / agents_;

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
    }
    else
    {
        unsigned iPa[agents_];
        iPa[0] = totalNumberOfImages_ * 0.6;
        iPa[1] = totalNumberOfImages_ * 0.2;
        iPa[2] = totalNumberOfImages_ * 0.2;
        unsigned aux = totalNumberOfImages_ - (iPa[0] + iPa[1] + iPa[2]);
        iPa[2] = iPa[2] + aux;

        aux = 0;
        for (unsigned i = 0; i < agents_; i++)
        {
            firstImage[i] = aux;
            std::vector<std::string> fiAgent;
            for (unsigned f = 0; f < iPa[i]; f++)
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

    globalImagePointer_ = firstImage;

    for (unsigned i = 0; i < filesPerAgent_.size(); i++)
    {
        AgentDistributed *a = new AgentDistributed(&agenTrees_,
                                                   agents_,
                                                   filesPerAgent_[i],
                                                   i,
                                                   firstImage[i],
                                                   &currentImagePerAgent_,
                                                   filter_,
                                                   original_,
                                                   p_,
                                                   &prevKps_,
                                                   &prevDescs_,
                                                   mScore_,
                                                   fResult_,
                                                   islandSize_,
                                                   minConsecutiveLoops_,
                                                   minInliers_,
                                                   nndrBf_,
                                                   epDist_,
                                                   confProb_,
                                                   globalImagePointer_,
                                                   &locker_);

        agentSim_.create_thread(boost::bind(&AgentDistributed::run, a));
    }

    agentSim_.join_all();
}

unsigned middleLayer::getNumberVwords()
{
    unsigned result = 0;
    for (unsigned i = 0; i < agents_; i++)
    {
        result = result + agenTrees_[i].numDescriptors();
    }
    return result;
}
