#include <chrono>
#include <cstdio>
#include <iostream>

#include <boost/filesystem.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "obindex2/binary_index.h"
#include "ibow-lcd/LCDetectorMultiCentralized.h"
#include "ibow-lcd/Agent.h"

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/chrono.hpp>
#include <unistd.h>

void getFilenames(const std::string &directory,
                  std::vector<std::string> *filenames)
{
  using namespace boost::filesystem;

  filenames->clear();
  path dir(directory);

  // Retrieving, sorting and filtering filenames.
  std::vector<path> entries;
  copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
  sort(entries.begin(), entries.end());
  for (auto it = entries.begin(); it != entries.end(); it++)
  {
    std::string ext = it->extension().c_str();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".png" || ext == ".jpg" ||
        ext == ".ppm" || ext == ".jpeg")
    {
      filenames->push_back(it->string());
    }
  }
}

int main(int argc, char **argv)
{

  std::cout << "Importing files..." << std::endl;
  std::vector<std::string> filenames; //import images
  getFilenames(argv[1], &filenames);
  unsigned agents = 6;

  /******************Previous Code****************************/
  // unsigned nImages = filenames.size();
  // unsigned imagesPerAgent = nImages / agents;
  /***********************************************************/

  std::cout << "Files imported." << std::endl;
  if (agents > filenames.size() || agents == 0)
  {
    std::cout << "You should check the number of declared agents" << std::endl;
    return 0;
  }
  std::cout << "Total number of images to import " << filenames.size() << std::endl;
  obindex2::ImageIndex centralOb(16, 150, 4, obindex2::MERGE_POLICY_NONE, true);
  std::cout << "Initiallizing central agents manager..." << std::endl;
  LCDetectorMultiCentralized LCM(agents, &centralOb, 10);
  std::cout << "Initialllizing agents..." << std::endl;
  LCM.process(filenames);

  /************************Previous Code************************/
  // unsigned aux = 0;
  // std::vector<boost::thread *> agentObjects;
  // std::vector<std::pair<unsigned, cv::Mat>> outP;
  // // std::vector<cv::Mat> outP;
  // boost::mutex locker;
  // for (unsigned i = 0; i < agents; i++)
  // {
  //   std::vector<std::string> fiAgent;
  //   for (unsigned f = 0; f < imagesPerAgent; f++)
  //   {
  //     fiAgent.push_back(filenames[aux]);
  //     aux++;
  //   }

  //   if (i == agents - 1)
  //   {
  //     while (aux < nImages)
  //     {
  //       fiAgent.push_back(filenames[aux]);
  //       aux++;
  //     }
  //   }

  //   std::cout << "I'm agent " << i << " and I have to process " << fiAgent.size() << " Images" << std::endl;

  //   Agent *a = new Agent(&centralOb, fiAgent, &locker, &outP, i);
  //   boost::thread *trd = new boost::thread(&Agent::run, a);
  //   agentObjects.push_back(trd);
  // }

  // for (unsigned i = 0; i < agentObjects.size(); i++)
  // {
  //   agentObjects[i]->join();
  // }
  /*******************Uncomment to display results*******************/
  // for (unsigned i = 0; i < agentObjects.size(); i++)
  // {
  //   cv::namedWindow(std::to_string(i), cv::WINDOW_AUTOSIZE);
  // }

  // std::cout << "Displaying results..." << std::endl;
  // for (unsigned i = 0; i < outP.size(); i++)
  // {
  //   cv::imshow(std::to_string(outP[i].first), outP[i].second);
  //   cv::waitKey(5);
  //   // usleep(5000000);
  //   usleep(100000);
  // }
  std::cout << "All images processed" << std::endl;
  // cv::destroyAllWindows();
  return 0;
}