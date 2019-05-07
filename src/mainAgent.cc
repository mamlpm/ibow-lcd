#include <ros/ros.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sys/stat.h>
#include <fstream>

#include <boost/filesystem.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "obindex2/binary_index.h"
#include "ibow-lcd/LCDetectorMultiCentralized.h"
#include "ibow-lcd/Agent.h"

#include <unordered_map>
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
  ros::init(argc, argv, "ibow-lcd-multi");
  ros::NodeHandle n("~");
  unsigned agents;
  unsigned p;
  double mScore;
  std::string dataSetName;
  unsigned islandSize;
  int minConsecutiveLoops;
  unsigned minInliers;
  float nndrBf;
  double epDist;
  double confProb;
  if (true)
  {
    int tAgents;
    int tislandSize;
    int tminInliers;
    int tP;

    n.param<int>("numberAgents", tAgents, 3);
    n.param<int>("p", tP, 10);
    n.param<double>("mScore", mScore, 0.3);
    n.param<std::string>("dataset", dataSetName, "Lip6In");
    n.param<int>("islandSize", tislandSize, 7);
    n.param<int>("minConsecutiveLoops", minConsecutiveLoops, 5);
    n.param<int>("minInliers", tminInliers, 22);
    n.param<float>("nndrBf", nndrBf, 0.8);
    n.param<double>("epDist", epDist, 2);
    n.param<double>("confProb", confProb, 0.985);

    p = static_cast<unsigned>(tP);
    agents = static_cast<unsigned>(tAgents);
    islandSize = static_cast<unsigned>(tislandSize);
    minInliers = static_cast<unsigned>(tminInliers);
  }

  std::string datasetPad;

  if (dataSetName == "Lip6In")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/Lip6_indoor/images";
  }else if (dataSetName == "Lip6Out")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/Lip6_outdoor/images";
  }else if (dataSetName == "CityCentre")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/CityCentre/images";
  }else if (dataSetName == "KITTI00")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/KITTI/00/images";
  }else
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/KITTI/05/images";
  }

  std::cout << "Importing Parameters..." << std::endl
            << "Max number of agents to process -> " << agents << std::endl
            << "P -> " << p << std::endl
            << "Minimum score -> " << mScore << std::endl
            << "Data set name -> " << dataSetName << std::endl
            << "Island size -> " << islandSize << std::endl
            << "Minimum consecutive loops -> " << minConsecutiveLoops << std::endl
            << "Minimum number of inliers -> " << minInliers << std::endl
            << "nndrBf -> " << nndrBf << std::endl
            << "Epipolar distance -> " << epDist << std::endl
            << "confidence probability -> " << confProb << std::endl
            << "Params imported" << std::endl;

  std::cout << "Importing files..." << std::endl;
  std::vector<std::string> filenames; //import images
  getFilenames(datasetPad, &filenames);

  std::string folderName = "/home/mamlpm/Documentos/TrabajoFinMaster/Results/";

  boost::filesystem::path res_dir = folderName + dataSetName;
  boost::filesystem::remove_all(res_dir);
  boost::filesystem::create_directory(res_dir);

  std::cout << "Files imported." << std::endl;
  if (agents > filenames.size() || agents == 0)
  {
    std::cout << "You should check the number of declared agents" << std::endl;
    return 0;
  }
  for (unsigned i = 1; i <= agents; i++)
  {
    std::vector<std::vector<int>> fResult;
    fResult.resize(filenames.size());
    std::vector<int> aux;
    aux.push_back(-1);
    aux.push_back(-1);
    aux.push_back(-1);
    aux.push_back(-1);
    for (unsigned j = 0; j < fResult.size(); j++)
    {
      fResult.at(j) = aux;
    }

    std::cout << "Total number of images to import " << filenames.size() << std::endl;
    obindex2::ImageIndex centralOb(16, 150, 4, obindex2::MERGE_POLICY_NONE, true);

    std::cout << "Initiallizing central agents manager..." << std::endl;
    LCDetectorMultiCentralized LCM(i, &centralOb, p, mScore, &fResult, islandSize, minConsecutiveLoops, minInliers, nndrBf, epDist, confProb);

    std::cout << "Initialllizing agents..." << std::endl;
    LCM.process(filenames);

    std::cout << "---" << std::endl;
    std::cout << "All images processed with " << i << " agent(s)" << std::endl;

    std::cout << "Storing results to a file..." << std::endl
              << std::endl;

    std::string pathName = folderName + dataSetName + "/" + std::to_string(i) + ".txt";
    char outputFileName[500];
    sprintf(outputFileName, "%s", pathName.c_str());
    std::ofstream outputFile(outputFileName);

    for (unsigned j = 0; j < fResult.size(); j++)
    {
      //std::cout << j << " | " << fResult[j].size() << std::endl;
      outputFile << fResult[j][0] << "\t";
      outputFile << fResult[j][1] << "\t";
      outputFile << fResult[j][2] << "\t";
      outputFile << fResult[j][3] << "\t";
      outputFile << std::endl;
    }
    outputFile.close();
  }
  //std::unordered_map<unsigned, std::vector<std::pair<unsigned, obindex2::ImageMatch>>> fResult;

  // std::cout << fResult.size() << "---"  << std::endl;

  // unsigned nImages = filenames.size() / agents;
  // unsigned ii = 0;
  // unsigned jj = 0;
  // for (unsigned i = 0; i < fResult.size(); i++)
  // {
  //   cv::namedWindow(std::to_string(i), cv::WINDOW_AUTOSIZE);
  //   cv::namedWindow(std::to_string(i) + "." + std::to_string(i), cv::WINDOW_AUTOSIZE);
  //   for (unsigned j = 0; j < fResult[i].size(); j++)
  //   {
  //     if (fResult[ii][jj].first == j)
  //     {
  //       unsigned imageTorepresent = i * nImages + j;
  //       unsigned imageTocompare = fResult[ii][jj].second.agentId * nImages + fResult[ii][jj].second.image_id;

  //       std::cout << "Image " << imageTorepresent << " could close " << imageTocompare << std::endl
  //                 << "---" << std::endl;
  //       cv::Mat imgTreps = cv::imread(filenames[imageTorepresent]);
  //       cv::Mat imgTocmp = cv::imread(filenames[imageTocompare]);
  //       imshow(std::to_string(i), imgTreps);
  //       cv::waitKey(5);
  //       imshow(std::to_string(i) + "." + std::to_string(i), imgTocmp);
  //       cv::waitKey(0);
  //       jj++;
  //     }
  //     else
  //     {
  //       std::cout << "Image " << i * nImages + j << " unable to match" << std::endl
  //                 << "---" << std::endl;
  //     }
  //   }
  //   ii++;
  //   cv::destroyAllWindows();
  // }

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

  // cv::destroyAllWindows();
  return 0;
}