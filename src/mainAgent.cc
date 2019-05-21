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
#include "ibow-lcd/AgentDistributed.h"
#include "ibow-lcd/middleLayer.h"

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
  unsigned step;
  double mScore;
  std::string dataSetName;
  unsigned islandSize;
  int minConsecutiveLoops;
  unsigned minInliers;
  float nndrBf;
  double epDist;
  double confProb;
  bool filter;
  bool original;
  bool purge;
  bool distributed;

  if (true)
  {
    int tAgents;
    int tislandSize;
    int tminInliers;
    int tP;
    int tFilter;
    int tOriginal;
    int tPurge;
    int tStep;
    int tdistributed;

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
    n.param<int>("filter", tFilter, 1);
    n.param<int>("original", tOriginal, 1);
    n.param<int>("purge", tPurge, 1);
    n.param<int>("paso", tStep, 1);
    n.param<int>("distributed", tdistributed, 0);

    filter = static_cast<bool>(tFilter);
    original = static_cast<bool>(tOriginal);
    purge = static_cast<bool>(tPurge);
    distributed = static_cast<bool>(tdistributed);

    step = static_cast<unsigned>(tStep);
    p = static_cast<unsigned>(tP);
    agents = static_cast<unsigned>(tAgents);
    islandSize = static_cast<unsigned>(tislandSize);
    minInliers = static_cast<unsigned>(tminInliers);
  }

  std::string datasetPad;

  if (dataSetName == "Lip6In")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/Lip6_indoor/images";
  }
  else if (dataSetName == "Lip6Out")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/Lip6_outdoor/images";
  }
  else if (dataSetName == "CityCentre")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/CityCentre/images";
  }
  else if (dataSetName == "KITTI00")
  {
    datasetPad = "/home/mamlpm/Documentos/TrabajoFinMaster/datasets/KITTI/00/images";
  }
  else
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
            << "Step -> " << step << std::endl
            << "Filter? -> " << filter << std::endl
            << "Purge? -> " << purge << std::endl
            << "Params imported" << std::endl;

  std::cout << "Importing files..." << std::endl;
  std::vector<std::string> filenames; //import images
  getFilenames(datasetPad, &filenames);

  std::string folderName;
  if (purge)
  {
    if (filter && original)
    {
      folderName = "/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtrados/";
    }
    else if (filter && !original)
    {
      folderName = "/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtradosNuevo/";
    }
    else
    {
      folderName = "/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/noFiltrados/";
    }
  }
  else
  {
    if (filter && original)
    {
      folderName = "/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtrados/";
    }
    else if (filter && !original)
    {
      folderName = "/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtradosNuevo/";
    }
    else
    {
      folderName = "/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/noFiltrados/";
    }
  }

  std::cout << "Files imported." << std::endl;
  if (agents > filenames.size() || agents == 0 || step > agents)
  {
    std::cout << "You should check the number of declared agents" << std::endl;
    return 0;
  }

  unsigned nVisualWords[(agents / step) + 1];
  double timeExecution[(agents / step) + 1];
  unsigned agentsN[(agents / step) + 1];
  unsigned countAux = 0;

  if (distributed)
  {
    boost::filesystem::path res_dir = folderName + "distributed/" + dataSetName;
    boost::filesystem::remove_all(res_dir);
    boost::filesystem::create_directory(res_dir);

    for (unsigned i = 0; i <= agents; i += step)
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

      std::cout << "Initiallizing agents manager..." << std::endl;

      unsigned agnts;
      if (countAux == 0)
      {
        agnts = 1;
      }
      else
      {
        agnts = i;
      }

      middleLayer MD(agnts, purge, filter, original, p, mScore, &fResult, islandSize, minConsecutiveLoops, minInliers, nndrBf, epDist, confProb);
      std::cout << "Initiallizing agents..." << std::endl;

      auto start = std::chrono::steady_clock::now();
      MD.process(filenames);
      auto end = std::chrono::steady_clock::now();

      auto diff = end - start;
      timeExecution[countAux] = std::chrono::duration<double, std::milli>(diff).count();

      //nVisualWords[countAux] = centralOb.numDescriptors();
      agentsN[countAux] = agnts;

      std::cout << "---" << std::endl;
      std::cout << "All images processed with " << i << " agent(s)" << std::endl;

      std::cout << "Storing results to a file..." << std::endl
                << std::endl;

      std::string pathName = folderName + dataSetName + "/" + std::to_string(agnts) + ".txt";
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
      countAux++;
    }
  }
  else
  {
    boost::filesystem::path res_dir = folderName + "noDistributed/" + dataSetName;
    boost::filesystem::remove_all(res_dir);
    boost::filesystem::create_directory(res_dir);

    for (unsigned i = 0; i <= agents; i += step)
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
      obindex2::ImageIndex centralOb(16, 150, 4, obindex2::MERGE_POLICY_NONE, purge);

      std::cout << "Initiallizing central agents manager..." << std::endl;

      unsigned agnts;
      if (countAux == 0)
      {
        agnts = 1;
      }
      else
      {
        agnts = i;
      }

      LCDetectorMultiCentralized LCM(agnts, &centralOb, p, mScore, &fResult, islandSize, minConsecutiveLoops, minInliers, nndrBf, epDist, confProb, filter, original);
      std::cout << "Initiallizing agents..." << std::endl;

      auto start = std::chrono::steady_clock::now();
      LCM.process(filenames);
      auto end = std::chrono::steady_clock::now();

      auto diff = end - start;
      timeExecution[countAux] = std::chrono::duration<double, std::milli>(diff).count();

      nVisualWords[countAux] = centralOb.numDescriptors();
      agentsN[countAux] = agnts;

      std::cout << "---" << std::endl;
      std::cout << "All images processed with " << i << " agent(s)" << std::endl;

      std::cout << "Storing results to a file..." << std::endl
                << std::endl;

      std::string pathName = folderName + dataSetName + "/" + std::to_string(agnts) + ".txt";
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
      countAux++;
    }
  }
  std::string pathName;
  if (distributed)
  {
    pathName = folderName + "distributed/" + dataSetName + "/" + "resultadosExtra.txt";
  }else
  {
    pathName =  folderName + "noDistributed/" + dataSetName + "/" + "resultadosExtra.txt";
  }
  
  char outputFileName[500];
  sprintf(outputFileName, "%s", pathName.c_str());
  std::ofstream outputFile(outputFileName);

  for (unsigned i = 0; i < (1 + (agents / step)); i++)
  {
    outputFile << agentsN[i] << "\t";
    outputFile << nVisualWords[i] << "\t";
    outputFile << timeExecution[i] << "\t";
    outputFile << std::endl;
  }
  outputFile.close();

  return 0;
}