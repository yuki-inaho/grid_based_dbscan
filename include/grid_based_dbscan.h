#pragma once

#include "header.h"
#include <pcl/filters/conditional_removal.h>
#include <forward_list> 
#include <algorithm>  // std::sort, std::unique
#include <unordered_map>   
#include <map> 
#include <omp.h>
#include <boost/iterator/zip_iterator.hpp>

using namespace std;
using namespace pcl;


class GridBasedDBSCAN{
    public:
    GridBasedDBSCAN(double _epsilon, int _minPts, double _min_x, double _max_x , double _min_y, double _max_y,  double _min_z, double _max_z);
    void setPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud);
    void runConditionalFilter();
    void generateVoxelGridHash();
//    void _generateNearestNeighborIdx(const int64_t &hash_point);    
    void generateNearestNeighborIdx();
    void checkCoreOrNoise();
    void run();    
    void run_test();        
    void getResult(vector<int> &label,  int &cluster_num);
    void getResultVG(vector<int> &label,  int &cluster_num);
    void testVoxelCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vg, std::vector<bool> &is_inlier);    
    void getRandomMap(vector<int> &label);
    
    private:
    double min_x,max_x,min_y,max_y,min_z,max_z;
    double epsilon;
    int minPts;
    int grid_size;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    std::multimap<int64_t, int> map_hash2pointidx;
    std::unordered_map<int64_t, int> hashmap_of_notempty_grid; //for fast decision whether voxel is empty or not
    std::vector<int64_t> hashlist_of_notempty_grid;
    std::multimap<int64_t, int64_t> map_hash2neighbor_idx;
    std::unordered_map<int64_t, bool> hash_isnoise;
    std::unordered_map<int64_t, int> map_hash2pointcount;
    std::unordered_map<int, bool> point_isnoise;
    


    std::unordered_map<int64_t, int64_t> map_hash2label;
    std::unordered_map<int64_t, bool> map_hash2visit;    
    std::unordered_map<int64_t, int> map_hash2index;

    int _threads;
};