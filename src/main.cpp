#include <stdio.h>
#include <iostream>

#include <pcl/visualization/pcl_visualizer.h>
#include "header.h"
#include "grid_based_dbscan.h"
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>

using namespace std;
using namespace pcl;

//removeNan: NaN要素を点群データから除去するメソッド
//input : target(NaN要素を除去する対象の点群)
//output: cloud(除去を行った点群)
PointCloud<PointXYZ>::Ptr removeNan(PointCloud<PointXYZ>::Ptr target){
  PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  int n_point = target->points.size();
  for(int i=0;i<n_point; i++){
    PointXYZ tmp_point;
    if(std::isfinite(target->points[i].x) || std::isfinite(target->points[i].y) || std::isfinite(target->points[i].z)){
      tmp_point.x = target->points[i].x;
      tmp_point.y = target->points[i].y;
      tmp_point.z = target->points[i].z;
      cloud->points.push_back(tmp_point);
    }
  }
//  cout << "varid points:" << cloud->points.size() << endl;
  return cloud;
}

void
addRGBtoPointCloudWithBool(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::vector<bool> &flaglist, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_color){
  int n_point = cloud->points.size();

  for(int i=0; i<n_point ; i++){
    pcl::PointXYZRGB point;
    point.x = cloud->points[i].x;
    point.y = cloud->points[i].y;
    point.z = cloud->points[i].z;
    if(flaglist[i]){
        point.r = 255;
        point.g = 0;
        point.b = 0;
    }else{
        point.r = 0;
        point.g = 255;
        point.b = 0;
    }
    cloud_color->points.push_back(point);
  }
}

void hsv2rgb(const unsigned char &src_h, const unsigned char &src_s, const unsigned char &src_v, unsigned char &dst_r, unsigned char &dst_g, unsigned char &dst_b)
{
    float h = src_h *   2.0f; // 0-360
    float s = src_s / 255.0f; // 0.0-1.0
    float v = src_v / 255.0f; // 0.0-1.0

    float r, g, b; // 0.0-1.0

    int   hi = (int)(h / 60.0f) % 6;
    float f  = (h / 60.0f) - hi;
    float p  = v * (1.0f - s);
    float q  = v * (1.0f - s * f);
    float t  = v * (1.0f - s * (1.0f - f));

    switch(hi) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    dst_r = (unsigned char)(r * 255); // dst_r : 0-255
    dst_g = (unsigned char)(g * 255); // dst_r : 0-255
    dst_b = (unsigned char)(b * 255); // dst_r : 0-255
}

void
addRGBtoPointCloudWithLabel(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::vector<int> &labellist, const int &cluster_num, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_color){
  int n_point = cloud->points.size();
  for(int i=0; i<n_point ; i++){
    pcl::PointXYZRGB point;
    point.x = cloud->points[i].x;
    point.y = cloud->points[i].y;
    point.z = cloud->points[i].z;

    if(labellist[i] == 0){
        point.r = 255;
        point.g = 0;
        point.b = 0;
    }else{
        int cluster_i = labellist[i];
        unsigned char h = std::floor(double(cluster_i)/double(cluster_num) * 180.0);
        unsigned char s = 255;
        unsigned char v = 255;
        unsigned char r,g,b;
        hsv2rgb(h,255,255,r,g,b);
        point.r = r;
        point.g = g;
        point.b = b;
    }
    cloud_color->points.push_back(point);
  }
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> 
VisInit ()
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (0.01);
  viewer->initCameraParameters ();
  return viewer;
}



int main()
{    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile ("../data/test_data.pcd", *cloud);
    cloud = removeNan(cloud);
    
   pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointXYZ> ());
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, 0)));
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, 1.2)));    
    pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
    condrem.setCondition (range_cond);
    condrem.setInputCloud (cloud);
    condrem.setKeepOrganized(true);
    condrem.filter (*cloud);
    cloud = removeNan(cloud);    

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vg (new pcl::PointCloud<pcl::PointXYZ>);
    vector<bool> is_inlier;
    std::chrono::system_clock::time_point  start, end; // 型は auto で可

    start = std::chrono::system_clock::now(); // 計測開始時間                
    GridBasedDBSCAN dbs(0.01, 10, -1.0, 1.0, -1.0, 1.0, 0.0, 1.5);
    dbs.setPointCloud(cloud);
    dbs.runConditionalFilter();
    dbs.generateVoxelGridHash();
    dbs.generateNearestNeighborIdx();
    dbs.checkCoreOrNoise();
    dbs.testVoxelCentroid(cloud_vg, is_inlier);
    dbs.run();
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << "elapsed:" << elapsed << endl;
    vector<int> label;
    int num_cluster;
    dbs.getResult(label,num_cluster);    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color (new pcl::PointCloud<pcl::PointXYZRGB>);
    addRGBtoPointCloudWithLabel(cloud, label, num_cluster, cloud_color);    
  
/*
std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator(16);
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (30);
  normal_estimator.compute (*normals);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (30.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (0.5);
  std::vector <pcl::PointIndices> clusters;

  reg.extract (clusters);    
  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  cout << "elapsed:" << elapsed << endl;
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud_color = reg.getColoredCloud ();
*/

  

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = VisInit();
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> inlier_rgb(cloud_color);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud_color, inlier_rgb, "inlier");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inlier");

//    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered);    
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

  return 0;
}