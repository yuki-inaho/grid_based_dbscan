#include "header.h"
#include <forward_list> 
#include <algorithm>  // std::sort, std::unique

using namespace std;
using namespace pcl;
vector<int>
dbscan(pcl::PointCloud<PointXYZ>::Ptr cloud, double epsilon, int minPts){
    vector<bool> label_arrival;
    vector<int> label_cluster;    
    int n_pts = cloud->points.size();
    for(int i=0;i<n_pts;i++){
        label_arrival.push_back(false);
        label_cluster.push_back(0);        
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    
    for(int i=0;i<n_pts;i++){

        pcl::PointXYZ point_i = cloud->points[i];
        label_cluster[i]==i;
        
        if ( kdtree.radiusSearch (point_i, epsilon, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0){
            if(pointIdxRadiusSearch.size() > minPts){
                std::forward_list<int> pointIdx_deq;
                for(auto it = pointIdxRadiusSearch.begin();it!=pointIdxRadiusSearch.end();++it){
                    pointIdx_deq.push_front(*it);
                }

                int count=0;
                for(auto it=pointIdx_deq.begin(), it_end=pointIdx_deq.end(); it != it_end; ++it){
                    if(!label_arrival[*it]){
                        label_arrival[*it] = true;
                        if(label_cluster[*it]==0){
                            label_cluster[*it] = i;
                        }
                        std::vector<int> _pointIdxRadiusSearch;
                        std::vector<float> _pointRadiusSquaredDistance;
                        if ( kdtree.radiusSearch (cloud->points[*it], epsilon, _pointIdxRadiusSearch, _pointRadiusSquaredDistance) > 0){
                            if(_pointIdxRadiusSearch.size() > minPts){
                                for(int k=0;k<_pointIdxRadiusSearch.size();k++){
                                    pointIdx_deq.insert_after(it, _pointIdxRadiusSearch[k]);
                                    it_end=pointIdx_deq.end();
                                }
                            }
                        }
                    }else{

                    }

                }
            }else{
                label_cluster[i] = -1;
            }
        }
    }

    return label_cluster;
}
