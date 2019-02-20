#include "grid_based_dbscan.h"

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

using namespace std;
using namespace pcl;

#pragma omp declare reduction (merge : std::vector<boost::tuple<int64_t,int64_t>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge : std::vector<boost::tuple<int,int64_t>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

GridBasedDBSCAN::GridBasedDBSCAN(double _epsilon, int _minPts, double _min_x, double _max_x ,
                                 double _min_y, double _max_y,  double _min_z, double _max_z)
{
    epsilon = _epsilon;
    minPts = _minPts;
    min_x = _min_x;
    max_x = _max_x;
    min_y = _min_y;
    max_y = _max_y;
    min_z = _min_z;
    max_z = _max_z;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud (new pcl::PointCloud<pcl::PointXYZ>);   
    cloud = _cloud;
    _threads = omp_get_max_threads();
    cout << _threads << endl;
};

void
GridBasedDBSCAN::setPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud)
{
    pcl::copyPointCloud(*_cloud, *cloud);
}

void
GridBasedDBSCAN::runConditionalFilter()
{
   pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointXYZ> ());
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::GT, min_x)));
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::LT, max_x)));    
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::GT, min_y)));
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::LT, max_y)));    
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, min_z)));
   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr 
    (new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, max_z)));    

    pcl::ConditionalRemoval<pcl::PointXYZ> condrem;

    condrem.setCondition (range_cond);
    condrem.setInputCloud (cloud);
    condrem.setKeepOrganized(true);
    // apply filter
    condrem.filter (*cloud);
}


void
GridBasedDBSCAN::generateVoxelGridHash()
{
    hashmap_of_notempty_grid.reserve(cloud->points.size());
    hashlist_of_notempty_grid.reserve(cloud->points.size());
    grid_size = std::floor((max_x-min_x)/epsilon); //assume (max_x-min_x) == (max_y-min_y) == (max_z-min_z)

    std::vector<boost::tuple<int, int64_t>> index_hash_tuple_list;
    index_hash_tuple_list.resize(cloud->points.size());

    #pragma omp parallel for shared(index_hash_tuple_list) num_threads(_threads) 
    for(int i=0;i<cloud->points.size();i++){
        double x = cloud->points[i].x - min_x ; 
        double y = cloud->points[i].y - min_y ;
        double z = cloud->points[i].z - min_z ;
        if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)) 
        {
            index_hash_tuple_list[i] = boost::make_tuple(i,-1);
            continue;
        }
        if(x > 1.0|| y>1.0 || z>1.0 ||x < 0.0 || y < 0.0 || z < 0.0 )
        {
            index_hash_tuple_list[i] = boost::make_tuple(i,-1); 
            continue;
        }
        int64_t x_int = std::floor(x/epsilon);
        int64_t y_int = std::floor(y/epsilon);
        int64_t z_int = std::floor(z/epsilon);
        int64_t hash = x_int + y_int*grid_size + z_int*grid_size*grid_size;
        //map_hash2pointidx.insert(std::pair<int64_t, int>(hash, i));
        //std::unordered_map<int64_t, int>::iterator it = hashmap_of_notempty_grid.find(hash);
        index_hash_tuple_list[i] = boost::make_tuple(i,hash);         
    }

    for(int idx=0;idx<index_hash_tuple_list.size();idx++){
        boost::tuple<int,int64_t> index_hash_tuple = index_hash_tuple_list[idx];
        int i = index_hash_tuple.get<0>();
        int64_t hash = index_hash_tuple.get<1>();
        map_hash2pointidx.insert(std::pair<int64_t, int>(hash, i));
        std::unordered_map<int64_t, int>::iterator it = hashmap_of_notempty_grid.find(hash);
        if(it == hashmap_of_notempty_grid.end()){
            hashmap_of_notempty_grid.insert(std::pair<int64_t, int>(hash, i));
            hashlist_of_notempty_grid.push_back(hash);            
            map_hash2pointcount.insert(std::pair<int64_t, int>(hash, 1));
        }else{
            ///auto  it = map_hash2pointcount.find(hash);
            map_hash2pointcount.at(hash) = map_hash2pointcount.at(hash) + 1;
        }
    }
}

/*
void
GridBasedDBSCAN::_generateNearestNeighborIdx(const int64_t &hash_point)
{

}
*/

void
GridBasedDBSCAN::generateNearestNeighborIdx()
{
    int64_t kernel_size = 3;
    std::vector<boost::tuple<int64_t,int64_t,int64_t>> dxyz_tuple_list;
    for(int64_t dz=-kernel_size;dz<=kernel_size;dz++){
        for(int64_t dy=-kernel_size;dy<=kernel_size;dy++){
            for(int64_t dx=-kernel_size;dx<=kernel_size;dx++){
                dxyz_tuple_list.push_back(boost::make_tuple(dx,dy,dz));
            }
        }
    }

    vector<int64_t> hash_list;
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){
        hash_list.push_back(*it);
    }

    vector<boost::tuple<int64_t,int64_t>> hash_tuple_list; 
    hash_tuple_list.reserve(hash_list.size()*kernel_size*kernel_size*kernel_size);

    #pragma omp parallel for reduction(merge: hash_tuple_list) num_threads(_threads) 
    for(int it_idx=0;it_idx<hash_list.size();it_idx++){
        //int64_t hash_point = *it;
        int64_t hash_point = hash_list[it_idx];
        int64_t z_int = hash_point / (grid_size*grid_size);
        int64_t y_int = (hash_point- z_int*grid_size*grid_size) / grid_size;
        int64_t x_int = (hash_point- z_int*grid_size*grid_size -y_int*grid_size);

        for(int i=0;i<dxyz_tuple_list.size();i++){
            boost::tuple<int64_t,int64_t,int64_t> dxyz_tuple = dxyz_tuple_list[i];
            int64_t dx = dxyz_tuple.get<0>();
            int64_t dy = dxyz_tuple.get<1>();
            int64_t dz = dxyz_tuple.get<2>();

            if(dz==0 && dy==0 && dx==0){
                continue;
            }
            int64_t x_int_kernel = x_int + dx;
            int64_t y_int_kernel = y_int + dy;
            int64_t z_int_kernel = z_int + dz;

            if(x_int_kernel <0 || x_int_kernel >=grid_size)continue;
            if(y_int_kernel <0 || y_int_kernel >=grid_size)continue;
            if(z_int_kernel <0 || z_int_kernel >=grid_size)continue;
            int64_t hash_kernel = x_int_kernel + y_int_kernel*grid_size + z_int_kernel*grid_size*grid_size;
            auto _it = hashmap_of_notempty_grid.find(hash_kernel); 
            if(_it != hashmap_of_notempty_grid.end()){
    //            #pragma omp critical
    //            map_hash2neighbor_idx.insert(std::pair<std::int64_t, std::int64_t>(hash_point, hash_kernel));
                hash_tuple_list.push_back(boost::tuple<std::int64_t, std::int64_t>(hash_point, hash_kernel));    
            }
        }
    }

    for(int j=0;j<hash_tuple_list.size();j++){
        boost::tuple<std::int64_t, std::int64_t> hash_elem = hash_tuple_list[j];
        int64_t hash_point = hash_elem.get<0>();
        int64_t hash_kernel = hash_elem.get<1>();
        map_hash2neighbor_idx.insert(std::pair<std::int64_t, std::int64_t>(hash_point, hash_kernel));
    }
}

void
GridBasedDBSCAN::checkCoreOrNoise()
{
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){
        pcl::PointXYZ point;
        int64_t hash = *it;
        int count =0;

        std::pair<multimap<int64_t, int64_t>::iterator, multimap<int64_t, int64_t>::iterator> range = map_hash2neighbor_idx.equal_range(hash);
        auto it_neighbor = range.first;
        if(it_neighbor == range.second){
            hash_isnoise.insert(std::pair<int64_t, bool>(hash, true));
        }else{
            int count =map_hash2pointcount.at(hash);
            for(; it_neighbor != range.second; ++it_neighbor){
                count += map_hash2pointcount.at(it_neighbor->second);
            }
            if(count >= minPts){
                hash_isnoise.insert(std::pair<int64_t, bool>(hash, false)); //not noise
            }else{
                hash_isnoise.insert(std::pair<int64_t, bool>(hash, true));
            }
        }

    }
}

void 
GridBasedDBSCAN::testVoxelCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vg, vector<bool> &is_inlier)
{
    int count = 0;

    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        pcl::PointXYZ point;
        int64_t hash = *it;
        int64_t z_int = hash / (grid_size*grid_size);
        int64_t y_int = (hash- z_int*grid_size*grid_size) / grid_size;
        int64_t x_int = (hash- z_int*grid_size*grid_size -y_int*grid_size);
        point.x = double(x_int)*epsilon + min_x + epsilon/2;
        point.y = double(y_int)*epsilon + min_y + epsilon/2;
        point.z = double(z_int)*epsilon + min_z + epsilon/2;
        double x = point.x - min_x ; 
        double y = point.y - min_y ;
        double z = point.z - min_z ;
        if(x > 1.0|| y>1.0 || z>1.0 ||x < 0.0 || y < 0.0 || z < 0.0 ){
            continue;
        }
        cloud_vg->points.push_back(point);
        is_inlier.push_back(hash_isnoise.at(hash));
        map_hash2index.insert(std::pair<int64_t,int>(hash,count));
        count ++;
    }
}

void 
GridBasedDBSCAN::run()
{
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        map_hash2label.insert(std::pair<int64_t, int64_t>(*it, -2));
        map_hash2visit.insert(std::pair<int64_t, bool>(*it, false));
    }

    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        std::forward_list<int64_t> neighbor_list;

        if(map_hash2visit.at(*it))continue;
        map_hash2visit.at(*it) = true;
        if(hash_isnoise.at(*it)){
            map_hash2label.at(*it) = -1;
            continue;
        }else{
            map_hash2label.at(*it) = *it;
        }

        if(map_hash2neighbor_idx.count(*it)==0){
            map_hash2label.at(*it) = -1;
            continue;
        }

        std::pair<multimap<int64_t, int64_t>::iterator, multimap<int64_t, int64_t>::iterator> range = map_hash2neighbor_idx.equal_range(*it);
        auto it_neighbor = range.first;
        for(; it_neighbor != range.second; ++it_neighbor){
            neighbor_list.push_front(it_neighbor->second) ;
        }

        for(auto it_nb_list=neighbor_list.begin(), it_nb_list_end=neighbor_list.end();
                     it_nb_list!=it_nb_list_end; ++it_nb_list)
        {
            if(!map_hash2visit.at(*it_nb_list)){
                map_hash2visit.at(*it_nb_list) = true;

                std::pair<multimap<int64_t, int64_t>::iterator, multimap<int64_t, int64_t>::iterator> range_nn
                        = map_hash2neighbor_idx.equal_range(*it_nb_list);
                auto it_neighbor_neighbor = range_nn.first;
                for(; it_neighbor_neighbor != range_nn.second; ++it_neighbor_neighbor){
                    if(!hash_isnoise.at(*it)){
                        neighbor_list.insert_after(it_nb_list, it_neighbor_neighbor->second);                        
                    }
                }

                it_nb_list_end=neighbor_list.end(); // update of iterator
            }

            if(map_hash2label.at(*it_nb_list) <0) { //noise or unlabelled
                map_hash2label.at(*it_nb_list) = *it;
            }
        }
    }
}

void 
GridBasedDBSCAN::run_test()
{
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        map_hash2label.insert(std::pair<int64_t, int64_t>(*it, -2));
        map_hash2visit.insert(std::pair<int64_t, bool>(*it, false));
    }

    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        std::forward_list<int64_t> neighbor_list;
        
        if(map_hash2visit.at(*it))continue;
        map_hash2visit.at(*it) = true;
        map_hash2label.at(*it) = *it;

        std::pair<multimap<int64_t, int64_t>::iterator, multimap<int64_t, int64_t>::iterator> range = map_hash2neighbor_idx.equal_range(*it);
        auto it_neighbor = range.first;
        for(; it_neighbor != range.second; ++it_neighbor){
            neighbor_list.push_front(it_neighbor->second) ;
        }

        for(auto it_nb_list=neighbor_list.begin(), it_nb_list_end=neighbor_list.end();
                     it_nb_list!=it_nb_list_end; ++it_nb_list)
        {
            if(!map_hash2visit.at(*it_nb_list)){
                map_hash2visit.at(*it_nb_list) = true;

                std::pair<multimap<int64_t, int64_t>::iterator, multimap<int64_t, int64_t>::iterator> range_nn
                        = map_hash2neighbor_idx.equal_range(*it_nb_list);
                auto it_neighbor_neighbor = range_nn.first;
                for(; it_neighbor_neighbor != range_nn.second; ++it_neighbor_neighbor){
                    neighbor_list.insert_after(it_nb_list, it_neighbor_neighbor->second);                        
                }
                it_nb_list_end=neighbor_list.end(); // update of iterator
            }

            if(map_hash2label.at(*it_nb_list) <0) { //noise or unlabelled
                map_hash2label.at(*it_nb_list) = *it;
            }
        }
    }
}


void 
GridBasedDBSCAN::getResult(vector<int> &label,  int &cluster_num)
{
    std::vector<int64_t> label_unique;
    std::unordered_map<int64_t, int> map_label_unique;
    label_unique.reserve(hashlist_of_notempty_grid.size());
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){        
        label_unique.push_back(map_hash2label.at(*it));
    }
    std::sort(label_unique.begin(), label_unique.end() );
    label_unique.erase( unique(label_unique.begin(), label_unique.end() ), label_unique.end() );
    
    cluster_num = label_unique.size();
    for(int k=0; k<cluster_num; k++){
        //cout << label_unique[k] << endl;
        map_label_unique.insert(std::pair<int64_t, int>(label_unique[k], k));
    }

    for(int i=0;i<cloud->points.size();i++){
        double x = cloud->points[i].x - min_x ; 
        double y = cloud->points[i].y - min_y ;
        double z = cloud->points[i].z - min_z ;
        if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)){
            label.push_back(0);
            continue;
        }
        if(x > 1.0|| y>1.0 || z>1.0 ||x < 0.0 || y < 0.0 || z < 0.0 ){
            label.push_back(0);
            continue;
        }

        int64_t x_int = std::floor(x/epsilon);
        int64_t y_int = std::floor(y/epsilon);
        int64_t z_int = std::floor(z/epsilon);
        int64_t hash = x_int + y_int*grid_size + z_int*grid_size*grid_size;
        int64_t label_hash = map_hash2label.at(hash);
        int cluster_label = map_label_unique.at(label_hash);
        label.push_back(cluster_label);
    }
}


void 
GridBasedDBSCAN::getResultVG(vector<int> &label,  int &cluster_num)
{
    std::vector<int64_t> label_unique;
    std::unordered_map<int64_t, int> map_label_unique;
    label_unique.reserve(hashlist_of_notempty_grid.size());
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){        
        label_unique.push_back(map_hash2label.at(*it));
    }
    std::sort(label_unique.begin(), label_unique.end() );
    label_unique.erase( unique(label_unique.begin(), label_unique.end() ), label_unique.end() );
    
    cluster_num = label_unique.size();
    for(int k=0; k<cluster_num; k++){
        cout << label_unique[k] << endl;
        map_label_unique.insert(std::pair<int64_t, int>(label_unique[k], k));
    }

    vector<int> cluster_count;
    for(int k=0; k<cluster_num; k++){
        cluster_count.push_back(0);
    }

    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        pcl::PointXYZ point;
        int64_t hash = *it;
        int64_t label_hash = map_hash2label.at(hash);
        int cluster_label = map_label_unique.at(label_hash);
        cluster_count[cluster_label] ++;
        label.push_back(cluster_label);
    }

    cout << "cluster count" << endl;
    for(int k=0; k<cluster_num; k++){
        if(cluster_count[k] > 10){
            cout << cluster_count[k] << endl;
        }
    }

}

void 
GridBasedDBSCAN::getRandomMap(vector<int> &label)
{
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        label.push_back(0);
    }

    int count = 0;
    for(auto it=hashlist_of_notempty_grid.begin(), it_end=hashlist_of_notempty_grid.end(); it!=it_end;++it){    
        int64_t hash = *it;
        int i = map_hash2index.at(hash);

        if(count == 502){
            label[i] = 1;
            std::pair<multimap<int64_t, int64_t>::iterator, multimap<int64_t, int64_t>::iterator> range = map_hash2neighbor_idx.equal_range(hash);
            for (auto it_neighbor = range.first; it_neighbor != range.second; ++it_neighbor) {
                label[map_hash2index.at(it_neighbor->second)] = 1;
            }
            
        }else{
            label[i] = 0;
        }
        count ++;
    }
}

