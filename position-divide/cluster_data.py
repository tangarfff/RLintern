import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import os.path as osp
from sklearn.cluster import KMeans
import sys
#matplotlib inline
#config InlineBackend.figure_format = 'retina'
print(plt.style.available)
# 起始点
lc_start_x = 116.59681754
lc_start_y = 40.12989378
def read_file(path):
    """将坐标写入"""
    #file = osp.join(sys.path[0] + '/clean_test_{}_PM.csv'.format(datestr))
    file = osp.join(path)
    weights_list,lat_x,lon_y = [],[],[]
    lat_x.append(lc_start_x)
    lon_y.append(lc_start_y)
    weights_list.append(0)
    for line in open(file):
        content = line.strip('\n').split(',')
        if "VOL" not in content[2]:
            weights_list.append(float(content[2]))  # add weight
            lat_x.append(float(content[6]))
            lon_y.append(float(content[5]))
    node_num = len(lat_x)
    """在图上显示数据点和仓库"""
    data = pd.DataFrame({"x": lat_x, "y": lon_y}, index=[i for i in range(node_num)])
    plt.scatter(data.x, data.y, c='r', marker='o', s=2)
    plt.scatter(data.iloc[0].x, data.iloc[0].y, marker='*', c='r', s=40)
    # plt.scatter(lat_x, lon_y, c='b', marker='o', s=2)
    # plt.scatter(lc_start_x, lc_start_y, marker='*', c='r', s=40)
    # plt.axis([min(lat_x),max(lat_x),min(lon_y),max(lon_y)])
    plt.xlim(min(lat_x) - 0.05, max(lat_x) + 0.05)
    plt.ylim(min(lon_y) - 0.05, max(lon_y) + 0.05)
    plt.grid(True)
    return data,weights_list,lat_x,lon_y,node_num

#plt.show()
"""计算距离矩阵"""
#distance_func = lambda a,b: np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)#欧式距离
#distance_func = lambda ax,bx,ay,by: np.abs(ax-bx)+np.abs(ay-by)#曼哈顿距离
distance_func = lambda a,b: np.abs(a.x-b.x)+np.abs (a.y-b.y) #曼哈顿距离
"""def compute_distances(dfunc=distance_func):
    distance_matrix = np.zeros((node_num,node_num))
    for i in range(node_num):
        for j in range(i + 1, node_num):
            if i != j:
                distance_matrix[i][j] = distance_matrix[j][i] = dfunc(lat_x[i],lat_x[j],lon_y[i],lon_y[j])
    return distance_matrix"""
def compute_distances(dfunc=distance_func):
    for i in range(node_num):
        current=data.iloc[i]
        data[i]=dfunc(current, data)
    return data

#print(distance_matrix)
"""nearest_neighbor粗暴求解"""
"""def nearest_neighbor(unserved, stop_condition=lambda route, target: False):
    current=0 #depot
    result_path=[]
    while True:
        result_path.append(current)
        unserved.remove(current)
        if not unserved:
            break

        current=data.iloc[unserved,current+2].idxmin()
        if stop_condition(result_path, int(current)):
            if len(result_path)>1:
                break

    result_path.append(0)
    return result_path
estimate_cost = lambda route: sum(data.iloc[i][j] for i,j in zip(route, route[1:]))
get_coords = lambda route: list(zip(*[(data.iloc[i].x,data.iloc[i].y) for i in route]))
route = nearest_neighbor(list(range(node_num)))
print('cost={}\nroute={}'.format(estimate_cost(route),route))"""
"""聚类"""

#plt.show()
def get_max_clster_weight(clusters_result):
    node_sum_weight = []
    for i in clusters_result:
        node_sum_weight.append(sum(weights_list[n] for n in i))
    return max(node_sum_weight)

def add_node_to_data(node_list):
    data2_x_list,data2_y_list = [],[]
    for i in node_list:
        data2_x_list.append(lat_x[i])
        data2_y_list.append(lon_y[i])

    return data2_x_list,data2_y_list

def get_clusters_result(path):
    data, weights_list, lat_x, lon_y, node_num = read_file(path)
    distance_matrix = compute_distances()
    work_per_van = 18  # how many deliveries each van will start, ie granularity; a better approach would be capacity.
    model = KMeans(n_clusters=node_num // work_per_van + 1, init='k-means++', random_state=0)
    model.fit(data[['x', 'y']])
    clusters_result = []
    clusters_center = []
    for i in range(model.n_clusters):
        unserved = data[model.labels_ == i].index.tolist()  # 类中的node
        clusters_result.append(unserved)
        clusters_center.append(model.cluster_centers_[i])

    while get_max_clster_weight(clusters_result)>work_per_van:
        #print("get_max_clster_weight(model)",get_max_clster_weight(model))
        for c in range(len(clusters_result)):
            if sum(weights_list[n] for n in clusters_result[c])>18:
                data2_x_list, data2_y_list = add_node_to_data(clusters_result[c])
                data2 = pd.DataFrame({"x": data2_x_list, "y": data2_y_list}, index=[i for i in clusters_result[c]])
                if len(clusters_result[c]) // work_per_van + 1 == 1:
                    model2 = KMeans(n_clusters=2, init='k-means++',random_state=0)
                else:
                    model2 = KMeans(n_clusters=len(clusters_result[c]) // work_per_van + 1, init='k-means++', random_state=0)
                model2.fit(data2[['x', 'y']])
                clusters_result.remove(clusters_result[c])
                clusters_center[c] = []
                for j in range(model2.n_clusters):
                    clusters_result.append(data2[model2.labels_ == j].index.tolist())
                    clusters_center.append(model2.cluster_centers_[j])
    #print(clusters_result, clusters_center)
    clusters_center_list = []
    for i in clusters_center:
        if i != []:
            clusters_center_list.append(i)
    #print(clusters_result, clusters_center_list)
    return clusters_result,clusters_center_list

clusters_result,clusters_center_list = get_clusters_result()
#print(distance_matrix)
#print("distance_matrix",distance_matrix[64][1])
print(clusters_result)
"""for i in range(model.n_clusters):
    unserved = data[model.labels_ == i].index.tolist() #类中的node
    sum_weight = sum(weights_list[n] for n in unserved) #node的重量和
    #如果聚类所有点的weight和<18，将此类结果放入聚类中，否则循环一直分到所有类中的weight和<18为止
    if sum_weight<=work_per_van:
        clusters_result.append(unserved)
    print(unserved)
    print(sum(weights_list[n] for n in unserved))"""

"""显示聚类结果"""
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k']
for i in range(len(clusters_result)):
    plt.scatter(clusters_center_list[i][0], clusters_center_list[i][1], marker='+', c='r', s=50)
    for j in clusters_result[i]:
        plt.scatter(lat_x[j], lon_y[j], marker='o', c=color_list[i], s=5)
#plt.scatter(clusters_center_list[:,0], clusters_center_list[:,1], marker='+', c='r', s=50)
plt.scatter(data.iloc[0].x, data.iloc[0].y, marker='*', c='r', s=400)
plt.show()