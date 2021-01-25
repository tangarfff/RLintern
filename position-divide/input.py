import csv
import math
from itertools import product

import networkx as nx
import numpy as np
from vrp_problem import VRPProblem
import os.path as osp
import sys
import pandas as pd
# Creates VRPProblem from test file.
# path - path to test file
# capacity - True if vehicles have capacities, False otherwise
def encode(costs_matrix,dest_list):
    dest_list = [0]+dest_list
    dest = []
    for i in range(len(dest_list)-1):
        dest.append(i+1)
    distance_matrix = np.zeros((len(dest_list), len(dest_list)))
    for i in range(len(dest_list)):
        for j in range(i + 1, len(dest_list)):
            if i != j:
                distance_matrix[i][j] = distance_matrix[j][i] = costs_matrix[dest_list[i]][dest_list[j]]
    print(len(distance_matrix),len(distance_matrix[0]))
    return distance_matrix,dest

def read_test(part_solution,path, capacity = False):
    """将坐标写入"""
    # 起始点
    lc_start_x = 116.59681754
    lc_start_y = 40.12989378
    # file = osp.join(sys.path[0] + '/clean_test_{}_PM.csv'.format(datestr))
    file = osp.join(sys.path[0]+path)
    weights_list, lat_x, lon_y = [], [], []
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

    """计算距离矩阵"""
    # distance_func = lambda a,b: np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)#欧式距离
    # distance_func = lambda ax,bx,ay,by: np.abs(ax-bx)+np.abs(ay-by)#曼哈顿距离
    distance_func = lambda a, b: np.abs(a.x - b.x) + np.abs(a.y - b.y)  # 曼哈顿距离
    def compute_distances(dfunc=distance_func):
        for i in range(node_num):
            current = data.iloc[i]
            data[i] = dfunc(current, data)
        print(data)
        return data


    distance_matrix = compute_distances()
    sources = [0]
    costs,dests = encode(distance_matrix,part_solution)
    weights = []
    capacities = [10000000]
    #print(sources, costs, capacities, dests, weights)
    return VRPProblem(sources, costs, capacities, dests, weights)



"""sources=[0, 1]
    costs= [[0,77,48, 48, 69, 53, 24, 19, 37, 73, 55, 39],
     [67,0,20,42,45,52,47,42,48,73,39,39],
     [90,76 ,0 ,66, 73, 55, 62, 57, 73, 95, 62, 64],
     [37,81,29,0, 36, 10, 57, 52, 59, 57, 9, 50],
     [72,72,42,41,0,51,64,59,77,70,44,74],
     [65,90,51,28,64,0,79,74,87,85,37,78],
     [61,54,24,24,46,34,0,25,33,58,33,24],
     [72,76,40,48,62,58,26,0,18,55,44,21],
     [93,70,63,69,62,79,49,23,0,78,65,44],
     [62,74,44,25,57,35,45,43,55,0,34,16],
     [28,72,20,29,61,16,52,47,50,48,0,41],
     [81,87,57,57,76,67,33,28,46,69,58,0]]
    capacities=[390, 14, 550]
    dests=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]"""
    #weights=[0  0 11 86 70 70 75 81 47 82 81 33]




