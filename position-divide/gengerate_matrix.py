# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:07:55 2018

@author: Dylan
"""

import numpy as np
import random
import copy
import numpy as np
import os
import sys
import os.path as osp

allow_time = 600  # min


class Node(object):
    '''
    顾客点类：
    c_id:Number,顾客点编号
    x:Number,点的横坐标
    y:Number,点的纵坐标
    demand:Number,点的需求量
    ready_time:Number,点的最早访问时间
    due_time:Number,点的最晚访问时间
    service_time:Number,点的服务时间
    belong_veh:所属车辆编号
    '''

    def __init__(self, c_id, x, y, demand, ready_time, due_time, service_time):
        self.c_id = c_id
        self.x = x
        self.y = y
        self.demand_left = demand
        self.demand_togo = demand
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        self.belong_veh = None


class Vehicle(object):
    '''
    车辆类：
    v_id:Number,车辆编号
    cap:Number,车的最大载重量
    load:Number,车的载重量
    distance:Number,车的行驶距离
    violate_time:Number,车违反其经过的各点时间窗时长总和
    route:List,车经过的点index的列表
    start_time:List,车在每个点的开始服务时间
    '''

    def __init__(self, v_id: int, cap: int):
        self.v_id = v_id
        self.cap = cap
        self.load = [0]
        self.distance = [0]
        self.violate_time = [0]
        self.route = [[0]]
        self.start_time = [[0]]
        self.allow_time_left = allow_time  # 600 min
        self.cur_route_seq = 0

    # 插入节点
    def insert_node(self, node: int, index: int = 0) -> None:
        if (len(self.route) < (
                    self.cur_route_seq + 1)):  # append init elements after updating route_seq to avoid indexing failure
            self.load.append(0)
            self.distance.append(0)
            self.violate_time.append(0)
            self.route.append([0])
            self.start_time.append([0])
        if index == 0:
            self.route[self.cur_route_seq].append(node)
        else:
            self.route[self.cur_route_seq].insert(index, node)
        # node.belong_veh = self.v_id
        self.update_info()

    # 根据索引删除节点
    def del_node_by_index(self, index: int) -> None:
        self.route[self.cur_route_seq].pop(index)
        self.update_info()

    # 根据对象删除节点
    def del_node_by_node(self, node: Node) -> None:
        self.route[self.cur_route_seq].remove(node.c_id)
        self.update_info()

    # 更新载重、距离、开始服务时间、时间窗违反
    def update_info(self) -> None:
        # 更新载重
        cur_load = 0
        for n in self.route[self.cur_route_seq]:
            cur_load += nodes[n].demand_togo
        self.load[self.cur_route_seq] = cur_load
        # 更新距离
        cur_distance = 0
        for i in range(len(self.route[self.cur_route_seq]) - 1):
            cur_distance += distance_matrix[self.route[self.cur_route_seq][i]][self.route[self.cur_route_seq][i + 1]]
        self.distance[self.cur_route_seq] = cur_distance
        # 更新违反时间窗时长总和(硬时间窗,早到等待，不可晚到)
        arrival_time = 0
        if (self.cur_route_seq > 0):  # Not first route
            self.start_time[self.cur_route_seq] = self.start_time[self.cur_route_seq - 1][
                                                  -1:]  # Last depot arrival time
        else:
            self.start_time[self.cur_route_seq] = [0]
        cur_violate_time = 0
        arrival_time += self.start_time[self.cur_route_seq][0]
        for i in range(1, len(self.route[self.cur_route_seq])):
            last_service_time = nodes[self.route[self.cur_route_seq][i - 1]].service_time
            if i > 2:
                delta_x = abs(
                    nodes[self.route[self.cur_route_seq][i - 1]].x - nodes[self.route[self.cur_route_seq][i - 2]].x)
                delta_y = abs(
                    nodes[self.route[self.cur_route_seq][i - 1]].y - nodes[self.route[self.cur_route_seq][i - 2]].y)
                if (delta_x + delta_y) < 1E-5:
                    last_service_time = 0.0
            arrival_time += distance_matrix[self.route[self.cur_route_seq][i - 1]][
                                self.route[self.cur_route_seq][i]] + last_service_time
            if arrival_time > nodes[self.route[self.cur_route_seq][i]].due_time:
                cur_violate_time += arrival_time - nodes[self.route[self.cur_route_seq][i]].due_time
            elif arrival_time < nodes[self.route[self.cur_route_seq][i]].ready_time:
                arrival_time = nodes[self.route[self.cur_route_seq][i]].ready_time
            self.start_time[self.cur_route_seq].append(arrival_time)
        self.violate_time[self.cur_route_seq] = cur_violate_time
        # Update total time left
        self.allow_time_left = allow_time - arrival_time  # all travel time, including time cost on getting back to depot

    def __str__(self):  # 重载print()
        routes = [n for n in self.route]
        return '车{}:距离[{:.4f}];载重[{}];时间违反[{:.4f}]\n路径{}\n开始服务时间{}\n'.format(self.v_id, self.distance, self.load,
                                                                             self.violate_time, routes, self.start_time)


# 读取数据文件，返回车辆最大载重，最大车辆数，所有Node组成的列表
def read_data(path: str) -> (int, int, list):
    with open(path, 'r', ) as f:
        lines = f.readlines()
    capacity = (int)(lines[4].split()[-1])
    max_vehicle = (int)(lines[4].split()[0])
    # print("max_vehicle",max_vehicle)
    lines = lines[9:]
    nodes = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 7:
            node = Node(*info)
            nodes.append(node)
    return capacity, max_vehicle, nodes


# 计算距离矩阵
def cal_distance_matrix(nodes: list) -> np.array:
    distance_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if i != j:
                # dis = np.sqrt((nodes[i].x - nodes[j].x) ** 2 + (nodes[i].y - nodes[j].y) ** 2)
                dis = abs(nodes[i].x - nodes[j].x) + abs(nodes[i].y - nodes[j].y)
                distance_matrix[i][j] = distance_matrix[j][i] = dis
    # print(len(distance_matrix),len(distance_matrix[0]))
    # print(distance_matrix[34][34])
    return distance_matrix

def mergeByLL(in_df, datestr, vol_lim=5):
    df = in_df.sort_values(by=['LAT', 'LON', 'VOL'], ascending=[True, True, True])
    ibatch = 0
    last_lat = np.inf
    last_lon = np.inf
    cumvol = 0
    for irec, rec in df.iterrows():
        if (last_lat != rec.LAT) | (last_lon != rec.LON) | ((cumvol + rec.VOL) > vol_lim):
            ibatch += 1
            last_lat, last_lon = rec.LAT, rec.LON
            cumvol = rec.VOL
            df.loc[irec, 'label'] = ibatch
            df.loc[irec, 'cumvol'] = cumvol
        elif (last_lat == rec.LAT) & (last_lon == rec.LON) & ((cumvol + rec.VOL) < vol_lim):
            last_lat, last_lon = rec.LAT, rec.LON
            cumvol += rec.VOL
            df.loc[irec, 'label'] = ibatch
            df.loc[irec, 'cumvol'] = cumvol
    df['label'] = df['label'].astype(int)
    cum_vol = df.groupby(['label'], as_index=False).max()[['label', 'cumvol']]  # df.groupby(['label']).max()['cumvol']
    df = df.drop(columns=['cumvol'])
    df = df.merge(cum_vol, on='label')
    merged_df = df.groupby('label', as_index=False).first()[['cumvol', 'LAT', 'LON']]
    merged_df = merged_df.rename(columns={'cumvol': 'VOL'})
    merged_df.to_csv('./merged_input_with_label_{}.csv'.format(datestr), index=False, encoding='utf-8-sig')
    df = df.drop(columns=['cumvol'])
    df.to_csv('./input_with_label_{}.csv'.format(datestr), index=False, encoding='utf-8-sig')
    return merged_df, df


# 生成D-wave中vrp测试匹配的地址
def generate_data(datestr):
    import pandas as pd
    # 读取数据

    root_path =  sys.path[0]
    print("root_path",root_path)
    lc_start_x = 116.59681754
    lc_start_y = 40.12989378

    #data1 = pd.read_csv(data_folder + '/ORDER_MAN_INPUT_{}_PM.csv'.format(datestr))
    data1 = pd.read_csv(sys.path[0] + '/clean_test_{}_PM.csv'.format(datestr))

    data1_merged, data1 = mergeByLL(data1, datestr=datestr + '_PM', vol_lim=5.0)


    deg2min = 85000 / 500.0  # 85000 m (1deg) / 30km/h (500m/min)

    nodes = list()
    nodes.append(Node(0, lc_start_x * deg2min, lc_start_y * deg2min, 0, 0, 600, 0))
    print("data1_merged.shape[0]",data1_merged.shape)
    for k in range(data1_merged.shape[0]):
        nodes.append(
            Node(k + 1, data1_merged.iat[k, 2] * deg2min, data1_merged.iat[k, 1] * deg2min, data1_merged.iat[k, 0], 0,
                 999, 18))  # till 13:00
    print(len(nodes))

    file = osp.join(sys.path[0] + '/clean_test_{}_PM.csv'.format(datestr))
    weights_list = []
    for line in open(file):
        content = line.strip('\n').split(',')
        if "VOL" not in content[2]:
            weights_list.append(float(content[2]))  # add weight

    distance_matrix = cal_distance_matrix(nodes)  # 将距离矩阵赋值给车辆的类变量
    print("nodes",len(nodes))
    sources = [0]
    weight_sum = 0
    for i in range(len(weights_list)):
        weight_sum = weight_sum + float(weights_list[i])

    capacities = [18 for n in range(12)]
    dests = [n for n in range(1, len(weights_list))]
    print("distance_matrix",len(distance_matrix))
    print("capacities", capacities)
    print("weights_list",weights_list,len(weights_list))
    print("dests",dests,len(dests))

    return sources, distance_matrix, capacities,dests, weights_list

generate_data('20190605')



