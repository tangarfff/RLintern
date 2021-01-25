"""
中国区物流数据清洗，只保留海淀区的数据。为D-wave测试做准备
1.15 by Chu
"""
import numpy as np
import os
import os.path as osp


#generated a csv output
def data_Reader(path):
    file = osp.join(path)
    result_list = []
    adress = []
    address_text = []
    for line in open(file):
        lta = []
        content = line.strip('\n').split(',')
        if content[2]>18:
            continue
        if [content[5], content[6]] in adress:
            for i in result_list:
                if[content[5], content[6]]==[i[5],i[6]]:
                    print(content[4],i[2])
                    i[2]=float(i[2])+float(content[2])
                    print(i[2])

        if [content[5],content[6]] not in adress and content[4]not in address_text:
            if "EX_ID" not in content[0]:
                adress.append([content[5],content[6]])
                address_text.append(content[4])
                lta.append(content[0]) #EXID
                lta.append(content[1]) #QTY
                lta.append(content[2]) #VOL
                lta.append(content[3]) #WGT
                lta.append(content[4])#ADDR
                lta.append(content[5])  # LAT
                lta.append(content[6])  # LON
                lta.append(content[7])  # VAN_ID
                result_list.append(lta)
    name = "clean_test"+ path[24:]
    #print(name)

    np.savetxt(name, result_list, fmt='%s', delimiter=",")

for data_file in sorted(os.listdir("data/")):
    data_Reader("data/"+data_file)

#记录所有日期字符串
"""date_name_list = []
for data_file in sorted(os.listdir("all_data/")):
    date_name_list.append(data_file[16:24])
np.savetxt("date_name_list", date_name_list, fmt='%s', delimiter=",")"""
