import numpy as np
import sys
import time
import heapq
import random
FILEPATH = ''
TERMINATION = 0
RANDOMSEED = 0
NAME = ''
VERTICES = 0
DEPOT = 0
REQUIRED_EDGES = 0
NON_REQUIRED_EDGES = 0
VEHICLES = 0
CAPACITY = 0
TOTAL_COST_OF_REQUIRED_EDGES = 0
TOTAL_DEMAND_OF_REQUIRED_EDGES = 0
GRAPH = []


class Node:
    def __init__(self, cost, demand):
        self.cost = cost
        self.demand = demand

def dijkstra(GRAPH, VERTICES):
    '''
    :param GRAPH: 二维数组，包含了每条边的cost和demand
    :return:DIJK 每两个点之间最小的距离的二维矩阵
    '''
    #S 数组，元组左边的值是点到sourse点的距离，右边的值是到该点名称
    #U 数组
    DIJK = np.ones((VERTICES + 1, VERTICES + 1), dtype='int64')

    for sourse_name in range(1, VERTICES + 1):
        S = [(0, sourse_name)]
        U = []
        #U 此时没有包含DEPOT
        for point_name in range(1, sourse_name):
            heapq.heappush(U, (88888888, point_name))
        for point_name in range(sourse_name + 1, VERTICES+1):
            heapq.heappush(U, (88888888, point_name))

        while len(S) != VERTICES:
            anchor = S[-1]
            # 锚点与源点之间的距离
            anchor_length = anchor[0]
            # 锚点的名称，连接锚点的所有距离可能会更新
            anchor_name = anchor[1]
            # 先将锚点附近的点的值更新一下下
            for i in range(1, VERTICES + 1):
                u_length = len(U)
                for j in range(u_length):
                    this_name = U[j][1]
                    # 发现了这个点了，开始算这个点与锚点之间的性质
                    if this_name == i:
                        # U[j][0]指的是这个点到源点的距离
                        # GRAPH[anchor_name][this_name]指的是这个点到锚点的距离
                        this_length = U[j][0]
                        anchor_and_this_length = GRAPH[anchor_name][this_name].cost
                        this_new_length = anchor_length + anchor_and_this_length
                        # 如果大于，则更新
                        if this_length > this_new_length:
                            U.pop(j)
                            item = (this_new_length, this_name)
                            heapq.heapify(U)
                            heapq.heappush(U, item)
                        break
            item = heapq.heappop(U)
            S.append(item)

        for S_index in range(len(S)):
            S_point_length = S[S_index][0]
            S_point_name = S[S_index][1]
            DIJK[sourse_name][S_point_name] = S_point_length

    return DIJK


def pathscanning(RANDOMSEED,
                 GRAPH, DIJK,
                 unallocateList, DEPOT,
                 VEHICLES, CAPACITY,
                 total_cost,
                 total_demand,
                 num_edge_demand,alpha):
    '''
    进行path的扫描
    :param GRAPH: 图上邻接点间的cost和demand的矩阵
    :param DIJK: dijkstra的矩阵，每两个点之间的最短距离的矩阵
    :param unallocateList: 未分配任务的边的集合
    :return:
    '''
    rem_demand = total_demand
    #所有的车的cost集合
    outterCostList = []
    #所有的车的任务集合
    outterTaskList = []
    vehicle = 0
    while(len(unallocateList) != 0):
        #就是说所有未分配任务的边都成功分配了任务，剩下车也无妨
        vehicle = vehicle + 1
        remainderCAPACITY = CAPACITY
        nowVertex = DEPOT
        # 因为有很多辆车，因此分别对应每一辆车，它的所有任务将会放到这里
        innerTaskList = []
        #其中一辆车的cost
        innerCost = 0

        while True:
            candidateDistanceMin = 88888888
            candidateList = []

            for i in range(len(unallocateList)):
                unallocateEdge = unallocateList[i]
                x = unallocateEdge[0]
                y = unallocateEdge[1]

                if GRAPH[x][y].demand <= remainderCAPACITY:
                    if candidateDistanceMin > DIJK[nowVertex][x]:
                        candidateList = []
                        candidateList.append((x, y))
                        candidateDistanceMin = DIJK[nowVertex][x]
                    elif candidateDistanceMin == DIJK[nowVertex][x]:
                        candidateList.append((x, y))

            point = (0,0)
            if len(candidateList) == 1:
                point = candidateList[0]

            elif len(candidateList) > 1:
                r = random.randint(0, int(len(candidateList))-1)
                point = candidateList[r]

            # 此时车没有可以运输的点
            elif len(candidateList) == 0:
                # 直接回家,跳出迭代
                innerCost = innerCost + DIJK[nowVertex][DEPOT]
                break

            point_x = point[0]
            point_y = point[1]
            # 先从unallocateList中去掉（point_x,point_y）
            # 再从unallocateList中去掉（point_y,point_x）
            unallocateList.remove((point_x, point_y))
            unallocateList.remove((point_y, point_x))
            # rem_demand = rem_demand - GRAPH[point_x][point_y].demand
            # 计算此时的cost
            innerCost = innerCost + DIJK[nowVertex][point_x] + GRAPH[point_x][point_y].cost
            # 计算此时的Task
            innerTaskList.append((point_x, point_y))
            # 更新此时点的位置
            nowVertex = point_y
            # 更新此时剩余CAPACITY
            remainderCAPACITY = remainderCAPACITY - GRAPH[point_x][point_y].demand

        # 还没完，这个时候还要将innerCost和innerTaskList放进outter里面
        outterCostList.append(innerCost)
        outterTaskList.append(innerTaskList)
    return (outterCostList,outterTaskList,unallocateList,vehicle)


def getArg():
    '''
    这个方法是用在命令行中获得参数
    '''
    global FILEPATH, TERMINATION, RANDOMSEED
    FILEPATH = sys.argv[1]
    TERMINATION = int(sys.argv[3])
    RANDOMSEED = int(sys.argv[5])
    return (FILEPATH, TERMINATION, RANDOMSEED)


def lastWord(s):
    """
    在file_line中寻找最后一个词
    """
    s = s.strip()
    return s.split(" ")[-1]


def readFile(filepath):
    '''
    :param filepath:CARP_samples路径的值
    :return:顺便返回未分辨的数组，主要是用来读取数据的，构造图
    '''
    global NAME, VERTICES, DEPOT, REQUIRED_EDGES, NON_REQUIRED_EDGES, VEHICLES, CAPACITY, TOTAL_COST_OF_REQUIRED_EDGES,GRAPH,TOTAL_DEMAND_OF_REQUIRED_EDGES

    unallocateList = []

    file = open(filepath)
    file_line = file.readlines()

    NAME = lastWord(file_line[0])
    VERTICES = int(lastWord(file_line[1]))
    DEPOT = int(lastWord(file_line[2]))
    REQUIRED_EDGES = int(lastWord(file_line[3]))
    NON_REQUIRED_EDGES = int(lastWord(file_line[4]))
    VEHICLES = int(lastWord(file_line[5]))
    CAPACITY = int(lastWord(file_line[6]))
    TOTAL_COST_OF_REQUIRED_EDGES = int(lastWord(file_line[7]))
    TOTAL_DEMAND_OF_REQUIRED_EDGES = 0
    node = Node(88888888, 0)
    GRAPH = np.ones((VERTICES+1, VERTICES+1), dtype='object')
    for i in range(VERTICES+1):
        for j in range(VERTICES+1):
            GRAPH[i][j] = node

    #从这里开始读取点
    offset = 9
    while file_line[offset] != 'END':
        line = file_line[offset].strip().split()
        x = int(line[0])
        y = int(line[1])
        cost = int(line[2])
        demand = int(line[3])

        TOTAL_DEMAND_OF_REQUIRED_EDGES += demand
        node = Node(cost, demand)
        GRAPH[x][y] = node
        GRAPH[y][x] = node
        if demand != 0:
            unallocateList.append((x, y))
            unallocateList.append((y, x))
        offset = offset + 1

    return unallocateList

def output(min_tasklist, min_sum_cost):
    outputS = ''
    outputS += 's '
    for index1 in range(len(min_tasklist)):
        outputS += '0,'
        for index2 in range(len(min_tasklist[index1])):
            tup = str(min_tasklist[index1][index2]) + ','
            tup = tup.replace(' ', '')
            outputS += tup
        outputS += '0,'
    outputS = outputS[:-1]
    outputQ = 'q ' + str(min_sum_cost)
    return (outputS,outputQ)


if __name__ == '__main__':
    start = int(time.time())
    getArg()
    unallocateList = readFile(FILEPATH)
    stable_unallocateList = unallocateList[:]
    DIJK = dijkstra(GRAPH, VERTICES)
    min_costlist = []
    min_tasklist = []
    min_sum_cost = 88888888
    alpha_list = [0.9, 1.0,1.1,1.2,1.3,1.4, 1.5,1.6,1.7,1.8,1.9, 2.0,2.1,2.2,2.3,2.4,2.5,2.6]
    alpha = 1
    while int(time.time()) - start < TERMINATION - 2:

    #每种rule都来一次，直到找到最低cost的值
        unallocateList = stable_unallocateList[:]
        ran_alpha = random.randint(0,len(alpha_list)-1)
        alpha = alpha_list[ran_alpha]

        (costlist,tasklist,unallocateList,vehicle) = pathscanning(RANDOMSEED,
                                                          GRAPH, DIJK,
                                                          unallocateList, DEPOT,
                                                          VEHICLES, CAPACITY,
                                                          TOTAL_COST_OF_REQUIRED_EDGES,
                                                          TOTAL_DEMAND_OF_REQUIRED_EDGES,
                                                          REQUIRED_EDGES,alpha)
        sum = 0

        for costlist_index in range(len(costlist)):
            sum = costlist[costlist_index] + sum
        if sum == 0:
            alpha_list.remove(alpha)
            continue
        if sum < min_sum_cost and len(unallocateList) == 0:
            min_costlist = costlist[:]
            min_tasklist = tasklist[:]
            min_sum_cost = sum

        if VERTICES > 100:
            begin_flip_time = 20
        elif VERTICES > 70:
            begin_flip_time = 10
        elif VERTICES > 40:
            begin_flip_time = 5
        else:
            begin_flip_time = 2

        if VERTICES > 100:
            cost_add = 150
        elif VERTICES > 70:
            cost_add = 125
        elif VERTICES > 40:
            cost_add = 100
        else:
            cost_add = 50
        # -----------------遍历所有Flip----------------------------------------
        if sum < min_sum_cost + cost_add and int(time.time()) - start < TERMINATION - 6 and int(time.time()) - start > begin_flip_time and len(unallocateList) == 0:
            temp_sum = sum
            temp_tasklist = tasklist[:]
            # 经历一次flip
            for i in range(0, len(tasklist)):
                for j in range(0, len(tasklist[i])):
                    SR_bef_y = 0
                    SR_x = 0
                    SR_y = 0
                    SR_aft_x = 0
                    SR_edge = []
                    SR_after = []
                    if (len(temp_tasklist[i]) == 1):
                        continue
                    if j == 0:
                        SR_edge = temp_tasklist[i][j]
                        SR_after = temp_tasklist[i][j + 1]
                        SR_x = SR_edge[0]
                        SR_y = SR_edge[1]
                        SR_bef_y = 1
                        SR_aft_x = SR_after[0]
                    elif j >= 1 and j <= len(temp_tasklist[i]) - 2:
                        SR_edge = temp_tasklist[i][j]
                        SR_before = temp_tasklist[i][j - 1]
                        SR_after = temp_tasklist[i][j + 1]
                        SR_x = SR_edge[0]
                        SR_y = SR_edge[1]
                        SR_bef_y = SR_before[1]
                        SR_aft_x = SR_after[0]
                    elif j == len(temp_tasklist[i]) - 1:
                        SR_edge = temp_tasklist[i][j]
                        SR_before = temp_tasklist[i][j - 1]
                        SR_x = SR_edge[0]
                        SR_y = SR_edge[1]
                        SR_bef_y = SR_before[1]
                        SR_aft_x = 1
                    chliaristic = - DIJK[SR_bef_y][SR_x] + DIJK[SR_bef_y][SR_y] - DIJK[SR_y][SR_aft_x] + DIJK[SR_x][SR_aft_x]
                    newsum = sum + chliaristic
                    if newsum < temp_sum:
                        temp_sum = newsum
                        costlist[i] = costlist[i] + chliaristic
                        temp_tasklist[i][j] = (SR_y, SR_x)
            if temp_sum < min_sum_cost:
                min_sum_cost = temp_sum
                min_tasklist = temp_tasklist[:]
                    # ------------------------MS---MS----MS--------------------------------
            elif temp_sum >= min_sum_cost:
                temp_temp_sum = temp_sum
                best_ms_tasklist = temp_tasklist[:]
                count = 0
                alpha_count = 0
                local_op = 0
                local_op_count = 0
                while int(time.time()) - start < TERMINATION - 5:
                    count = count + 1
                    alpha_count = alpha_count + 1
                    p = 2 # 任取两条线
                    ms_tasklist = best_ms_tasklist[:]
                    new_unallocatesearch = []
                    rr = random.randint(0, 4)
                    # 取得第一条线
                    r1 = random.randint(0, VEHICLES-1)
                    for i in range(len(ms_tasklist[r1])):
                        new_unallocatesearch.append(ms_tasklist[r1][i])
                    ms_tasklist.remove(ms_tasklist[r1])
                    # 取得第二条线
                    if VEHICLES - 2 > 0 and rr > 1:
                        r2 = random.randint(0, VEHICLES-2)
                        for j in range(len(ms_tasklist[r2])):
                            new_unallocatesearch.append(ms_tasklist[r2][j])
                        ms_tasklist.remove(ms_tasklist[r2])
                    # 取得第三条线
                    if VEHICLES - 3 > 0 and rr > 2:
                        r3 = random.randint(0, VEHICLES-3)
                        for k in range(len(ms_tasklist[r3])):
                            new_unallocatesearch.append(ms_tasklist[r3][k])
                        ms_tasklist.remove(ms_tasklist[r3])
                    # 取得第四条线
                    if VEHICLES - 4 > 0 and rr > 3:
                        r4 = random.randint(0, VEHICLES-4)
                        for q in range(len(ms_tasklist[r4])):
                            new_unallocatesearch.append(ms_tasklist[r4][q])
                        ms_tasklist.remove(ms_tasklist[r4])

                    total_cost_new_unall = 0
                    total_demand_new_unall = 0
                    for cost_index in range(len(new_unallocatesearch)):
                        new_edge = new_unallocatesearch[cost_index]
                        new_edge_x = new_edge[0]
                        new_edge_y = new_edge[1]
                        total_cost_new_unall += GRAPH[new_edge_x][new_edge_y].cost
                        total_demand_new_unall += GRAPH[new_edge_x][new_edge_y].demand
                    len_before_new_unallocatesearch = len(new_unallocatesearch)
                    for i in range(len_before_new_unallocatesearch):
                        new_unallocatesearch_edge = new_unallocatesearch[i]
                        new_unallocatesearch_edge_x = new_unallocatesearch_edge[0]
                        new_unallocatesearch_edge_y = new_unallocatesearch_edge[1]
                        new_unallocatesearch.append((new_unallocatesearch_edge_y,new_unallocatesearch_edge_x))

                    ran_alpha = random.randint(0, len(alpha_list) - 1)
                    alpha = alpha_list[ran_alpha]

                    (t_ms_costlist, t_ms_tasklist, new_unallocateList,vehicle) = pathscanning(RANDOMSEED,
                                                                        GRAPH, DIJK,
                                                                        new_unallocatesearch, DEPOT,
                                                                        2, CAPACITY,
                                                                        total_cost_new_unall,
                                                                        total_demand_new_unall,
                                                                        len(new_unallocatesearch)/2, alpha)
                    temp_sum = 0
                    if len(new_unallocatesearch) == 0:
                        for i in range(len(t_ms_tasklist)):
                            ms_tasklist.append(t_ms_tasklist[i][:])
                        # --------------------------从现在开始，再算一次cost-----------
                        temp_sum = 0
                        for i in range(len(ms_tasklist)):
                            for j in range(len(ms_tasklist[i])):
                                SR_bef_y = 0
                                SR_x = 0
                                SR_y = 0
                                SR_aft_x = 0
                                SR_edge = []
                                SR_after = []
                                if (len(ms_tasklist[i]) == 1):
                                    SR_edge = ms_tasklist[i][j]
                                    SR_x = SR_edge[0]
                                    SR_y = SR_edge[1]
                                    temp_sum = temp_sum + DIJK[DEPOT][SR_x] + GRAPH[SR_x][SR_y].cost + DIJK[SR_y][DEPOT]
                                    continue
                                if j == 0:
                                    SR_edge = ms_tasklist[i][j]
                                    SR_after = ms_tasklist[i][j + 1]
                                    SR_x = SR_edge[0]
                                    SR_y = SR_edge[1]
                                    SR_bef_y = 1
                                    SR_aft_x = SR_after[0]
                                    temp_sum = temp_sum + DIJK[SR_bef_y][SR_x] + GRAPH[SR_x][SR_y].cost
                                elif j >= 1 and j <= len(ms_tasklist[i]) - 2:
                                    SR_edge = ms_tasklist[i][j]
                                    SR_before = ms_tasklist[i][j - 1]
                                    SR_after = ms_tasklist[i][j + 1]
                                    SR_x = SR_edge[0]
                                    SR_y = SR_edge[1]
                                    SR_bef_y = SR_before[1]
                                    SR_aft_x = SR_after[0]
                                    temp_sum = temp_sum + DIJK[SR_bef_y][SR_x] + GRAPH[SR_x][SR_y].cost
                                elif j == len(ms_tasklist[i]) - 1:
                                    SR_edge = ms_tasklist[i][j]
                                    SR_before = ms_tasklist[i][j - 1]
                                    SR_x = SR_edge[0]
                                    SR_y = SR_edge[1]
                                    SR_bef_y = SR_before[1]
                                    SR_aft_x = 1
                                    temp_sum = temp_sum + DIJK[SR_bef_y][SR_x] + GRAPH[SR_x][SR_y].cost + DIJK[SR_y][1]
                        # ------------------------------------------------------------
                        temp_tasklist = ms_tasklist[:]
                        if temp_sum < min_sum_cost:
                            count = 0
                            min_sum_cost = temp_sum
                            min_tasklist = ms_tasklist[:]
                            best_ms_tasklist = ms_tasklist[:]
                            temp_temp_sum = temp_sum
                        if temp_temp_sum > temp_sum:
                            best_ms_tasklist = ms_tasklist[:]
                            temp_temp_sum = temp_sum
                            count = 0
                        if count > 50 and temp_temp_sum + cost_add > temp_sum:
                            count = 0
                            temp_temp_sum = temp_sum
                            best_ms_tasklist = ms_tasklist[:]
                        # 经历一次flip
                        for i in range(0, len(temp_tasklist)):
                            for j in range(0, len(temp_tasklist[i])):
                                SR_bef_y = 0
                                SR_x = 0
                                SR_y = 0
                                SR_aft_x = 0
                                SR_edge = []
                                SR_after = []
                                if (len(temp_tasklist[i]) == 1):
                                    continue
                                if j == 0:
                                    SR_edge = temp_tasklist[i][j]
                                    SR_after = temp_tasklist[i][j + 1]
                                    SR_x = SR_edge[0]
                                    SR_y = SR_edge[1]
                                    SR_bef_y = 1
                                    SR_aft_x = SR_after[0]
                                elif j >= 1 and j <= len(temp_tasklist[i]) - 2:
                                    SR_edge = temp_tasklist[i][j]
                                    SR_before = temp_tasklist[i][j - 1]
                                    SR_after = temp_tasklist[i][j + 1]
                                    SR_x = SR_edge[0]
                                    SR_y = SR_edge[1]
                                    SR_bef_y = SR_before[1]
                                    SR_aft_x = SR_after[0]
                                elif j == len(temp_tasklist[i]) - 1:
                                    SR_edge = temp_tasklist[i][j]
                                    SR_before = temp_tasklist[i][j - 1]
                                    SR_x = SR_edge[0]
                                    SR_y = SR_edge[1]
                                    SR_bef_y = SR_before[1]
                                    SR_aft_x = 1
                                chliaristic = - DIJK[SR_bef_y][SR_x] + DIJK[SR_bef_y][SR_y] - DIJK[SR_y][SR_aft_x] + \
                                              DIJK[SR_x][SR_aft_x]
                                newsum = temp_sum + chliaristic
                                if (newsum < temp_sum):
                                    temp_sum = newsum
                                    temp_tasklist[i][j] = (SR_y, SR_x)
                        if temp_sum < min_sum_cost:
                            count = 0
                            min_sum_cost = temp_sum
                            min_tasklist = temp_tasklist[:]
                            best_ms_tasklist = ms_tasklist[:]
                            temp_temp_sum = temp_sum
                        if temp_temp_sum > temp_sum:
                            count = 0
                            best_ms_tasklist = ms_tasklist[:]
                            temp_temp_sum = temp_sum
                        if count > 50 and temp_temp_sum + cost_add > temp_sum:
                            count = 0
                            best_ms_tasklist = ms_tasklist[:]
                            temp_temp_sum = temp_sum
    (outS, outQ) = output(min_tasklist, min_sum_cost)

    print(outS)
    print(outQ)
