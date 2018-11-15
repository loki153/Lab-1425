# -*-coding:utf-8
# @Time     :2018/11/15 21:36
# @Author   :Loki
# @FileName :Variant.py  
# @SoftWare :PyCharm


import os

import re
from math import log as log

import numpy as np
from matplotlib import  pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def Segment(string,m):
    """
    0-1 sequence segment
    :param string:
    :param m:
    :return:
    """
    return [string[i:i + m] for i in range(0, len(string), m)]


def Shift(name,m):
    """
    Shifting module
    :param name:
    :param m:
    :return:
    """
    isExist = os.path.exists("move/")
    if (isExist == False):
        os.makedirs("move/" + name + ' move')  # 生成相关的目录

    fr = open('data/' + name + '.txt', 'r')
    line = fr.read()
    a = Segment(line, m)
    N = len(a)
    fr.close()

    i = 0
    while (i < m):
        cache = []  # 储存当前的输入数据Xi
        move = 1 + i  # 游标

        j = 0
        while (j < N - 1):
            temp = a[j][move:] + a[j + 1][0:move]  # 分段移位
            cache.append(temp)
            j += 1

        fw = open('move/' + name + ' move/' + name + ' r=' + str(move) + '.txt', 'w+')  # 写入新的数据
        newline = ''.join(cache)
        fw.write(newline)
        fw.close()


        i += 1

    f = open('move/' + name + ' move/' + name + ' r=' + str(0) + '.txt', 'w+')
    f.write(line)
    f.close()
    return

def Measurement(name,r,m):
    """
    basic messureament module
    :param name:
    :param r:
    :param m:
    :return:
    """
    f = open('move/' + name + ' move/' + name + ' r=' + str(r) + '.txt', 'r')
    line = f.read()
    f.close()
    a = Segment(line, m)
    N = len(a)

    P = [0.0 for i in range(N + 1)]
    Q = [0.0 for i in range(N + 1)]
    Pnum = [0.0 for i in range(m + 1)]
    Qnum = [0.0 for i in range(m + 1)]
    PQnum = np.zeros([(m + 1), (int(m / 2 + 1)), ])

    i = 0
    while (i < N):
        v1 = len(re.findall("1", a[i]))
        v2 = len(re.findall("01", a[i]))

        if (a[i][0] == '1' and a[i][-1] == '0'):
            v2 += 1

        P[i] = v1
        Q[i] = v2
        Pnum[v1] += 1.0
        Qnum[v2] += 1.0
        PQnum[v1][v2] += 1  # 变值测量部分
        i += 1

    return P, Q, Pnum, Qnum, PQnum, N

def Projection(name,r,m,P,Q,Pnum,Qnum,PQnum,N):
    """
    make projection figure on the basis of the P,Q distribution
    :param name:
    :param r:
    :param m:
    :param P:
    :param Q:
    :param Pnum:
    :param Qnum:
    :param PQnum:
    :param N: 
    :return:
    """
    x = [i for i in range(m + 1)]

    isExist = os.path.exists('figure/')
    if (isExist == False):
        os.makedirs('figure/' + name + ' fig/' + 'P/')
        os.makedirs('figure/' + name + ' fig/' + 'Q/')
        os.makedirs('figure/' + name + ' fig/' + 'PQ/')
        os.makedirs('figure/' + name + ' fig/' + '3D/')

    for j in range(len(Pnum)):
        Pnum[j] = Pnum[j] / N
        Qnum[j] = Qnum[j] / N

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + '_1dP' + '_' + str(r))
    plt.bar(x, Pnum, color='b')
    plt.savefig('figure/' + name + ' fig/P/' + name + '_' + str(m) + '_' + str(r))
    plt.close(fig1)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.set_title('name' + '1dQ' + '_' + str(r))
    plt.bar(x, Qnum, color='b')
    plt.savefig('figure/' + name + ' fig/Q/' + name + '_' + str(m) + '_' + str(r))
    plt.close(fig2)

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.set_title(name + '_2dPQ' + '_' + str(r))
    plt.hist2d(P, Q, bins=40, norm=LogNorm())
    plt.xlim(15, 100)
    plt.ylim(15, 50)
    plt.savefig('figure/' + name + ' fig/PQ/' + name + '_' + str(m) + '_' + str(r))
    plt.close(fig3)
    return



def Maximal(Pnum,Qnum,PQnum,N):
    """
    calculate the maximal of the each distribution
    :param Pnum: 
    :param Qnum: 
    :param PQnum: 
    :param N: 
    :return: 
    """
    i = 0
    tempmax = 0.0
    while( i < len(PQnum)):
        if (tempmax <= max(PQnum[i])):
            tempmax = max(PQnum[i])
        i += 1

    maxP = max(Pnum) / N
    maxQ = max(Qnum) / N
    maxPQ = tempmax / N

    return maxP,maxQ,maxPQ

def Entropy(Pnum,Qnum,PQnum,N,r,name,m):
    """
    calculate the entropy of each distribution
    :param Pnum: 
    :param Qnum: 
    :param PQnum: 
    :param N: 
    :param r: 
    :param name: 
    :param m: 
    :return: 
    """
    P = []
    Q = []
    PQ = []

    i = 0
    while (i < len(Pnum)):
        if (Pnum[i] != 0):
            Pentropy = (Pnum[i] / N) * -(log(Pnum[i] / N))
            P.append(Pentropy)
        if (Qnum[i] != 0):
            Qentropy = (Qnum[i] / N) * -(log(Qnum[i] / N ))
            Q.append(Qentropy)
        i += 1

    i = 0
    j = 0
    while(i < m + 1):
        while(j < (m/2 + 1)):
            if (PQnum[i][j] != 0.0):
                PQentropy = (PQnum[i][j] / N) * -(log(PQnum[i][j] / N))
                PQ.append(PQentropy)
            j += 1
        j = 0
        i += 1

    entropyP = sum(P)
    entropyQ = sum(Q)
    entropyPQ = sum(PQ)
    print ("R = " + str(r) + ' ' + str(entropyP) + '  ' + str(entropyQ) + '	' + str(entropyPQ))
    return entropyP,entropyQ,entropyPQ

def Savenpy(name,Pentropy,Qentropy,PQentropy,Pmax,Qmax,PQmax):
    """
    Saving the data to file in .npy format
    :param name: 
    :param Pentropy: 
    :param Qentropy: 
    :param PQentropy: 
    :param Pmax: 
    :param Qmax: 
    :param PQmax: 
    :return: 
    """
    np.save('move/' + name + ' move/' + name + '_Pmax.npy',Pmax)
    np.save('move/' + name + ' move/' + name + '_Qmax.npy', Qmax)
    np.save('move/' + name + ' move/' + name + '_PQmax.npy', PQmax)

    np.save('move/' + name + ' move/' + name + '_Pentropy.npy', Pentropy)
    np.save('move/' + name + ' move/' + name + '_Qentropy.npy', Qentropy)
    np.save('move/' + name + ' move/' + name + '_PQentropy.npy', PQentropy)
    return


def Result(name):
    """
    output module in pipeline 
    :param name: 
    :return: 
    """
    ymajorFormatter = FormatStrFormatter('%1.6f')
    isExists = os.path.exists('figure/' + name + ' fig/plot/')
    if (isExists == False):
        os.makedirs('figure/' + name + ' fig/plot/')

    Pmax = np.load('move/' + name + ' move/' + name + '_Pmax.npy')
    Qmax = np.load('move/' + name + ' move/' + name + '_Qmax.npy')
    PQmax = np.load('move/' + name + ' move/' + name + '_PQmax.npy')

    Pentropy = np.load('move/' + name + ' move/' + name + '_Pentropy.npy')
    Qentropy = np.load('move/' + name + ' move/' + name + '_Qentropy.npy')
    PQentropy = np.load('move/' + name + ' move/' + name + '_PQentropy.npy')

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "_maximal_all")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(Pmax, color='blue', label='MAX_P')
    plt.plot(Qmax, color='red', label='MAX_Q')
    plt.plot(PQmax, color='green', label='MAX_PQ')
    ax1.legend(loc=0, bbox_to_anchor=(0.6, 0.55))
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_maximum_all.png')
    plt.close(fig1)

    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "_maximal_P")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(Pmax, color='blue', label='MAX_P')
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_maximum_P.png')
    plt.close(fig1)

    fig1 = plt.figure(3)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "_maximal_Q")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(Qmax, color='blue', label='MAX_Q')
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_maximum_Q.png')
    plt.close(fig1)

    fig1 = plt.figure(4)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "_maximal_PQ")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(PQmax, color='blue', label='MAX_PQ')
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_maximum_PQ.png')
    plt.close(fig1)

    fig1 = plt.figure(5)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "entropy_PQ")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(Pentropy, color='blue', label='ENTROPY_P')
    plt.plot(Qentropy, color='red', label='ENTROPY_Q')
    plt.plot(PQentropy, color='green', label='ENTROPY_PQ')
    ax1.legend(loc=0, bbox_to_anchor=(0.6, 0.55))
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_entropy_all.png')
    plt.close(fig1)

    fig1 = plt.figure(6)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "_entropy_P")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(Pentropy, color='blue', label='ENTROPY_P')
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_entropy_P.png')
    plt.close(fig1)

    fig1 = plt.figure(7)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "_entropy_Q")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(Qentropy, color='blue', label='ENTROPY_Q')
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_entropy_Q.png')
    plt.close(fig1)

    fig1 = plt.figure(8)
    ax1 = fig1.add_subplot(111)
    ax1.set_title(name + "_entropy_PQ")
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    plt.plot(PQentropy, color='blue', label='ENTROPY_PQ')
    plt.savefig('figure/' + name + ' fig/plot/' + name + '_entropy_PQ.png')
    plt.close(fig1)

    dPmax = max(Pmax) - min(Pmax)
    dQmax = max(Qmax) - min(Qmax)
    dPQmax = max(PQmax) - min(PQmax)

    avgPmax = np.average(Pmax)
    avgQmax = np.average(Qmax)
    avgPQmax = np.average(PQmax)

    rPmax = dPmax / avgPmax
    rQmax = dQmax / avgQmax
    rPQmax = dPQmax / avgPQmax

    dPentropy = max(Pentropy) - min(Pentropy)
    dQentropy = max(Qentropy) - min(Qentropy)
    dPQentropy = max(PQentropy) - min(PQentropy)

    avgPentropy = np.average(Pentropy)
    avgQentropy = np.average(Qentropy)
    avgPQentropy = np.average(PQentropy)

    rPentropy = dPentropy / avgPentropy
    rQentropy = dQentropy / avgQentropy
    rPQentropy = dPQentropy / avgPQentropy

    avgPmax = round(avgPmax, 5)
    avgQmax = round(avgQmax, 5)
    avgPQmax = round(avgPQmax, 5)

    dPmax = round(dPmax, 5)
    dQmax = round(dQmax, 5)
    dPQmax = round(dPQmax, 5)

    rPmax = round(rPmax, 5)
    rQmax = round(rQmax, 5)
    rPQmax = round(rPQmax, 5)

    avgPentropy = round(avgPentropy, 5)
    avgQentropy = round(avgQentropy, 5)
    avgPQentropy = round(avgPQentropy, 5)

    dPentropy = round(dPentropy, 5)
    dQentropy = round(dQentropy, 5)
    dPQentropy = round(dPQentropy, 5)

    rPentropy = round(rPentropy, 5)
    rQentropy = round(rQentropy, 5)
    rPQentropy = round(rPQentropy, 5)

    fw = open(name + '.txt', 'w+')

    title = "{:<15}".format(name) + "{:<15}".format('Px') + "{:<15}".format('Qx')+"{:<15}".format('PQx')+"{:<15}".format('dP')+"{:<15}".format('dQ')+"{:<15}".format('dPQ')+"{:<15}".format('PxR') + "{:<15}".format('QxR')+"{:<15}".format('PQxR')
    maximal = "{:<15}".format('maximal') + "{:<15}".format(str(avgPmax)) + "{:<15}".format(str((avgQmax))) + "{:<15}".format(str(avgPQmax)) + "{:<15}".format(str(dPmax)) + "{:<15}".format(str(dQmax)) + "{:<15}".format(str(dPQmax)) + "{:<15}".format(str(rPmax)) + "{:<15}".format(str(rQmax)) + "{:<15}".format(str(rPQmax))
    entropy = "{:<15}".format('entropy') + "{:<15}".format(str(avgPentropy)) + "{:<15}".format(str((avgQentropy))) + "{:<15}".format(str(avgPQentropy)) + "{:<15}".format(str(dPentropy)) + "{:<15}".format(str(dQentropy)) + "{:<15}".format(str(dPQentropy)) + "{:<15}".format(str(rPentropy)) + "{:<15}".format(str(rQentropy)) + "{:<15}".format(str(rPQentropy))

    fw.write(title)
    fw.write('\n')
    fw.write(maximal)
    fw.write('\n')
    fw.write(entropy)
    fw.close()
    return

def Pipeline(name,m):
    """
    Pipeline mode
    :param name:
    :param m:
    :return:
    """
    Pmax = []
    Qmax = []
    PQmax = []

    Pentropy = []
    Qentropy = []
    PQentropy = []

    for i in range(m + 1):
        r = i
        P,Q,Pnum,Qnum,PQnum,N = Measurement(name,r,m)
        Projection(name,r,m,P,Q,Pnum,Qnum,PQnum,N)
        maxP, maxQ, maxPQ = Maximal(Pnum, Qnum, PQnum, N)
        entropyP,entropyQ,entropyPQ = Entropy(Pnum,Qnum,PQnum,N,r,name,m)


        Pmax.append(maxP)
        Qmax.append(maxQ)
        PQmax.append(maxPQ)

        Pentropy.append(entropyP)
        Qentropy.append(entropyQ)
        PQentropy.append(entropyPQ)

    Savenpy(name,Pentropy,Qentropy,PQentropy,Pmax,Qmax,PQmax)
    return

def Main():
    """
    Main entrance
    :return:
    """
    name = 'test'
    m = 128
    Shift(name, m)  # 分段移位模块
    Pipeline(name, m)  # 变值测量、投影、高维统计模块
    Result(name)  # 结果输出模块

if __name__ == "__main__":
    Main()


