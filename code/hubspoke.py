#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hubspoke.py
# Author: Takara Sakai
# Date: 2025-01-28
# Description: This script implements a demand-driven hub-and-spoke model.

import numpy as np
import pandas as pd
import networkx as nx
from heapq import heappush, heappop
import matplotlib.pyplot as plt



class Parameter:
    def __init__(self, phi, rho, t, nu, d, N):
        self.phi = phi
        self.rho = rho
        self.t   = t
        self.nu  = nu
        self.d   = d
        self.N   = N # Number of nodes
        self.L   = (N-1)*(N-1) # Number of links

class Network:
    def __init__(self, prm):
        self.prm = prm
        _ = self.make_data()

    def make_data(self):
        self.Node2Link = {}
        self.Link2Node = {}
        self.OutLinks  = {i:[] for i in range(1, self.prm.N)}
        self.InLinks   = {i:[] for i in range(self.prm.N)}

        
        self.distance = np.zeros(self.prm.L)
        self.IT_MT_bool = np.ones(self.prm.L)
        
        link_idx = 0
        for from_node_idx in range(1, self.prm.N):
            self.Node2Link[from_node_idx] = {}
            for to_node_idx in range(self.prm.N):
                if from_node_idx == to_node_idx:
                    continue
                else:
                    

                    if abs(from_node_idx - to_node_idx) == 1:
                        # If link is IT, then the value is 0
                        self.IT_MT_bool[link_idx] = 0.0
                        
                    self.distance[link_idx] = abs(from_node_idx - to_node_idx)
                    self.Node2Link[from_node_idx][to_node_idx] = link_idx
                    self.Link2Node[link_idx] = (from_node_idx, to_node_idx)

                    self.OutLinks[from_node_idx] = self.OutLinks[from_node_idx] + [link_idx]
                    self.InLinks[to_node_idx] = self.InLinks[to_node_idx] + [link_idx]
                    
                    link_idx = link_idx + 1
        
        return True
        
class Leaf:
    def __init__(self, prm, net, flow_lower, flow_upper):
        self.prm        = prm
        self.net        = net
        self.flow_lower = flow_lower
        self.flow_upper = flow_upper


    def solve(self):
        # dijkstara
        self.SP_tree = self.dijkstra()
        self.flow = self.generate_flow(self.SP_tree)
        
        self.Z = self.vF(self.flow).sum()
        self.app_Z = self.vG(self.flow).sum()
        
        FminusG = np.array([self.F(*self.net.Link2Node[link_idx], self.flow[link_idx]) - self.G(*self.net.Link2Node[link_idx], self.flow[link_idx]) for link_idx in range(self.prm.L)])
        
        self.max_link_idx = np.argmax(FminusG)

        
        return True

    def dijkstra(self):
        spl        = [np.inf]*self.prm.N
        flag       = [False]*self.prm.N
        SP_tree    = [-1]*self.prm.N
        node_set = set(range(self.prm.N))

        source = 0
        hq = [(0.0, source)]
        spl[source] = 0.0
        flag[source] = True # 探索済に変更

        while hq:
            pivot_node_idx = heappop(hq)[1]
                        
            for from_node_idx in (node_set - set([pivot_node_idx])):
                if flag[from_node_idx] == True:
                    continue
                else:
                    tmp_cost = spl[pivot_node_idx] + self.link_cost(from_node_idx, pivot_node_idx)
                
                    if tmp_cost < spl[from_node_idx]:
                        spl[from_node_idx] = tmp_cost
                        SP_tree[from_node_idx] = pivot_node_idx
                    
                        heappush(hq, (tmp_cost, from_node_idx))
                        
            flag[pivot_node_idx] = True

        return SP_tree

    def generate_flow(self, SP_tree):
        flow = np.zeros(self.prm.L)
        for node_idx in range(1, self.prm.N):
            tr_node_idx = node_idx
            while tr_node_idx != 0:
                link_idx = self.net.Node2Link[tr_node_idx][SP_tree[tr_node_idx]]
                flow[link_idx] = flow[link_idx] + self.prm.nu
                tr_node_idx = SP_tree[tr_node_idx]
        return flow

    # link cost function
    def link_cost(self, from_node_idx, to_node_idx):
        link_idx = self.net.Node2Link[from_node_idx][to_node_idx]

        u = self.flow_upper[link_idx]
        l = self.flow_lower[link_idx]
        Fu = self.F(from_node_idx, to_node_idx, u)
        Fl = self.F(from_node_idx, to_node_idx, l)

        if u==l:
            return 0.0
        else:
            return (Fu- Fl)/(u-l)

    # objective function (element)
    def F(self, from_node_idx, to_node_idx, flow):
        # Integral form
        distance = abs(to_node_idx - from_node_idx)
        if distance == 1:
            return self.prm.t*self.prm.d*distance*flow
        else:
            return (self.prm.phi + self.prm.rho*self.prm.d*distance)*np.log(flow+1.0)

    # linear approximation function (element)
    def G(self, from_node_idx, to_node_idx, flow):
        link_idx = self.net.Node2Link[from_node_idx][to_node_idx]

        u = self.flow_upper[link_idx]
        l = self.flow_lower[link_idx]
        Fu = self.F(from_node_idx, to_node_idx, u)
        Fl = self.F(from_node_idx, to_node_idx, l)

        if u == l:
            return 0.0
        else:
            return flow*((Fu- Fl)/(u-l)) + (u*Fl - l*Fu)/(u-l)


    # objective function (vector)
    def vF(self, flow):
        return (1-self.net.IT_MT_bool)*(self.prm.t*self.prm.d*self.net.distance*flow) + self.net.IT_MT_bool*((self.prm.phi + self.prm.rho*self.prm.d*self.net.distance)*np.log(flow+1.0))

    # linear approximation function (vector)
    def vG(self, flow):
        u = self.flow_upper
        l = self.flow_lower
        Fu = self.vF(u)
        Fl = self.vF(l) 

        # Avoiding division by zero
        u_l = np.vectorize(lambda x : 1.0 if x==0 else x)(u-l)
        return flow*((Fu- Fl)/u_l) + (u*Fl - l*Fu)/(u_l)
    

class BB_model:
    def __init__(self, prm, net):
        self.prm = prm
        self.net = net

        self.opt_Z = np.inf

    def solve(self, init_flow_lower, init_flow_upper):
        k = 0
        init_leaf = Leaf(self.prm, self.net, init_flow_lower, init_flow_upper)
        _ = init_leaf.solve()
        
        app_Z = init_leaf.app_Z
        _ = self.check_opt(init_leaf)
        
        hq = [(app_Z, k)]
        leaves = {}
        leaves[k] = init_leaf

        while hq:
            tr_leaf = leaves[heappop(hq)[1]]
            app_Z = tr_leaf.app_Z

            if (app_Z  < self.opt_Z):

                # 分割 (1)
                k = k+1
                new_upper = tr_leaf.flow_upper.copy()
                new_upper[tr_leaf.max_link_idx] = tr_leaf.flow[tr_leaf.max_link_idx]
                
                new1_leaf = Leaf(self.prm, self.net, tr_leaf.flow_lower.copy(), new_upper)
                _ = new1_leaf.solve()
               
                new1_app_Z = new1_leaf.app_Z
                leaves[k] = new1_leaf
                _ = self.check_opt(new1_leaf)
                heappush(hq, (new1_app_Z, k))

                
                # 分割 (2)
                k = k+1
                new_lower = tr_leaf.flow_lower.copy()
                new_lower[tr_leaf.max_link_idx] = tr_leaf.flow[tr_leaf.max_link_idx]
                
                new2_leaf = Leaf(self.prm, self.net, new_lower, tr_leaf.flow_upper.copy())
                _ = new2_leaf.solve()
               
                new2_app_Z = new2_leaf.app_Z
                leaves[k] = new2_leaf

                _ = self.check_opt(new2_leaf)
                heappush(hq, (new2_app_Z, k))
            else:
                print('k=', k)
                break

        
        return self.opt_Z, self.opt_SP_tree, self.opt_flow

    def check_opt(self, leaf):
        if leaf.Z < self.opt_Z:
            self.opt_Z = leaf.Z
            self.opt_SP_tree = leaf.SP_tree
            self.opt_flow    = leaf.flow
            print('opt_Z=', self.opt_Z)
        return True
    

def generate_flow(prm, net, SP_tree):
    flow = np.zeros(prm.L)
    for node_idx in range(1, prm.N):
        tr_node_idx = node_idx
        while tr_node_idx != 0:
            link_idx = net.Node2Link[tr_node_idx][SP_tree[tr_node_idx]]
            flow[link_idx] = flow[link_idx] + prm.nu
            tr_node_idx = SP_tree[tr_node_idx]
    return flow


# networkxを用いた可視化

def link_plot(prm, net, SP_tree):
    G_IT_link = nx.DiGraph()
    G_MT_link = nx.DiGraph()
    
    # 描画のために位置情報を与える
    pos  = {
        i : (i*35.0, 0.0)
        for i in range(prm.N)
    }

    # 隣合うノードを結ぶリンクはITリンク，それ以外はMTリンク
    for from_node_idx in range(1, prm.N):
        to_node_idx = SP_tree[from_node_idx]
        if abs(from_node_idx - to_node_idx) == 1:
            G_IT_link.add_edge(from_node_idx, to_node_idx)
        else:
            G_MT_link.add_edge(from_node_idx, to_node_idx)

    plt.figure(figsize=(12, 5))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)


    nx.draw_networkx(G_IT_link, 
                     pos=pos, 
                     node_size=500, 
                     edge_color='black',
                     node_color='white', 
                     with_labels=True, 
                     font_family = 'Times New Roman',
                     font_size=15,
                     edgecolors='black',
                     connectionstyle='arc3,rad=0.0')
    nx.draw_networkx(G_MT_link,
                     pos=pos,
                     node_size=500,
                     edge_color='red',
                     node_color='white',
                     with_labels=True,
                     font_family = 'Times New Roman',
                     font_size=15,
                     edgecolors='black',
                     connectionstyle='arc3,rad=0.3')

    return True