"""
Authors: Sahruday Patti

Brief : Implementation of Multi RRT* Alogoritm 

Course:  Planning for Autonomous Robots - ENPM661  [Proj5]

Date: 05/08/2022

"""
# Importing Required Libraries
import numpy as np
import random
import math
import os
from scipy.interpolate import splprep, splev, splrep
import matplotlib.cm as cm
import matplotlib.pyplot as plt


plt.ion()
# Executing Mult RRT* Algorithm 
class rrtstar:
    class node:
        def __init__(self, x, y, t=0, cost = 0, node_iter = 0):
            self.x = x
            self.y = y
            self.t = t
            self.parent = None
            self.x_path = None
            self.y_path = None
            self.cost = cost
            self.node_iter = node_iter

    def __init__(self, start, goal, s):
        self.start_node = self.node(start[0], start[1])
        self.goal_node = self.node(goal[0], goal[1])
        self.nodes = [self.start_node]
        self.lower_lim_x = 0
        self.lower_lim_y = 0
        self.upper_lim_x = 10
        self.upper_lim_y = 10 
        self.neigh_dist = 0.6
        self.vel = 0.2    # robot vel 
        self.r = 0.1 # robot rad
        self.c = 0.4     # robot clearance
        self.s = s
        self.thresh = self.c + self.r
        self.nodes_at_t = {}
        self.other_traj = []
        self.iter = 0
        
    # Checking for collision and obstacles
    def collision_checking(self, node):

    
        ret = False

        if node.x - self.thresh < self.lower_lim_x:
            ret = True
        elif node.x + self.thresh > self.upper_lim_x:
            ret = True
        elif node.y - self.thresh < self.lower_lim_y:
            ret = True
        elif node.y + self.thresh > self.upper_lim_y:
            ret = True

        if node.x > 4 - self.thresh and node.x < 6 + self.thresh and node.y > 3.5 - self.thresh and node.y < 7.5 + self.thresh:
            ret = True

        for traj in self.other_traj:
            t = node.t
            if node.t > len(traj[0])-1:
                t = len(traj[0])-1

            if np.sqrt((node.x - traj[0][t])**2 + (node.y - traj[1][t])**2) < self.s:
                ret = True

        return ret
    
    # Checking for goal node
    def check_for_goal(self, node):
        if self.get_dist(node, self.goal_node) < 0.6:
            return True

        return False
    
    # Genarating random nodes in the environment
    def generate_random_node(self):
        x = random.randint(1, 10)
        y = random.randint(1, 10)

        new_node = self.node(x, y)

        return new_node

    def get_dist(self, node1, node2):
        return math.sqrt((node1.x -node2.x)**2 + (node1.y - node2.y)**2)

    def get_nearest_node(self, rand_node, nodes):
        nearest_node_idx = 0
        min_dist = float('inf')

        for i, node in enumerate(nodes):
            dist = self.get_dist(rand_node, node)
            if dist < min_dist:
                nearest_node_idx = i
                min_dist = dist

        return nearest_node_idx

    def next_step(self, parent, dest_node):
        par_x = parent.x
        par_y = parent.y
        tm = parent.t

        dest_x = dest_node.x
        dest_y = dest_node.y

        theta = np.arctan2((dest_y-par_y), (dest_x-par_x))

        count = 0
        x_path = []
        y_path = []
        x = par_x
        y = par_y
        dist = 0

        while(count < 10): 
            dx = 0.1 * math.cos(theta) * self.vel
            dy = 0.1 * math.sin(theta) * self.vel
            x = x + dx
            y = y + dy

            dist = dist + np.sqrt(dx**2 + dy**2)
            
            if self.collision_checking(self.node(x, y, t=tm+count+1)):
                return None

            x_path.append(x)
            y_path.append(y)
            count = count + 1
        setter = 0
        if self.iter == 1:
            setter = 1
        elif self.iter == 2:
            setter = 2
        new_node = self.node(x, y, t=parent.t+10)
        new_node.parent = parent
        new_node.x_path = x_path
        new_node.y_path = y_path
        new_node.cost = parent.cost + dist
        new_node.node_iter = setter
        return new_node

    def adding_to_node_dict(self, new_node, index):
        t = new_node.t
        nodes = self.nodes_at_t.get(t)

        if nodes == None:
            self.nodes_at_t.update({t:[index]})

        else:
            nodes.append(index)
            self.nodes_at_t.update({t:nodes})

    # Generating Trajectories
    def get_trajectory(self, parent, child):
        dist = self.get_dist(parent, child)

        if (dist*10) % self.vel == 0:
            max_count = (dist*10)/self.vel
        else:
            max_count = (dist*10)/self.vel + 1

        par_x = parent.x
        par_y = parent.y
        t = parent.t

        child_x = child.x
        child_y = child.y

        theta = np.arctan2((child_y-par_y), (child_x-par_x))

        count = 0
        x_path = []
        y_path = []
        x = par_x
        y = par_y
        dist = 0

        if max_count == 0:
            return None, None

        while(count < max_count): 
            dx = 0.1 * math.cos(theta) * self.vel
            dy = 0.1 * math.sin(theta) * self.vel
            x = x + dx
            y = y + dy

            dist = dist + np.sqrt(dx**2 + dy**2)
            
            if self.collision_checking(self.node(x, y, t=t+count+1)):
                return None, None

            x_path.append(x)
            y_path.append(y)
            count = count + 1

        if x != child.x:
            child.x = x
        if y != child.y:
            child.y = y

        return x_path, y_path
    
    # Getting the neighbouring nodes
    def get_adj_nodes(self, new_node):
        ngh_indx = []

        for i, node in enumerate(self.nodes):
            dist = self.get_dist(new_node, node)
            if dist <= self.neigh_dist:
                ngh_indx.append(i)

        return ngh_indx
    
    # Assigning parent nodes
    def assign_parent(self, new_node, ngh_indx):
        for i in ngh_indx:
            dist = self.get_dist(new_node, self.nodes[i])
            if self.nodes[i].cost + dist < new_node.cost:
                x_path, y_path = self.get_trajectory(self.nodes[i], new_node)
                if x_path == None:
                    continue
                new_node.t = self.nodes[i].t + len(x_path)
                new_node.x_path = x_path
                new_node.y_path = y_path
                new_node.cost = self.nodes[i].cost + dist
                new_node.parent = self.nodes[i]
                
    # Deleting Child Nodes
    def removing_chiildren(self, parent):
        for idx, node in enumerate(self.nodes):
            if node.parent == parent:
                del self.nodes[idx]
                self.removing_chiildren(node)
                idx = idx-1
                
    # Cost function 
    def Cost_Function(self, parent):
        for i, node in enumerate(self.nodes):
            if node.parent == parent:
                dist = self.get_dist(parent, node)
                node.cost = parent.cost + dist
                node.t = parent.t + len(node.x_path)
                if self.collision_checking(node):
                    del self.nodes[i]
                    self.removing_chiildren(node)
                    i = i-1
                else:
                    self.Cost_Function(self.node)

    def rewiring(self, new_node, ngh_indx):
        new_path_x = []
        new_path_y = []

        for i in ngh_indx:
            dist = self.get_dist(new_node, self.nodes[i])
            if new_node.cost + dist < self.nodes[i].cost:
                x_path, y_path = self.get_trajectory(new_node, self.nodes[i])
                if x_path == None:
                    continue
                self.nodes[i].t = new_node.t + len(x_path)
                self.nodes[i].x_path = x_path
                self.nodes[i].y_path = y_path
                self.nodes[i].cost = new_node.cost + dist
                self.nodes[i].parent = new_node
                self.Cost_Function(self.nodes[i])
                new_path_x.append(x_path)
                new_path_y.append(y_path)

        return new_path_x, new_path_y
    # Implementing back tracking
    def backtracing(self, cur_node):
        if(cur_node.parent == None):
            return np.asarray([cur_node.x]), np.asarray([cur_node.y]), np.asarray([cur_node.t]), np.asarray([cur_node.x]), np.asarray([cur_node.x])

        x, y, t, path_x, path_y = self.backtracing(cur_node.parent)

        x_s = np.hstack((x, cur_node.x))
        y_s = np.hstack((y, cur_node.y))
        t_s = np.hstack((t, cur_node.t))
        path_x = np.hstack((path_x, cur_node.x_path))
        path_y = np.hstack((path_y, cur_node.y_path))


        return x_s, y_s, t_s, path_x, path_y
    
    # Generating Curved Paths
    def smoothing(self, res, ax, text_name, img_name):
        t = res.t

        # backtracing path 
        x_s, y_s, t_s, path_x, path_y = self.backtracing(res)

        step = 5*float(1/float(t))
        m = len(x_s)

        # Path smoothing
        m = m - math.sqrt(2*m)
        tck, u = splprep([x_s, y_s], s=m)
        u_s = np.arange(0, 1.01, step)
        new_points = splev(u_s, tck)
        new_points = np.asarray(new_points)

        # Plotting trajectories
        ax.plot(x_s, y_s, color = 'r', linewidth = 1.5)
        ax.plot(new_points[0], new_points[1], label="S", color = 'c', linewidth = 1.5)

        plt.savefig(img_name)

        # Saving data in text files
        out = new_points.T
        if os.path.exists(text_name):
            os.remove(text_name)
        f1 = open(text_name, "a")

        for i in range(len(out)):
            np.savetxt(f1, out[i], fmt="%s", newline=' ')
            f1.write("\n")

        return new_points

    def replot(self, ax):
        count = 0
        c = ['y','b','g']
        v = c[2]
        
        for n in self.nodes:
            if count == 0:
                count += 1
                continue
            if n.node_iter == 0:
                v = c[2]
            elif n.node_iter ==1:
                v = c[0]
            elif n.node_iter ==2:
                v = c[1]
            cir_node = plt.Circle((n.x, n.y), 0.02, fill=True, color = 'r')
            ax.add_patch(cir_node)
            ax.plot(n.x_path, n.y_path, color = v, linewidth = 1)

        for traj in self.other_traj:
            ax.plot(traj[0], traj[1], color = 'b', linewidth = 1)


    def plan(self, text_name, img_name, replan = False):
        if self.collision_checking(self.start_node):
            print("Start node present inside obstacle")
            exit()

        if self.collision_checking(self.goal_node):
            print("Goal node present inside obstacle")
            exit()

        fig, ax = plt.subplots()
        ax.set_xlim([0,10])
        ax.set_ylim([0,10])

        cir_start = plt.Circle((self.start_node.x, self.start_node.y), 0.18, fill=True, color = 'b')
        cir_goal = plt.Circle((self.goal_node.x, self.goal_node.y), 0.18, fill=True, color = 'c')

        ax.add_patch(cir_start)
        ax.add_patch(cir_goal)

        obs = plt.Rectangle((4,3.5),2,4,fill=True, color='k')
        ax.add_patch(obs)

        c = ['y','b','g']
        v = c[2]
        if replan:
            self.replot(ax)
            if self.iter == 1:
                v = c[0]
            if self.iter ==2:
                v = c[1]

        count = 0
        l = []
        while (True):
            rand_node = self.generate_random_node()
            if rand_node == None:
                continue

            nearest_node_idx = self.get_nearest_node(rand_node, self.nodes)
            new_node = self.next_step(self.nodes[nearest_node_idx], rand_node)

            # if collision detected, continue
            if new_node == None:
                continue

            ngh_indx = self.get_adj_nodes(new_node)
            self.assign_parent(new_node, ngh_indx)
            new_path_x, new_path_y = self.rewiring(new_node, ngh_indx)
            self.nodes.append(new_node)
            index = len(self.nodes)-1
            self.adding_to_node_dict(new_node, index)

            cir_node = plt.Circle((new_node.x, new_node.y), 0.02, fill=True, color = 'r')
            ax.add_patch(cir_node)
            ax.plot(new_node.x_path, new_node.y_path, color = v, linewidth = 1)

            for i in range(len(new_path_x)):
                ax.plot(new_path_x[i], new_path_y[i], color = v, linewidth = 1)

            plt.pause(0.01)

            if self.check_for_goal(new_node):
                print("Goal reached!!")
                for i in self.nodes:
                    l.append((i.x,i.y,i.x_path,i.y_path,i.node_iter))
                break

            count = count + 1

        traj = self.smoothing(new_node, ax, text_name, img_name)

        return traj,l

    def prune(self, t):
        indx = np.asarray([])
        for key in self.nodes_at_t.keys():
            if key >= t:
                i = np.asarray(self.nodes_at_t.get(key))
                indx = np.hstack((indx, i))
                self.nodes_at_t.update({key:[]})

        indx = np.asarray(indx, dtype='int')

        for idx in sorted(indx, reverse=True):
            del self.nodes[idx]

    def replan(self, trajs, t, text_name, img_name):
        self.other_traj = trajs
        self.prune(t)
        traj,l = self.plan(text_name, img_name, replan = True)
        return traj,l

def replan_checking(traj1, traj2, s):
    l = min(len(traj1[0]), len(traj2[0]))
    L = max(len(traj1[0]), len(traj2[0]))

    col = False

    for i in range(l):
        dist = np.sqrt((traj1[0][i]-traj2[0][i])**2 + (traj1[1][i]-traj2[1][i])**2)
        # print(dist)
        if dist < (s):
            print("Collision identified at: " + str(i))
            # print(dist)
            col = True
            break

    step = -1
    if not col:
        flag = False
        if len(traj1[0]) == l:
            flag = True
        for i in range(l, L):
            if flag:
                dist = np.sqrt((traj1[0][l-1]-traj2[0][i])**2 + (traj1[1][l-1]-traj2[1][i])**2)
            else:
                dist = np.sqrt((traj1[0][i]-traj2[0][l-1])**2 + (traj1[1][i]-traj2[1][l-1])**2)
            if dist < s:
                print("Collision identified at: " + str(i))
                # print(dist)
                col = True
                step = i
                break

    return col, step

def check_for_replanning(traj1, traj2, traj3, s, flag):
    if not flag:
        s = s - 0.1

    col12, t12 = replan_checking(traj1, traj2, s)
    col23, t23 = replan_checking(traj2, traj3, s)
    col31, t31 = replan_checking(traj1, traj3, s)

    col = [False, False, False]
    t = [-1, -1, -1]

    if col12 or col31:
        col[0] = True
        if not col12:
            t[0] = t31
        elif not col31:
            t[0] = t12
        else:
            t[0] = min(t12, t31)

    if col12 or col23:
        col[1] = True
        if not col12:
            t[1] = t23
        elif not col23:
            t[1] = t12
        else:
            t[1] = min(t12, t23)

    if col23 or col31:
        col[2] = True
        if not col23:
            t[2] = t31
        elif not col31:
            t[2] = t23
        else:
            t[2] = min(t23, t31)

    return col, t

def main():
    # start nodes (x, y) of 3 Agents 
    start1 = [3, 3]
    start2 = [2, 2]
    start3 = [1, 1]

    # goal nodes (x, y) of 3 Agents
    goal1 = [7, 6]
    goal2 = [7, 4]
    goal3 = [7, 7]

    if not os.path.exists('Output_File'):
        
        os.makedirs('Output_File')

    
    s = 0.5
    all_nodes_1 = []
    all_nodes_2 = []
    all_nodes_3 = []
    
    # Initial Planning
    rrt_star1 = rrtstar(start1, goal1, s=s)
    traj1,l1 = rrt_star1.plan("Output_File/Traj1.txt", "Output_File/Explored1.png")
    all_nodes_1.append(l1)

    rrt_star2 = rrtstar(start2, goal2, s=s)
    traj2,l2 = rrt_star2.plan("Output_File/Traj2.txt", "Output_File/Explored2.png")
    all_nodes_2.append(l2)

    rrt_star3 = rrtstar(start3, goal3, s=s)
    traj3,l3 = rrt_star3.plan("Output_File/Traj3.txt", "Output_File/Explored3.png")
    all_nodes_3.append(l3)

    # Plotting planned trajectories
    fig, ax = plt.subplots()
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.plot(traj1[0], traj1[1], color = 'b', linewidth = 1)
    ax.plot(traj2[0], traj2[1], color = 'r', linewidth = 1)
    ax.plot(traj3[0], traj3[1], color = 'g', linewidth = 1)
    plt.savefig("Output File/Plan0.png")

    replan = [True, True, True]
    traj_all = [traj1, traj2, traj3]
    rrt_star = [rrt_star1, rrt_star2, rrt_star3]

   
    flag = True

    for i in range(2):
        col, t = check_for_replanning(traj1, traj2, traj3, s, flag)
        flag = False
        print("Collision: ", col)
        pd1 = float('inf')
        pd2 = float('inf')
        pd3 = float('inf')

        if col[0] and replan[0]:
            rrt_star1.iter+=1
            new_traj1,l1 = rrt_star1.replan([traj2, traj3], t[0], "Output_File/Replanned"+str(i)+"_1.txt", "Output_File/Re_Explored"+str(i)+"_1.png")
            pd1 = (len(new_traj1[0])-len(traj1[0]))/float(len(traj1[0]))*100
            print("Appended")
            all_nodes_1.append(l1)

        if col[1] and replan[1]:
            rrt_star2.iter+=1
            new_traj2,l2 = rrt_star2.replan([traj1, traj3], t[1], "Output_File/Replanned"+str(i)+"_2.txt", "Output_File/Re_Explored"+str(i)+"_2.png")
            pd2 = (len(new_traj2[0])-len(traj2[0]))/float(len(traj2[0]))*100
            all_nodes_2.append(l2)

        if col[2] and replan[2]:
            rrt_star3.iter+=1
            new_traj3,l3 = rrt_star3.replan([traj1, traj2], t[2], "Output_File/Replanned"+str(i)+"_3.txt", "Output_File/re_explored"+str(i)+"_3.png")
            pd3 = (len(new_traj3[0])-len(traj3[0]))/float(len(traj3[0]))*100
            all_nodes_3.append(l3)

        m = min(pd1, pd2, pd3)

        if m == float('inf'):
            print("Final trajectories found at iteration: " + str(i+1))
            break

        if m == pd1:
            print("Trajectory 1 changed at iteration " + str(i+1))
            traj1 = new_traj1
            replan[0] = False

        elif m == pd2:
            print("Trajectory 2 changed at iteration " + str(i+1))
            traj2 = new_traj2
            replan[1] = False

        else:
            print("Trajectory 3 changed at iteration " + str(i+1))
            traj3 = new_traj3
            replan[2] = False

        fig6, ax6 = plt.subplots()
        ax6.set_xlim([0,10])
        ax6.set_ylim([0,10])
        ax6.plot(traj1[0], traj1[1], color = 'b', linewidth = 1)
        ax6.plot(traj2[0], traj2[1], color = 'r', linewidth = 1)
        ax6.plot(traj3[0], traj3[1], color = 'g', linewidth = 1)
        plt.savefig("Output_File/Plan"+str(i+1)+".png")

    fig2, ax2 = plt.subplots()
    ax2.set_xlim([0,10])
    ax2.set_ylim([0,10])
    from cycler import cycler
    obs = plt.Rectangle((4,3.5),2,4,fill=True, color='b')
    ax2.add_patch(obs)
    
    L1 = len(traj1[0])
    L2 = len(traj2[0])
    L3 = len(traj3[0])
    L = max(L1, L2, L3)
    ax2.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']))

    for i in range(L1-1):
        ax2.plot(traj1[0][i:i+2], traj1[1][i:i+2])
    
    cm = plt.get_cmap('plasma')
  

    ax2.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) )
    for i in range(L2-1):
        ax2.plot(traj2[0][i:i+2], traj2[1][i:i+2])

    cm = plt.get_cmap('plasma')
  
    ax2.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']))
    for i in range(L3-1):
        ax2.plot(traj3[0][i:i+2], traj3[1][i:i+2])
    
    plt.savefig("Output_File/final_traj.png")

    tr = [traj1, traj2, traj3]
    fig, ax9 = plt.subplots()
    ax9.set_xlim([0,10])
    c = ['b','g','y']
    flag2 = 0
    print("length",len(all_nodes_1))
    for j in all_nodes_1:
        flag2+=1
        flag = 0
        if flag2 == 1:
            v = c[0]
        elif flag2 == 2:
            v = c[1]
        else:
            v = c[2]
        for i in j:
            if i[4] == 0:
                v = c[1]
            elif i[4] ==1:
                v = c[2]
            else:
                v = c[0]
            cir_node = plt.Circle((i[0], i[1]), 0.02, fill=True, color = 'r')
            ax9.add_patch(cir_node)
            if (flag ==1):
                ax9.plot(i[2], i[3], color = v, linewidth = 1)
            flag =1
    plt.savefig("Output_File/AllExploredNodes_Turtlebot1"+".png")
    plt.show()

    fig, ax9 = plt.subplots()
    ax9.set_xlim([0,10])
    ax9.set_ylim([0,10])
    flag = 0
    c = ['b','g','y']
    flag2 = 0
    print("length",len(all_nodes_2))
    for j in all_nodes_2:
        flag2+=1
        flag = 0
        if flag2 == 1:
            v = c[0]
        elif flag2 == 2:
            v = c[1]
        else:
            v = c[2]
        for i in j:
            if i[4] == 0:
                v = c[1]
            elif i[4] ==1:
                v = c[2]
            else:
                v = c[0]
            cir_node = plt.Circle((i[0], i[1]), 0.02, fill=True, color = 'r')
            ax9.add_patch(cir_node)
            if (flag ==1):
                ax9.plot(i[2], i[3], color = v, linewidth = 1)
            flag =1
    plt.savefig("Output_File/AllExploredNodes_Turtlebot2"+".png")
    plt.show()

    fig, ax9 = plt.subplots()
    ax9.set_xlim([0,10])
    ax9.set_ylim([0,10])
    flag = 0
    c = ['b','g','y']
    flag2 = 0
    print("length",len(all_nodes_3))
    for j in all_nodes_3:
        flag2+=1
        flag = 0
        if flag2 == 1:
            v = c[0]
        elif flag2 == 2:
            v = c[1]
        else:
            v = c[2]
        for i in j:
            if i[4] == 0:
                v = c[1]
            elif i[4] ==1:
                v = c[2]
            else:
                v = c[0]
            cir_node = plt.Circle((i[0], i[1]), 0.02, fill=True, color = 'r')
            ax9.add_patch(cir_node)
            if (flag ==1):
                ax9.plot(i[2], i[3], color = v, linewidth = 1)
            flag =1
    plt.savefig("Output_File/AllExploredNodes_Turtlebot3"+".png")
    plt.show()

    # Saving data in text file
    out1 = traj1.T
    if os.path.exists("Output_File/final_traj1.txt"):
        os.remove("Output_File/final_traj1.txt")
    final1 = open("Output_File/final_traj1.txt", "a")

    for i in range(len(out1)):
        np.savetxt(final1, out1[i], fmt="%s", newline=' ')
        final1.write("\n")

    out2 = traj2.T
    if os.path.exists("Output_File/final_traj2.txt"):
        os.remove("Output_File/final_traj2.txt")
    final2 = open("Output_File/final_traj2.txt", "a")

    for i in range(len(out2)):
        np.savetxt(final2, out2[i], fmt="%s", newline=' ')
        final2.write("\n")

    out3 = traj3.T
    if os.path.exists("Output_File/final_traj3.txt"):
        os.remove("Output_File/final_traj3.txt")
    final3 = open("Output_File/final_traj3.txt", "a")

    for i in range(len(out3)):
        np.savetxt(final3, out3[i], fmt="%s", newline=' ')
        final3.write("\n")

    plt.show()
    plt.pause(15)
    plt.close()

if __name__ == "__main__":
    main()
