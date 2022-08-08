M = []


class DFS_hungary():

    def __init__(self, nx, ny, edge, cx, cy, visited):
        self.nx, self.ny = nx, ny
        self.edge = edge
        self.cx, self.cy = cx, cy
        self.visited = visited

    def max_match(self):  # X序列循环匹配，从A-B-C-D...
        res = 0
        for i in self.nx:
            print(f'{i},状态：{self.cx[i]}') #遍历取nx的每个值
            if self.cx[i] == -1:#未连接
                for key in self.ny:  # 将visited置0表示未访问过
                    self.visited[key] = 0
                res += self.path(i)
                # print(nx, ny) #一致
                # print(edge)
                # print(cx, cy)
                # print(visited)
                print('This is M:', M)
        return res, M

    def path(self, u):  # 寻找增广路径
        for v in self.ny:#有return 因此AG没有判断
            if self.edge[u][v] and (not self.visited[v]):
                self.visited[v] = 1
                print(f'---{u}{v}')
                print(f"visited",self.visited)
                print (f"cx : {self.cx}  ,cy : {self.cy}")
                print('---')
                if self.cy[v] == -1:
                    self.cx[u] = v
                    self.cy[v] = u
                    M.append((u, v))
                    return 1
                else:
                    print (f'remove : {self.cy[v]},{v}')
                    M.remove((self.cy[v], v))
                    if self.path(self.cy[v]):  # 递归,返回1说明找到对应的连接点
                        self.cx[u] = v
                        self.cy[v] = u
                        M.append((u, v)) #补充当前有冲突的结点
                        return 1
        return 0


def DFS_hungary_sample():
    nx, ny = ['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']
    edge = {'A': {'E': 1, 'F': 0, 'G': 1, 'H': 0}, 'B': {'E': 0, 'F': 1, 'G': 0, 'H': 1},
            'C': {'E': 1, 'F': 0, 'G': 0, 'H': 1}, 'D': {'E': 0, 'F': 0, 'G': 1, 'H': 0}}  # 1 表示可以匹配， 0 表示不能匹配
    cx, cy = {'A': -1, 'B': -1, 'C': -1, 'D': -1}, {'E': -1, 'F': -1, 'G': -1, 'H': -1}
    visited = {'E': 0, 'F': 0, 'G': 0, 'H': 0}

    print(DFS_hungary(nx, ny, edge, cx, cy, visited).max_match())



def BFS_hungary(g,Nx,Ny,Mx,My,chk,Q,prev):
    res=0
    for i in range(Nx):
        print('第几轮：',i)
        if Mx[i]==-1:
            qs=qe=0
            print('qs:',qs,'qe:',qe)
            Q[qe]=i
            print('qe:',qe,'Q:',Q)
            qe+=1
            print('qe:',qe)
            prev[i]=-1
            print('prev',prev)
            flag=0
            while(qs<qe and not flag):
                u=Q[qs]
                print('u:',u)
                for v in range(Ny):
                    print('----------')
                    if flag:continue
                    if g[u][v] and chk[v]!=i:
                        chk[v]=i
                        print('v:',v,'chk:',chk)
                        Q[qe]=My[v]
                        print('qe',qe,'Q',Q)
                        qe+=1
                        print('qe',qe)
                        if My[v]>=0:
                            prev[My[v]]=u
                            print('prev',prev)
                            print('lala')
                        else:
                            flag=1
                            d,e=u,v
                            print('d:',d,'e:',e)
                            while d!=-1:
                                t=Mx[d]
                                print('t:',t)
                                Mx[d]=e
                                print('d:',d,'Mx',Mx)
                                My[e]=d
                                print('e:',e,'My:',My)
                                d=prev[d]
                                print('d:',d)
                                e=t
                                print('e:',e)
                    print('-----')
                qs+=1
            if Mx[i]!=-1:
                res+=1
        print('Mx:',Mx,'My:',My,'chk:',chk,'Q:',Q,'prev',prev)
        print('----------------------------------------------------------------------------------')
    return res



def BFS_hungary_sample():
    g = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0]] #x每个元素匹配y的关系
    Nx = 4
    Ny = 4
    Mx = [-1, -1, -1, -1]
    My = [-1, -1, -1, -1]
    chk = [-1, -1, -1, -1]
    Q = [0 for i in range(5)]
    # 原代码
    # Q=[0 for i in range(100)]
    prev = [0, 0, 0, 0]
    print(BFS_hungary(g, Nx, Ny, Mx, My, chk, Q, prev))


    '''
    Mx 为以X序列对应的Y的序号
    My 为以Y序列对应的X的序号
    chk,  v   该轮对应的匹配关系以Y为底
    qe的数目代表v运行的次数，qe=1，代表v循环了一次。
      qe=0  qe=1    qe=2
    Q[ u,   My[v],  My[v],  0  ,   0]
    prev[，，，]代表是否访问过  -1，,-1，-1，-1  代表此前没有访问过，2 代表访问过两次。有两个X访问。
    '''




if __name__ == '__main__':
    #DFS_hungary_sample()
    BFS_hungary_sample()


'''
匈牙利算法
https://blog.csdn.net/qq_40297851/article/details/104991934
nx,ny表示X、Y序列
edge 表示匹配初始关系
cx、cy表示匹配过程中的配对关系
visited 表示是否访问过。？？？
在此过程中注意return的使用位置，很有讲究。
'''
'''
运行结果
A,状态：-1
---AE
visited {'E': 1, 'F': 0, 'G': 0, 'H': 0}
cx : {'A': -1, 'B': -1, 'C': -1, 'D': -1}  ,cy : {'E': -1, 'F': -1, 'G': -1, 'H': -1}
---
This is M: [('A', 'E')]
B,状态：-1
---BF
visited {'E': 0, 'F': 1, 'G': 0, 'H': 0}
cx : {'A': 'E', 'B': -1, 'C': -1, 'D': -1}  ,cy : {'E': 'A', 'F': -1, 'G': -1, 'H': -1}
---
This is M: [('A', 'E'), ('B', 'F')]
C,状态：-1
---CE
visited {'E': 1, 'F': 0, 'G': 0, 'H': 0}
cx : {'A': 'E', 'B': 'F', 'C': -1, 'D': -1}  ,cy : {'E': 'A', 'F': 'B', 'G': -1, 'H': -1}
---
remove : A,E
---AG
visited {'E': 1, 'F': 0, 'G': 1, 'H': 0}
cx : {'A': 'E', 'B': 'F', 'C': -1, 'D': -1}  ,cy : {'E': 'A', 'F': 'B', 'G': -1, 'H': -1}
---
This is M: [('B', 'F'), ('A', 'G'), ('C', 'E')]
D,状态：-1
---DG
visited {'E': 0, 'F': 0, 'G': 1, 'H': 0}
cx : {'A': 'G', 'B': 'F', 'C': 'E', 'D': -1}  ,cy : {'E': 'C', 'F': 'B', 'G': 'A', 'H': -1}
---
remove : A,G
---AE
visited {'E': 1, 'F': 0, 'G': 1, 'H': 0}
cx : {'A': 'G', 'B': 'F', 'C': 'E', 'D': -1}  ,cy : {'E': 'C', 'F': 'B', 'G': 'A', 'H': -1}
---
remove : C,E
---CH
visited {'E': 1, 'F': 0, 'G': 1, 'H': 1}
cx : {'A': 'G', 'B': 'F', 'C': 'E', 'D': -1}  ,cy : {'E': 'C', 'F': 'B', 'G': 'A', 'H': -1}
---
This is M: [('B', 'F'), ('C', 'H'), ('A', 'E'), ('D', 'G')]
(4, [('B', 'F'), ('C', 'H'), ('A', 'E'), ('D', 'G')])

'''


