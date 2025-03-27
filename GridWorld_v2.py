import numpy as np
import random


# policy为(25state,5action)的二维数组，由determine变为stochastic
# 引入trajectory
class GridWorld_v2(object):

    def __init__(self,rows=5,columns=5,seed=-1,forbiddenNum=4,targetNum=1,forbiddenScore=-1,score=1,otherScore=0,desc=None):
        if desc is not None:
            self.rows=len(desc)
            self.columns=len(desc[0])
            self.forbiddenScore=forbiddenScore
            self.score=score
            self.otherScore=otherScore

            # scoreMap   #为forbidden    T为target
            g=[[forbiddenScore if desc[i][j]=='#' else score if desc[i][j]=='T' else otherScore for j in range(self.columns)] for i in range(self.rows)]
            self.scoreMap=np.array(g)
            self.stateMap=[i for i in range(self.rows*self.columns)]
            self.stateMap=np.array(self.stateMap).reshape(self.rows,self.columns)

        else:
            self.columns=columns
            self.rows=rows
            self.forbiddenScore=forbiddenScore
            self.score=score
            self.otherScore=otherScore

            random.seed(seed)
            self.scoreMap=[[0 for j in range(self.columns)] for i in range(self.rows)]

            self.stateMap = [i for i in range(self.rows * self.columns)]
            self.stateMap = np.array(self.stateMap)
            random.shuffle(self.stateMap)

            for i in range(forbiddenNum):
                i=self.stateMap[i]
                x=i//self.columns
                y=i%columns

                self.scoreMap[x][y]=self.forbiddenScore
            for i in range(targetNum):
                i = self.stateMap[i+forbiddenNum]
                x = i // self.columns
                y = i % columns
                self.scoreMap[x][y] = self.score
            self.stateMap=self.stateMap.resize(self.rows,self.columns)
    def show(self):
        for i in range(self.rows):
            s=''
            for j in range(self.columns):
                if self.scoreMap[i][j]==self.forbiddenScore:
                    s+='🚫'
                elif self.scoreMap[i][j]==self.score:
                    s+='✅'
                else:
                    s+='⬜️'
            print(s)
    def getNewState(self,state,action):
        if state<0 or state>=self.columns*self.rows:
            print('state超出范围')
            return
        if action<0 or action>=5:
            print('aciton超出范围')
            return
        x=state//self.columns
        y=state%self.columns

        actionAll=[(-1,0),(0,1),(1,0),(0,-1),(0,0)]
        new_x=x+actionAll[action][0]
        new_y=y+actionAll[action][1]

        if new_x<0 or new_x>=self.rows or new_y<0 or new_y>=self.columns:
            return (-1,state)
        new_state=self.stateMap[new_x][new_y]
        return (self.scoreMap[new_x][new_y],new_state)


    def getTrajectoryScore(self,step,state,policy,action,stop_when_target=False):
        #policy是一个 (rows*columns) * actions的二维列表，其中每一行的总和为1，代表每个state选择五个action的概率总和为1
        #Attention: 返回值是一个大小为steps+1的列表，因为第一步也计算在里面了
        #其中的元素是(nowState, nowAction, score, nextState, nextAction)元组
        
        res = []
        nextState = state
        nextAction = action
        if stop_when_target == True:
            steps = 20000
        for i in range(steps+1):
            nowState = nextState
            nowAction = nextAction

            score, nextState = self.getNewState(nowState, nowAction)
            nextAction = np.random.choice(range(5), size=1, replace=False, p=policy[nextState])[0]

            res.append((nowState, nowAction, score, nextState, nextAction))

            if (stop_when_target):
                # print(nextState)
                # print(self.scoreMap)
                nowx = nowState // self.columns
                nowy = nowState % self.columns
                if self.scoreMap[nowx][nowy] == self.score:
                    return res
        return res

    # policy是一维数组
    def showPolicy(self,policy):
        s = ''
        for state in range(len(policy)):


            x = state // self.columns
            y = state % self.columns

            if y==0:
                s=''
            if self.scoreMap[x][y]==self.forbiddenScore:
                tmp = {0:"⏫️",1:"⏩️",2:"⏬",3:"⏪",4:"🔄"}
                s+=tmp[policy[state].argmax()]
            elif self.scoreMap[x][y]==self.score:
                s+="✅"
            else:
                tmp = {0: "⬆️", 1: "➡️", 2: "⬇️", 3: "⬅️", 4: "🔄"}
                s += tmp[policy[state].argmax()]
            if y==self.columns-1:
                print(s)
