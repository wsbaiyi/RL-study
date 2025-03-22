# RL-study
强化学习

脉络图

![image-20250321192320446](pic/image-20250321192320446.png)

# ch1 basic

## 概念

**Action:A** 

**State:S**

**State transition probability**：概率

p(s2|s1,a2)=1

p(s_i|s1,a2)=0   i!=2

**Policy:pi**

某个S将采取的A

pi(a1|s1)=0

pi(a2|s1)=0.5

pi(a3|s1)=0.5

pi(a4|s1)=0

pi(a5|s1)=0

**Tabular representation**

表格形式：只能表示deterministic，而不是stochastic

![image-20250321193403827](pic/image-20250321193403827.png)

**Reward**

代表某一action的评价数

由当前state和action决定

**trajectory**

state-action-reward **chain**

**return**

沿着trajectory得到的reward的总和

**discounted return:gamma**

由于trajectory无限长，引入衰减系数gamma

gamma趋于1，更关注near future

gamma趋于0，更关注far future

**episode**

有限步

continuing tasks：没有终止状态，无限运行

本课将target state作为normal state，agent可以离开target 并重新进入

## Markov decision process MDP

M:Markov property

D:policy

P:set probability

**set**

S，A(s)，R(s,a)

**probability**

State transition probability：p(s'|s,a)       

Reward probability：p(r|s.a)

**policy**

pi(a|s)

**Markov property**

无记忆性

p(s_t+1|a_t+1,s_t.....a_1,s_0)=p(s_t+1|a_t+1,s_t)

p(r_t+1|a_t+1,r_t.....a_1,r_0)=p(r_t+1|a_t+1,r_t)

# ch2 BE

v_i表示从s_i开始获得的return

某一状态的return 依赖于其他状态的return

## state value:V

state value就是状态s未来的return的期望

S_t----A_t--->R_t，S_t+1---A_t+1--->R_t+1，S_t+2----A_t-+2-->R_t+2,,,

这些跳跃由probability决定

![image-20250321200532658](pic/image-20250321200532658.png)
$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+2}+........
$$

$$
% 期望
v_\pi(s)= \mathbb{E}[G_t|S_t=s]
$$

state value和state和policy有关

return是对单个trajectory求值

state value是对多个trajectory求期望

## Bellman equation BE推导

贝尔曼公式

![image-20250322095000782](pic/image-20250322095000782.png)

immediate reward mean

![image-20250322095020085](pic/image-20250322095020085.png)

future reward mean

![image-20250322095038068](pic/image-20250322095038068.png)

**公式**

实际上每个s都有一个state value，因此有很多个state value；因此可以联立解

![image-20250322095542311](pic/image-20250322095542311.png)
$$
\pi(a|s)是policy，如果是固定的，整个式子就是Policy-evaluation
\\括号里的蓝色式子是model，有model-base和model-free
$$
**例子**

可以直观看出来也可以根据公式算出

可以算出所有s的state value，哪个大就说明s好

另外也可以代入不同的policy，判断同一s的state value，大的那个policy好

![image-20250322100507867](pic/image-20250322100507867.png)

**矩阵向量形式 matrix vector form**

![image-20250322103940380](pic/image-20250322103940380.png)

![image-20250322104310489](pic/image-20250322104310489.png)

**policy evaluation：给定policy，得到state value**

**iterative solution**

可以证明当k无穷时，v_k趋于v_pi

![image-20250322104537227](pic/image-20250322104537227.png)

## Action value

action value是s选择action后获得的未来return的期望
$$
q_\pi(s,a)=\mathbb E[G_t|S_t=s,A_t=a]
$$
![image-20250322105652398](pic/image-20250322105652398.png)

![image-20250322105917831](pic/image-20250322105917831.png)

## summary

![image-20250322110659444](pic/image-20250322110659444.png)

# ch3 BOE

**improve policy：use action value**

选择action value最大的action作为policy

## optimal policy

$$
v_{\pi1}(s)>v_{\pi2}(s) \quad for \quad all\quad s \in S
$$

$$
此时可以说{\pi_1}\quad better \quad than \quad {\pi_2}
$$

optimal policy就是对所有策略上式都成立

## Bellman optimality equation BOE推导

**贝尔曼最优公式：policy不确定，需要求解**

element wise form：向量元素对应相乘
$$
v(s)=\underset{\pi}{max}\sum_{a}\pi(a|s)(\sum_{r}p(r|s,a)r+\gamma\sum_{s\prime}p(s\prime|s,a)v(s\prime)),\quad \forall \in S
\\=\underset{\pi}{max}\sum_a\pi(a|s)q(s,a) \quad s \in S
$$
matrix-vector form
$$
v=\underset{\pi}{max}(r_{\pi}+\gamma P_{\pi}v)
$$
**有两个未知数v和pi，如何求解贝尔曼最优公式？**

很简单，先求右式子的max得到pi，再求解整个式子
$$
\underset{\pi}{max}\sum_a\pi(a|s)q(s,a) \quad s \in S
$$
求解上式，只要对应a最大的pi为1
$$
\underset{\pi}{max}\sum_a\pi(a|s)q(s,a)=\underset{a\in A(s)}{max}q(s,a)
\\ \pi(a|s)=
\left\{
\begin{aligned}
&1 \quad a=a^*
\\&0 \quad a\ne a^*
\end{aligned}
\right.
$$
将BOE变为关于v的函数
$$
v=f(v)=\underset{\pi}{max}(r_\pi+\gamma P_\pi v)
$$

## contraction mapping theorem

**fixed point**: 

f(x)=x

**contraction mapping**（or contraction function):

收缩映射
$$
||f(x_1)-f(x_2)||\le\gamma ||x_1-x_2||
\\ where \quad \gamma \in (0,1)
$$
**contraction mapping theorem**

任何等式满足f(x)=x，而且f是一个contraction mapping，则有以下性质：

1：exist fixed point x*

2：x*是unique

3：趋于无穷时，x会收敛到x*****，可以迭代式求解x*

## contraction property of BOE

对BOE
$$
v=f(v)=\underset{\pi}{max}(r_\pi+\gamma P_\pi v)
$$
有
$$
||f(x_1)-f(x_2)||\le\gamma ||x_1-x_2||
\\ \gamma \quad is \quad the \quad discount \quad rate
$$
因此BOE有contraction property  1，2，3

**policy optimality**

BOE实际上是特殊条件下的BE
$$
v^*=\underset{\pi}{max}(r_\pi+\gamma P_\pi v^*)
\\ \pi^*=\underset{\pi}{argmax}(r_\pi+\gamma P_\pi v^*)
\\v^*=r_{\pi^*}+\gamma P_{\pi^*}v^*
$$
证明得到
$$
v^*  \quad is  \quad the \quad  largest  \quad state \quad  value
\\ \pi^* \quad  is \quad  the \quad  optimal \quad  policy
$$
**optimal policy**
$$
\\ \pi^*(a|s)=
\left\{
\begin{aligned}
&1 \quad a=a^*(s)
\\&0 \quad a\ne a^*(s)
\end{aligned}
\right.
$$

## **analyzing optimal policies**

红色的量是model提供的已知的

黑色的量是未知需要求解的（概率）

![image-20250322145842051](pic/image-20250322145842051.png)

gamma越小越短视

gamma越大越长视

forbidden reward惩罚越重就会绕开forbidden area

单纯缩放，调整所有的r，不会改变策略；因为optimal policies关注的是相对reward

**optimal policies由gamma和reward共同约束**

## summary

![image-20250322150622403](pic/image-20250322150622403.png)

![image-20250322150647201](pic/image-20250322150647201.png)

# ch4 value/policy iteration

##  value iteration

给定state value
$$
v=f(v)=\underset{\pi}{max}(r_\pi+\gamma P_\pi v)
\\ v_{k+1}=f(v_k)=\underset{\pi}{max}(r_\pi+\gamma P_\pi v_k)\quad k=1,2,3...
$$
**step1: policy update**
$$
\pi_{k+1}=\underset{\pi}{argmax}(r_\pi+\gamma P_\pi v_k)
$$
**step2: value update**
$$
v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}} v_k
$$
**伪代码**

![image-20250322151705240](pic/image-20250322151705240.png)

## policy iteration

给定policy

**step1: policy evaluation （PE）**

计算state value可以用closed-form（矩阵算法）或iterative solution（是一个迭代算法）
$$
v_{\pi_{k}}=r_{\pi_{k}}+\gamma P_{\pi_{k}} v_{\pi_{k}}
$$
**step2: policy improvement （PI）**
$$
\pi_{k+1}=\underset{\pi}{argmax}(r_\pi+\gamma P_\pi v_{\pi_{k}})
$$
**伪代码**

![image-20250322152945637](pic/image-20250322152945637.png)

**举例：grid world越靠近target area的策略越快变好，因为state value依赖其他的state value，只有其他的state value变好后自己的state value才会变好**

## truncated policy iteration 

**model-based  MBRL**

**policy iteration和value iteration比较**

![image-20250322153753700](pic/image-20250322153753700.png)

![image-20250322154126521](pic/image-20250322154126521.png)

step4会出现
$$
v_{\pi_1}\ge v_{1}
$$
**解释**

truncated仅进行j步

policy iteration 理论上不存在

![image-20250322154505525](pic/image-20250322154505525.png)

**伪代码**

![image-20250322154645787](pic/image-20250322154645787.png)

![image-20250322154932229](pic/image-20250322154932229.png)

# ch5 Monte Carlo Learning

**model-free**  

no precise distribution

采样估计
$$
\mathbb E[X]\approx \overline{x}=\frac{1}{N}\sum_{j=1}^Nx_j
$$

## MC Basic

![image-20250322160228617](pic/image-20250322160228617.png)
$$
g^{i} \ \quad is \quad a \quad sample \quad of \quad G_t
\\q_\pi(s,a)=\mathbb E[G_t|S_t=s,A_t=a]\approx \frac{1}{N}\sum_{i=1}^Ng^{i}(s,a)
$$
when model is unavailable , use data

in RL we call **experience**

step1和policy iteration不同，MC是直接得到q

**policy iteration先计算state value，之后得到action value**

**MC 是直接估计action value**

![image-20250322160859139](pic/image-20250322160859139.png)

**伪代码**

![image-20250322160828584](pic/image-20250322160828584.png)

episode越短，只有离target area最近的state value非零

episode变长，离target area越远的state value逐渐增大

## MC Exploring Starts

**visit**

MC Exploring Starts仅用一次episode

![image-20250322163841409](pic/image-20250322163841409.png)

**高效使用数据**

1 **first-visit method**

仅估计第一次出现的(s,a)pair的return

2 **every-visit method**

每出现一次(s,a)pair就估计一次return，改进策略

**高效更新策略**

1 得到所有的episode的average来估计action value

2 每得到一个episode的return就估计action value

**Generalized policy iteration GPI**

在policy evaluation 和policy improvement不断切换，而且进行有限次迭代计算state value

**伪代码**

first-visit method

![image-20250322164214070](pic/image-20250322164214070.png)

细节：逆序计算（为了利用已经计算的数据）

缺陷：很难找到拥有每个(s,a)开头的episode

## MC e-Greedy

**stocastic**

soft policy：每个action都有概率选择

greedy policy

![image-20250322165527350](pic/image-20250322165527350.png)

e-greedy 可以balance exploitation(充分利用) and exploration(充分探索)

e=1，more exploration，最优性越差

e=0，more exploitation，最优性越好

not require exploring starts

**技巧**

当e很小的时候是consistent，因为它的策略和最优策略是相同的；

但当e逐渐增大时，策略和最优策略差别越来越大

使用MC e-Greedy的技巧：让e取较大的值，充分发挥exploration，接着逐渐减少e，最后得到最优策略

**伪代码**

every-visit method：下面没有does not appear

![image-20250322170016470](pic/image-20250322170016470.png)

# ch6 stochastic approximation

## **mean estimate**

若收集全部样本再求mean，太费时了
$$
\mathbb E[X]\approx \overline{x}=\frac{1}{N}\sum_{j=1}^Nx_j
$$
**incremental and iterative**

每采样一个样本就计算mean

![image-20250322175949598](pic/image-20250322175949598.png)

下面的是一种stochastic approximation也是一种stochastic gradient descent
$$
mean\quad estimate  \quad algorithm
\\w_{k+1}=w_k-\alpha_k (w_k-x_k)
$$

## Robbins-Monro  RM

求解
$$
g的表达式可以不知道
\\g(w)=0
$$
最优解是
$$
w^*
$$
第k次的解
$$
w_k
$$
RM算法求解
$$
w_{k+1}=w_k-a_k\widetilde{g}(w_k,\eta_k)
$$
g~是g的第k次的noisy obsevation
$$
\widetilde{g}(w_k,\eta_k)=g(w_k)+\eta_k
$$
![image-20250322181546695](pic/image-20250322181546695.png)

1：要求递增且递增率有限

2：系数一次和发散，说明a收敛不能太快；系数平方和收敛，说明a收敛至0

例如 a= 1/k

3：均值为0且方差有界


$$
假设g(w)=w-\mathbb E(X)\
\\
\\ \widetilde{g}(w,x)=w-x
\\
\\\widetilde{g}(w,\eta)=w-x=g(w)-\eta
\\
\\mean\quad estimate是特殊的RM算法
\\w_{k+1}=w_k-\alpha_k\widetilde{g}(w_k,\eta_k)=w_k-\alpha_k(w_k-x_k)
$$

## Stochastic gradient descent

**求解问题**
$$
\underset{w}{min}\quad J(w)=\mathbb E[f(w,X)]
$$
GD：沿着梯度下降的方向更新参数w
$$
w_{k+1}=w_k-\alpha_k\nabla_w\mathbb E[f(w_k,X)]
\\\quad\quad\quad\quad=w_k-\alpha_k \mathbb E[\nabla_wf(w_k,X)]
$$
BGD
$$
\mathbb E[\nabla_wf(w_k,X)]\approx \frac{1}{n}\sum_{i=1}^n\nabla_wf(w_k,x_i)
\\w_{k+1}=w_k-\alpha_k \frac{1}{n}\sum_{i=1}^n\nabla_wf(w_k,x_i)
$$
SGD
$$
w_{k+1}=w_k-\alpha_k\nabla_wf(w_k,x_k)
$$
**an example**
$$
\underset{w}{min}\quad J(w)=\mathbb E[f(w,X)]=
\mathbb E[\frac{1}{2}||w-X||^2]
$$
GD
$$
true\quad gradient
\\w_{k+1}=w_k-\alpha_k\nabla_w J(w_k)
\\\quad\quad\quad\quad=w_k-\alpha_k \mathbb E[\nabla_wf(w_k,X)]
\\\quad\quad=w_k-\alpha_k \mathbb E[w_k-X]
$$
SGD
$$
stochastic\quad gradient
\\mean\quad estimate是特殊的SGD算法
\\w_{k+1}=w_k-\alpha_k\nabla_wf(w_k,x_k)=w_k-\alpha_k(w_k-x_k)
$$
**show that SGD is a special RM algorithm**
$$
\underset{w}{min}\quad J(w)=\mathbb E[f(w,X)]
\\g(w)=\nabla_wJ(w)=\mathbb E[\nabla_wf(w,X)]=0
\\
\\\widetilde g(w,\eta)=\nabla_w f(w,x)
\\
\\=\mathbb E[\nabla_wf(w,X)]+\nabla_w f(w,x)-\mathbb E[\nabla_wf(w,X)]
\\ 
\\SGD是特殊的RM算法
\\w_{k+1}=w_k-\alpha_k\widetilde g(w_k,\eta_k)=w_k-a_k\nabla_wf(w_k,x_k)
$$
**convergence pattern**

当w越接近最优解时，随机性越大

当w越远离最优解时，随机性越小

![image-20250322191629713](pic/image-20250322191629713.png)

**a deterministic formulation**
$$
x_i是real\quad number,x不是随机变量的样本
\\\underset{w}{min}\quad J(w)=\frac{1}{n}\sum_{i=1}^nf(w,x_i)
$$
手动引入随机变量X
$$
p(X=x_i)=\frac{1}{n}
\\将deterministic变成sotchastic
\\每个数据随机抽样
\\\underset{w}{min}\quad J(w)=\frac{1}{n}\sum_{i=1}^nf(w,x_i)=\mathbb E[f(w,X)]
$$

## BGD,MBGD,SGD

当m=n时，MBGS is not BGD strictly speaking

MBGD是在n个样本中随机抽取n个

BGD是抽这n个样本

![image-20250322193045334](pic/image-20250322193045334.png)

SGD估计最慢，因为每次都是一个样本

![image-20250322193859636](pic/image-20250322193859636.png)

## summary

![image-20250322194044012](pic/image-20250322194044012.png)

# ch7 Temporal-Difference Learning

时序差分方法：model free，incremental / iterative

## TD learning of state values

$$
data/experience
\\(s_0,r_1,s_1....s_t,r_{t+1},s_{t+1}...) \quad or \quad {(s_t,r_{t+1},s_{t+1})}
$$

**TD learning algorithm**
$$
v_t(s_t)是v_{\pi}(s_t)在t时刻的估计值
\\\alpha_t(s_t)是s_t在时间t时的学习率
\\
\\v_{t+1}(s_t)=v_{t}(s_t)-\alpha_t(s_t)[v_{t}(s_t)-[r_{t+1}+\gamma v_{t}(s_{t+1})]]
\\
\\ v_{t+1}(s_t)=v_{t}(s_t)\quad \forall s\ne s_t
$$
![image-20250322200221678](pic/image-20250322200221678.png)

**TD target**

为什么叫TD target

![image-20250322200553405](pic/image-20250322200553405.png)

**TD error**

![image-20250322201016589](pic/image-20250322201016589.png)

这个TD algorithm只能估计给定policy的state value，

不能估计action value，也不能搜索optimal plicy

**TD 算法是在没有模型的情况下求解贝尔曼公式**

**TD和MC的比较**

![image-20250322202409923](pic/image-20250322202409923.png)

![image-20250322202537018](pic/image-20250322202537018.png)

TD是有偏估计，因为涉及初始值的估计；方差小，随机变量少

MC是无偏估计；方差大，因为只取一个episode，一个episode有很多Reward采样，同时实际上有很多episode。

## TD learning of action values: Sarsa





## TD learning of action values: Expected Sarsa





## TD learning of action values:  n-step Sarsa





## TD learning of optimal action values: Q-learning



## summary
