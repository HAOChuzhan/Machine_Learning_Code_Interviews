# Machine_learning_code_interviews
In this repo, I use Python to write common machine learning algorithms in interviews. The following is a record of some interview related common knowledge.

------

1. [Langboat](https://langboat.com/techs/search) — Two interviews    :smile:

   1. 主要有预训练（[孟子](https://github.com/Langboat/Mengzi)）、搜索、文本生成三个团队
   2. 流程：主要是两轮面试
      1. 一面：主要就是围绕着简历的论文展开
      2. 二面：类似于一面（忘记记录具体面试内容了总体来说把简历上的细节熟悉清楚，这类创业公司面试相对简单）

2. Didi Tech —Algorithm Engineer    :smile:

   1. 滴滴出行的安全策略团队：主要是处理出行期间记录的音频视频多模态信息来判断出行风险

   2. 流程：进行了一面，我可能的入职时间太晚第二面就没有进行暂时先保持联系

      1. 一面：由于我之前的实习与滴滴的业务还是比较相关的，于是重点在实习经历上，由于我用到了相关的机器学习的模型，面试官问了些具体的业务实现之后，再从LR和Xgboost出发。time：70-80min

         （1）我们one-hot编码的特征过于稀疏如何解决？会影响最后的模型效果嘛？  

         （2）介绍下你所使用的Auc指标。  

         （3）在大量的数据训练过程中，会有过拟合风险嘛？你知道哪些解决方法？（开始我只记得dropout了—乘机还问了dropout为什么可以解决过拟合，与集成学习有什么联系没有，在他提醒下想起来L1、L2正则化）。  

         （4）Xgboost模型为什么比Lr模型的效果好（集成学习 还有xgb特征筛选能力），xgb的树的深度等参数如何选择还有节点的分裂一些原理，感觉这部分还需要加强。  

         （5）Spark也顺便提了一嘴，因为我当时模型训练用的是spark。  

         （6）接下来就是我详细讲了论文内容，和面试官沟通。  

         （7）手写算法题：连续最长子序列-dp来做就行

3. JD Tech —Algorithm Engineer

   1. 主要是用户增长和营销算法
   2. 主要也是讲实习的经历，在其中问了一些（1）Bert、transformer的结构和一些功能。（2）LSTM模型为什么可以解决长距离依赖和梯度消失和梯度爆炸。[LSTM如何解决梯度消失或爆炸的?](https://www.cnblogs.com/bonelee/p/10475453.html)（3）还有一些细枝末节的有些忘了。。。感觉这个面试氛围十分轻松

4. NIO — One interview    :expressionless:

   感觉面试官就是走流程一样，让我自己自我介绍完了之后按照简历挨个介绍实习和论文工作；最后让我反问，他跟我讲解了下他们的业务，主要是做车机上的对话查询，文本信息主要是车机手册，可以利用做阅读理解或者是依托于知识图谱来做问答查询。最后估计面试只持续了30-40min就结束了，感觉十分像KPI面。Base：Beijing or Shanghai

5. Meituan —NLP Algorithm Engineer

