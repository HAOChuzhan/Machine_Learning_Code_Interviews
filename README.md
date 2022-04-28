# Machine_learning_code_interviews
In this repo, I use Python to write common machine learning algorithms in interviews. The following is a record of some interview related common knowledge.

```mermaid
gantt
    title work plan
    dateFormat  YYYY-MM-DD
    section intern
    seeking an intern :a1, 2022-03-01, 75d
    #Another task     :after a1  , 20d
    #section Another
    #Task in sec      :2022-01-12  , 12d
    #another task      : 24d
```

 [Langboat](#1) | [Didi Tech](#2) | [JD Tech](#3) | [NIO](#4) | [Meituan](#5) | [Haikang](#6)

---

1. <span id="1">[Langboat](https://langboat.com/techs/search) — Two interviews</span>:smile: 
   
   1. 主要有预训练（[孟子](https://github.com/Langboat/Mengzi)）、搜索、文本生成三个团队
   2. 流程：主要是两轮面试
      1. 一面：主要就是围绕着简历的论文展开
   
      2. 二面：类似于一面（忘记记录具体面试内容了总体来说把简历上的细节熟悉清楚，这类创业公司面试相对简单）
      
         **Time：均为1h** | **Base：Beijing**
   
2. <span id="2">Didi Tech —Algorithm Engineer</span>:smile:
   
1. 滴滴出行的安全策略团队：主要是处理出行期间记录的音频视频多模态信息来判断出行风险
   
2. 流程：进行了一面，我可能的入职时间太晚第二面就没有进行暂时先保持联系
   
   ​	**一面：**
   
   1. 由于我之前的实习与滴滴的业务还是比较相关的，于是重点在实习经历上，由于我用到了相关的机器学习的模型，面试官问了些具体的业务实现之后，再从LR和Xgboost出发。
   
      （1）我们one-hot编码的特征过于稀疏如何解决？会影响最后的模型效果嘛？  
   
      （2）介绍下你所使用的Auc指标。  
   
      （3）在大量的数据训练过程中，会有过拟合风险嘛？你知道哪些解决方法？（开始我只记得dropout了—乘机还问了dropout为什么可以解决过拟合，与集成学习有什么联系没有，在他提醒下想起来L1、L2正则化）。  
   
      （4）Xgboost模型为什么比Lr模型的效果好（集成学习 还有xgb特征筛选能力），xgb的树的深度等参数如何选择还有节点的分裂一些原理，感觉这部分还需要加强。  
   
      （5）Spark也顺便提了一嘴，因为我当时模型训练用的是spark。  
   
      （6）接下来就是我详细讲了论文内容，和面试官沟通。  
   
      （7）手写算法题：连续最长子序列-dp来做就行
   
         **Time：1h15min** | **Base：Beijing**
   
3. <span id="3">JD Tech —Algorithm Engineer</span>&#x1f602;
   
   1. 主要是用户增长和营销算法团队
   
   2. 主要也是讲实习的经历，在其中问了一些（1）Bert、transformer的结构和一些功能。（2）LSTM模型为什么可以解决长距离依赖和梯度消失和梯度爆炸。[LSTM如何解决梯度消失或爆炸的?](https://www.cnblogs.com/bonelee/p/10475453.html)（3）还有一些细枝末节的有些忘了。。。感觉这个面试氛围十分轻松
   
      **Time：50min** | **Base：Beijing**
   
4. <span id="4">NIO — One interview</span>:expressionless:

   1. 部门主要是做语言理解的，为车机交互提供技术支撑

   2. 感觉面试官就是走流程一样，让我自己自我介绍完了之后按照简历挨个介绍实习和论文工作；最后让我反问，他跟我讲解了下他们的业务，主要是做车机上的对话查询，文本信息主要是车机手册，可以利用做阅读理解或者是依托于知识图谱来做问答查询。最后估计面试只持续了30-40min就结束了，感觉十分像KPI面。

      **Time：30-40min** | **Base：Beijing or Shanghai**

5. <span id="5">Meituan —NLP Algorithm Engineer 2022年4月24日</span>:smile:

   1. 主要是智能风控团队：做网络舆情、客户投诉等相关对话文本信息的检测。

   2. 流程：

      ​	**一面：**

      1. 上来先手撕两道算法题：（1）之字形打印二叉树；（2）在二维矩阵中搜索target；这两道题都是算比较简单的，他会问下有没有什么优化的点；
      
      2. 接下来估计是看都A了，然后又让我口述一道 "从多个升序数组中找出第k小的数" ，我大致讲了下使用优先队列的做法，然后他就问了下**堆的基本原理**是怎么实现的。—感觉还可以去查查其他的优化方法

      3. 然后开始围绕论文中的工作问答：

         （1）主要问了其中关系分类中，各关系标签的数量分布，如何解决其中的比例较少的关系的分类问题（感觉有些类似标签分布的长尾效应）；然后问了下知识图谱构建方面的我简要说明了一下；然后问我除了文章中这一类方法，做复杂多跳问答还有哪些方法？

         （2）紧接着问了问了除问答外对于其他NLP任务是否熟悉，我回答说了简单的文本匹配和文本分类任务（但他貌似想听到的是NER这类的任务），然后他接着问：你使用的文本匹配的方法有哪些？；答：双塔这种和直接 [SEP] 拼接好再经过bert编码后进行分类，然后他问了这两种方法的优缺点；答：从实现起来的难易程度和部署上线的存储空间的消耗方面回答的。

         （3）然后主要是再bert上进行展开，问：你先讲下Bert的基本结构？问：这里地动态地位置编码相较于之前的固定的位置编码有什么优势？答：可能是实验的结果下，可学习的绝对位置编码相较于余弦位置编码效果好一些([positional encoding位置编码详解：绝对位置与相对位置编码对比_相对位置编码](https://blog.csdn.net/xixiaoyaoww/article/details/105459376))；又问：你觉得Bert相较于之前的编码方式最大的优势是什么？以及缺点呢？答：动态编码，缺点就是模型参数太大难以上线；又问：有没有什么轻量化的方法和上线部署的方法？答：一些轻量化的AlBert和DistillBert等模型，或者是模型蒸馏等方法，TT分解等方法大致说了下。

         （4）继续就着Bert，问：Bert中的Mask机制你说一下；答：详细说了随机选15%Token 用[MASK]来替换，然后15%的token中的80%被替换成[MASK]，10%替换成随机的其他token，10%保持原先的词不变；再问：随机替换成其他token和保持不变的作用是什么？答：增加模型鲁棒性等等，保持不变是因为之后微调过程中是不存在[MASK]标记的，因此模型在预测的时候不知道输入对应位置是否为正确的词（10%概率）这也使得模型更多地依赖于上下文信息去预测词汇这就赋予了模型一定地纠错能力。

         [以上问题回答可以参考：关于BERT的若干问题整理记录 - 知乎 (zhihu.com) —3.1](https://zhuanlan.zhihu.com/p/95594311)

         （5）继续问：除了bert之外还了解哪些预训练语言模型；答：我又进一步解释了Roberta的原理；再问：除了这些之外呢。答：XLNet、ERNIE等等（忘记说GPT了），但不是特别了解具体的内部结构；

         （6）继续问：文本分类中的[fastText](https://zhuanlan.zhihu.com/p/32965521)、[TextCNN](https://zhuanlan.zhihu.com/p/77634533)这类方法你了解吗？答：之前使用过，但是太久没接触忘记具体原理了。。。最后就让我反问了。
      
            **Time：1h20min** | **Base：Beijing or Shanghai**
   
8. <span id="6">Haikang—Ai Algorithm Engineer 2022年4月27日</span>:smile:

   1. 主要是AI研究型实习生，主要是研究GNN在ogb任务上的应用；

   2. 流程：也是提前打电话大致聊了下主要的工作内容，询问了我的意向。

      ​	**一面：**

      1. 先是做了简单的自我介绍，然后从实习经历和论文内容讲起，最后是讲了我最近做的有关GNN的内容以及对GNN系列模型的原理理解。

      2. 在做特征过程中的数据的如何处理的？one-hot和数据分桶具体是怎么实现？ont-hot的降维方法。我的一些理解：（1）基于树的模型是不需要进行特征归一化，例如随机森林、bagging、boosting等方法，基于参数或者距离的模型是需要特征归一化。（2）tree model是不需要ont-hot编码的，其实树模型在动态的过程中生成了类似ont-hot+feature crossing的机制；决策树中是没有特征大小的概念，只有特征分布于那一部分的概念。

      3. GNN的原理推导以及相关的其他GNN模型，如果出现图的节点达到亿级别，我们该如何优化？

      4. 最后的算法题目：两个数字组成的字符串，实现两个字符串相乘。

         <details><summary>字符串相乘题解</summary>
         <pre><code>string multiply(string nums1, string nums2) {
             string res = "";
             int m = nums1.length();
             int n = nums2.length();
             if (nums1=="0" || nums2=="0") return "0";
             vector<int> vals(m+n, 0);
             for (int i=m-1; i>=0; i--) {
                 for (int j=n-1; j>=0; j--) {
                     int mul = (nums1[i]-'0')*(nums2[j]-'0');
                     int p1 = i+j, p2=i+j+1, sum = mul+vals[p2];
                     vals[p1] += sum/10;
                     vals[p2] = sum%10;   
                 }
             }
             for (int i:vals) {
                 if (!res.empty() || i!=0) res.push_back(i+'0');
             }
             return res;
         }
         </code></pre></details>

​						**Time：1h20min** | **Base：Hangzhou**



---

