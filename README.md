https://paperswithcode.com/paper/a-machine-learning-framework-for-stock

# A Machine Learning Framework for Stock Selection

Introduction
====
This project demonstrates how to apply machine learning algorithms to distinguish "good" stocks from the "bad" stocks. To this end, we construct 244 technical and fundamental features to characterize each stock, and label stocks according to their ranking with respect to the return-volatility ratio. Algorithms ranging from traditional statistical learning methods to recently popular deep learning method, e.g. Logistic Regression (LR), Random Forest (RF), Deep Neural Network (DNN), and Stacking Ensemble model, are trained to solve the classification task. Genetic Algorithm is also used to implement features selection. The effectiveness of the stock selection strategy is validated in Chinese stock market from both statistical and practical aspects, showing that:
- Stacking outperforms other models reaching an AUC score of 0.972;
- Genetic Algorithm picks a subset of 114 features and the prediction performances of all models remain almost unchanged after the selection procedure, which suggests some features are indeed redundant;
- LR and DNN are radical models; RF is risk-neutral model; Stacking is somewhere between DNN and RF.
- The portfolios constructed by our models outperform market in back tests.

About
====
- For details, you can have the research paper at:***http://arxiv.org/abs/1806.01743***.
- Currently, we do not plan to reveal the whole raw dataset, while a data sample will be shared.

Contributions
====
Contributors:
------- 
- ***XingYu Fu***
- ***JingHong Du*** ( https://github.com/jaydu1 )
- ***YiFeng Guo*** 
- ***MingWen Liu*** 
- ***Tao Dong***
- ***XiuWen Duan***

Institutions: 
------- 
- ***Likelihood Lab***
- ***ShiningMidas Private Fund***
- ***Sun Yat-sen University***

Set up
====
Python Version:
------- 
- ***3.5***

Modules needed:
------- 
- ***numpy***
- ***pandas***
- ***matplotlib***
- ***math***
- ***os***
- ***sklearn***
- ***tensorflow***
- ***keras***    

Contact
====
- fuxy28@mail2.sysu.edu.cn
