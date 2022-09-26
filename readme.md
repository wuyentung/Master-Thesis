# Marginal Productivity and Efficiency Analysis on Taiwan’s Life Insurance Industry through Meta-Data Envelopment Analysis
###### key paper: [Lee (2017)](https://doi.org/10.1057/s41274-016-0129-8)
###### process *main.py* will show all figures of the analysis (requirement: [Gurobi lisence](https://www.gurobi.com/academia/academic-program-and-licenses/))
## Abstract
This study proposes a novel analysis chart with Marginal Profit Consistency, which is a derivative from Directional Marginal Productivity, and Efficiency Change. Where the Marginal Profit Consistency can help us investigate how a DMU’s progress in history be related to the direction which leads to gain maximum marginal profit. We use life insurance industrial data in Taiwan as the application field, with the assumption that there is no technical change in the paneled 3 years. Using our proposing chart, we can highlight the industry event related to company performance and its marginal output portfolio. Besides, we further explain the usage when we face the merger effect in industry, which is always a worth discussing issue once incurred. To the best of our knowledge, we suggest how an insurance company can move forward if it falls in Laggard quadrant. 
## Research Purposes
- Propose a novel analysis chart examining how a decision making unit's (DMU) reaction to the margin will affect its performance. 
- The proposed chart aims to highlight the industrial major and minor events related to company performance and its marginal output portfolio.
## Application Field, Database, and Parameters
- Taiwan Life Insurance Industry, 2014-2016, 2018-2020
- Database: annual Income Statements collected by [Taiwan Insurance Institute](https://www.tii.org.tw/tii/actuarial/actuarial3/)
- Inputs:
    - Insurance Expenses ( $X_1$ ): expenses incurred in the service of insurance
    - Operation Expenses ( $X_2$ ): expenses incurred for other operations of an insurance company
- Outputs: 
    - Underwriting Profit ( $Y_1$ ): profit earned from the insurance business
    - Investment Profit ( $Y_2$ ): profit earned from the investment portfolio  
## Research Flowchart
- <img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/methodology%20flowchart.png" width="500" height="500" />
### Module Flowchart
- <img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/module%20flowchart.png" width="900" />
  
---
---
1. 改用保險業實證 dmp
    1. 三年資料，假設沒有 tech. change
    1. 先算整體的，有能力再用網路
1. 最後有時間再來 scope property

1. 效率估計改 output orient
1. 改單位
    - 改單位後就算得出來了
1. alpha 的 k 取全部
1. 先用 2003 年資料
// can be used: analyse simplified.py