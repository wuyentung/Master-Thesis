# Marginal Productivity and Efficiency Analysis on Taiwan’s Life Insurance Industry through Meta-Data Envelopment Analysis
###### Thesis [link](doi:10.6342/NTU202202824) (open access in 2025)
###### Key paper: [Lee (2017)](https://doi.org/10.1057/s41274-016-0129-8)
###### Process *main.py* will show all figures of the analysis (requirement: [Gurobi lisence](https://www.gurobi.com/academia/academic-program-and-licenses/))
## Abstract
This study proposes a novel analysis chart with Marginal Profit Consistency, which is a derivative from Directional Marginal Productivity, and Efficiency Change. Where the Marginal Profit Consistency can help us investigate how a DMU’s progress in history be related to the direction which leads to gain maximum marginal profit. We use life insurance industrial data in Taiwan as the application field, with the assumption that there is no technical change in the paneled 3 years. Using our proposing chart, we can highlight the industry event related to company performance and its marginal output portfolio. Besides, we further explain the usage when we face the merger effect in industry, which is always a worth discussing issue once incurred. To the best of our knowledge, we suggest how an insurance company can move forward if it falls in Laggard quadrant. 
## Research Purposes
---
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
<img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/methodology%20flowchart.png" width="500" height="500" />

### Module Flowchart
Figure below represents the module we use when implementing the research. Util modules help us construct fundamental of data preprocessing, analysis model, and analysis figure plotting. Execution modules then link all util modules to perform the analysis in different phases, or requirments.  
<img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/module%20flowchart.png" width="900" />  

## Analysis Result
### Result Tables (round in 3, amount: million NTD)
Table below is the example of the panel data and analysis value in period 2014-2016 (round in 3), including two insurance companies. The first four columns are the input and output data. From the row perspective, we combine insurer’s name with its year collected. For example, insurer AIA Taiwan 14 refers to company AIA (American International Assurance) Taiwan with 2014 data, AIA Taiwan 15 refers to company AIA Taiwan with 2015 data, etc. We use vectors to represent Underwriting Profit and Investment Profit respectively on column of Output Progress Direction and Marginal Profit Max Direction. Where Marginal Profit Max Direction is the mean of marginal profit max direction retrieved from DMPs under specifying input Insurance Expenses and Operation Expenses. 

| Company Name | Insurance Expenses | Operation Expenses |Underwriting Profit | Investment Profit | Scale|Profit | Output Progress Direction | Marginal Profit Max_Direction Insurance |  Marginal Profit Max_Direction Operation | Marginal Profit Consistency | Efficiency | Efficiency Change |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AIA Taiwan 14 | 1.002 | 1.553|162.949|1.702|2.555|164.651|[-0.14, 0.86]|[0.99, 0.01]|[0.99, 0.01]|-0.151|	1.009|1.001|
|AIA Taiwan 15|	0.938|	1.646|	162.911|1.93|2.584|	164.842|[0.797, 0.203]|	[0.99, 0.01]|[0.9, 0.1]|0.981|1.008|1.003|
|AIA Taiwan 16|	0.878|	1.733|	163.311|	2.032|	2.611|	165.343|	[nan, nan]|	[0.01, 0.99]|	[0.99, 0.01]| nan |	1.005 | nan |
|Cathay Life 14|32.215|	19.411|	28.884|	169.247|	51.626|	198.131|[-0.22, 0.78]|[0, 0]|[0.99, 0.01]|	-0.262|	1|	1|
|Cathay Life 15| 32.788|25.434|	25.141|	182.498|	58.222|	207.638| [-0.742, 0.258]|[0, 0]|[0, 0]|	0|	1|	1|
|Cathay Life 16|37.924|	30.191|	0|	191.24|	68.115|	191.24|	[nan, nan]|	[0, 0]|	[0, 0]| nan	|	1|	nan |

### Analysis Figures
#### 2014-2016
Empirical analysis for marginal profit consistency and efficiency change in period 2014-2016 is shown below. Our proposing chart can identify industrial major events. For example, Fubon Life and Shin Kong Life suffered efficiency regress during 2015 to 2016 seriously (see the red circle). From historical news, these two insurance companies did not invest well in 2015. Shin Kong Life even sold off its real estate.  
<img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/2014-2016.png" width="900" />  

#### Merger Effect
With our proposed analysis chart, we can intuitively find out the performance and portfolio change. This can also help us to investigate the merger effect. Taiwan Life merged with CTBT Life in 2016. The industry predicted this merger would make Taiwan Life become one of the top 6 large life insurance companies.  
We add a dummy DMU, named DUMMY Taiwan 16, naively summing up the inputs and outputs, represents the merged company of CTBC Life and Taiwan Life in theory. We then summarize the analysis of real word merger and the dummy merger in figure below. The merged Taiwan Life 16 performs better than DUMMY Taiwan 16 in efficiency. Namely, the real-world merging for CTBC Life and Taiwan Life in 2016 did better than theoretical predicted.  
<img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/merger%20effect.png" width="900" />

### Suggestion for DMUs
- In the analysis chart, we want not only the indusrial events, but also the manageral suggestion for DMUs perform bad. We then split the analysis figure into four quadrants like below. Where the vertical axis (Marginal Profit Consistency) is split by the mean of DMUs; the horizental axis (Efficiency Change) is seperated by $1$, as the boundary of efficiency progress and regress.  
<img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/four%20quadrant.png" width="900" />

- We would like to understand the improving direction for DMU falls in Laggard district. Belowing summarized the movement in panel data 2014-2016 and 2018-2020. In summary, our suggestion for DMU falling in Laggard quadrant:
    - Large scale DMU: focus on efficiency
    - Small scale DMU: change on marginal profit consistency

<img src="https://github.com/wuyentung/Master-Thesis/blob/main/IMAGES/suggestion.png" width="900" />