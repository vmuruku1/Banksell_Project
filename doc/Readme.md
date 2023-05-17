**Problem Statement:** Based on the census, personal details, financial details and other marketing campaign, predict if the customer will Subscribe for Term Deposit.

## Abstract:
This is the initial UCI Machine Learning Collection upload of the well-known marketing bank dataset. The dataset contains details on a financial institution's marketing campaign, which we will examine to identify potential future tactics for the bank's marketing initiatives.
Gaining the maximum value from a particular data collection is a challenging endeavor since it necessitates a thorough analysis of its many properties and accompanying values. To uncover hidden patterns, this process is typically accomplished by presenting data in a visual way.
In this study, a bank's direct marketing data collection is subjected to several visualization approaches.
The data set that was downloaded from the website of the UCI machine learning repository is unbalanced. As a result, several oversampling techniques are employed to improve the predictability of a client's subscription to a term deposit. The effectiveness of the visualization is evaluated by examining the impact of various classifier performance on oversampling strategies. 
With various feature engineering Technique, we be using different Ensemble Models to Predict the outcome.
Introduction:
The goal of this project is to develop a machine learning model that can predict whether a customer will subscribe to a term deposit or not, based on a set of features provided by the bank. The motivation behind this task is to help banks optimize their marketing strategies by identifying customers who are most likely to subscribe to a term deposit, thus reducing marketing costs and increasing the efficiency of marketing campaigns.
The approach taken in this project involves using Python programming language and various machine learning algorithms, such as logistic regression, decision trees, and random forests, to develop and evaluate the predictive model. The data used for training and testing the model is obtained from the UCI Machine Learning Repository, which contains a dataset of a Portuguese bank's direct marketing campaigns.

**Bank Marketing**
- In simple words, bank marketing is the design structure, layout and delivery of customer-needed services worked out by checking out the corporate objectives of the bank and environmental constraints.

**Why is marketing important for banks?**
- The banking sector plays a key role in the development of the economy. Banks are now giving importance to the marketing activities to create awareness regarding their services to the public. Customer satisfaction is important so that banks are introducing new instruments and ways to attract the customers.

Market analysis, data mining, and financial data analysis all require data visualization. It refers to the application of interactive, computer-supported visual representation to enhance cognition and communicate complex ideas underneath data. Charts, graphs, and design components effectively carry out this strategy. These techniques are frequently used by managers and knowledge workers to uncover information hidden in large amounts of data [4] and arrive at the best conclusions. The use of data visualization by decision-makers and their companies has several advantages [2], including helping people learn information in fresh, useful ways.
Identifying and responding to emerging trends can be aided by visualizing the connections and patterns between operational and business activity.

**Literature Review:**

**Deal Banking Marketing Campaign Dataset with Machine Learning**

Link: https://medium.com/@nutanbhogendrasharma/deal-banking-marketing-campaign-dataset-with-machine-learning-9c1f84ad285d
Referred this blog where they used data related to marketing campaigns (phone calls) of a Portuguese banking institution. The classification's objective is to foretell whether a client will sign up for a term deposit (variable y). We will use different type of model and see which model gives highest accuracy.

Analysis on Bank Marketing Campaign for Portuguese Bank
Link: https://github.com/HegdeChaitra/Bank-Marketing-Campaign-Analysis
The project's main objective was to analyze a Portuguese bank's previous marketing campaigns using various machine learning techniques, including Random Forests, Logistic Regression, Gradient Boosting, Decision Trees, and AdaBoost, and predict whether the user would purchase the bank's term deposit.
Recommendation on the marketing team, ways to better target customers using feature importance maps and business intuition.
 “Marketing analytics: Methods, practice, implementation, and links to other fields,” Expert Systems with Applications, 2018. S. L. France and S. Ghose,
The research in this paper examined the theoretical underpinnings of marketing analytics, a vast discipline that emerged from operations research, marketing, statistics, and computer science. One of the difficulties in doing a direct marketing analysis, they claimed, is forecasting consumer behavior. They also covered customer relationship management, multidimensional scaling, correspondence analysis, and latent Dirichlet allocation as big data visualization techniques for the marketing sector (CRM). They discussed the relative value of geographic visualization for retail location research and the overall trade-off between its customary methods and art. They also expanded on discriminant analysis as a method for marketing forecasting. Techniques including ensemble learning, feature reduction, and extraction are used in discriminant analysis. These methods address issues including buying, rating, loyalty etc.

S. Palaniappan, A. Mustapha, C. F. M. Foozy, and R. Atan, “Customer profiling using classification approach for bank telemarketing,” JOIV: International Journal on Informatics Visualization, vol. 1, no. 4-2, pp. 214–217, 2017.
Additionally, this article used the same data set for additional consumer profiling objectives. On the expanded version of the data set analyzed in the current work, naive Bayes, random forests, and decision trees were employed. Prior to assessing the classifiers, preprocessing and normalization were conducted. To conduct the trials and assessment procedures, RapidMiner was employed. Using a prior normalization approach, they demonstrated how each classifier's parameters might be adjusted. They also demonstrated how these parameter values affected recall, accuracy, and precision. Decision trees are the best classifier for consumer profile and behavior prediction, according to their findings.
Background:
Direct marketing is a common strategy used by banks to promote their financial products and services. Term deposits are an attractive investment option for banks as they offer a low-risk source of funding. However, persuading customers to subscribe to a term deposit can be a challenging task.
Machine learning techniques can be used to develop predictive models that can help banks identify customers who are most likely to subscribe to a term deposit. These models use historical data on customer behavior and demographic information to identify patterns and predict future outcomes. By identifying the customers who are most likely to subscribe to a term deposit, banks can optimize their marketing strategies and improve their marketing campaign's effectiveness.
Python programming language provides a wide range of libraries and tools for data analysis and machine learning, making it an ideal choice for developing predictive models for the bank marketing task. In this project, we will explore various machine learning algorithms and techniques and compare their performance to identify the best model for the bank marketing task.

Experiment Methodology:

About Dataset
	The information relates to telephone-based direct marketing activities of a Portuguese banking institution. The classification's objective is to foretell whether a client will sign up for a term deposit (variable y).
	The information relates to direct marketing initiatives run by a Portuguese bank. On phone conversations, the marketing efforts were based. In order to determine if the product (term deposit deposit) would be subscribed ('yes') or not ('no'), it was sometimes necessary to make more than one contact with the same client. 
	Data consists of 21 features and 41189 entries including personal and professional information of the Client
	Dataset Link: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#

Attribute Information:
Input variables:
Bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (Categorical: 'no','yes','unknown')
6 - housing: has housing loan? (Categorical: 'no','yes','unknown')
7 - loan: has personal loan? (Categorical: 'no','yes','unknown')
Related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important information: If duration=0, the output goal will be "no," for example. However, the length is unknown before to making a call. After the call is over, y is also clearly recognised. Thus, if the goal is to build a realistic predictive model, this input should only be used for benchmarking purposes and should be removed.
#other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
Social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)
Target variable:
21 - y - Has the customer signed up for a term deposit? (Binary: 'yes','no')





Overall Process:
•	Visualization and insights from the data based on different factors and variables.
•	Data cleaning and Feature Engineering.
•	Building Different ML models and evaluate the models.
•	Tuning the Model using different hyperparameters and Grid Search and Cross Validation.


Project Design and Milestone

Pre-processing to done Before Modelling
•	Looking for NA values
•	Train Test Split
•	Label encoding
•	Choosing relevant variables
•	Up sampling Data
•	Vectorizing the data

Overall Process:
•	Visualization and insights from the data based on different factors and variables.
•	Data cleaning and Feature Engineering.
•	Building Different ML models viz. Logistic, Random Forest, GB etc. and evaluate the models.
•	Tuning the Model using different hyperparameters and Grid Search and Cross Validation.

Architecture and Technology
•	We use Python for Data Processing and Modelling and 
•	The system OS we have used is Linux.
•	Environment on Colab Notebook.
•	User → Google Linux Server → Google Storage Bucket → Processing on Google Server.

                                          






Data Visualisation and Exploration has been performed:
Insights with the graph are mentioned in the Notebook Itself:
Graphs consists of:
•	Bar Graph
•	Distribution Graph
•	Correlation plot 
•	Density plot 

                                                       

Here we can see Data set is imbalanced
   
 
We can see from the information above that the bank has phoned individuals ranging in age from 18 to 95. The bulk of the clients that phoned were, however, in their 30s to 40s. The age distribution is more normally distributed, with a lower standard deviation. The age group of 30 to 40 years old has a higher subscription rate as well.
 Therefore, there is a need to use Perform up sampling techniques. 
Here we can see some basic statistical aggregations of the numerical columns. Like Age of the customers are between 18-95,etc
 
Correlation Plot
•	'Campaign outcome' has a strong correlation with 'duration',
•	A moderate correlation between 'Campaign outcome' and 'previous contacts'.
•	Mild correlations between 'balance', 'month of contact' and 'number of campaigns'.

 

Pair Plot
•	Age's column has a minor right bias. Most of the clientele ranges in age from 25 to 65.
•	The number of customers with "blue-collar," "management," and "technician" jobs is higher in this dataset for "job," which is biased to the right.
•	From the definition of "marriage" above, we may deduce that married clients are more likely to sign up for a term deposit.
•	Higher education is more common among "Education" clients, although there are also a lot of clients whose education level is unclear.
•	The balance column has a lengthy tail to the right, which would indicate an anomaly.
•	Customers who do not have a mortgage are more likely to sign up for a term deposit.
•	The fact that "Duration" is skewed suggests that the majority of calls are brief, and that the dataset has a significant number of outliers.
•	Most customers have received contact from the bank between one and five times throughout this "campaign." Some customers have received calls from the bank more than 20 times, and we can tell that their likelihood of opening a term deposit is very low or almost zero. greater subscription rate when calls fall under 5. 
By looking at the distribution:
We can see that distribution is rightly skewed and most of the customers are between the age of 25-60
We can see from the information above that the bank has phoned individuals ranging in age from 18 to 95. The bulk of the clients that phoned were, however, in their 30s to 40s. The age distribution is more normally distributed, with a lower standard deviation. The age group of 30 to 40 years old has a higher subscription rate as well.
 





                 
DEFAULT, HOUSING, LOAN Distribution
•	The majority of customers do not currently have credit cards, and those without credit cards also make up a large portion of the subscriber base.
•	The majority of consumers have mortgages, while those who do not have mortgages are more likely to sign up for term deposits.
•	The majority of customers do not have personal loans, and the proportion of subscribers with and without personal loans is about equal.

 
Data Pre-processing and cleaning:
•	Looking for NA values
•	Train Test Split
•	Label encoding
•	Choosing relevant variables
•	Up sampling Data
•	Vectorizing the data
For Preparing the data below Operations were performed to standardize the data to understand by ML models
1.	Scaled the data.
2.	Splitted the data

After Up Sampling the data
                  
Results: 
Applying SMOTE for Up Sampling
Machine Learning Models used:
Primary Models used for:
	Logistic Regression: Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).
 
With Test Area Under ROC after the Cross validation 0.8958125339684428

	K-Nearest Neighbours: The k-nearest neighbors algorithm, sometimes referred to as KNN or k-NN, is a supervised learning classifier that employs proximity to produce classifications or predictions about the grouping of a single data point. Easy to interpret.
o	can handle categorical features.
o	can handle multiclass classification.
o	feature scaling not necessary
o	can capture non-linearities and feature interactions
 
We can see the Roc: Test Area Under ROC: 0.6680978913072564
Without Smote:
Logistic Regression
 
K-Nearest Neighbour
 
Without SMOTE we are getting better results; we will run and test with more models.



Using Deep Learning
A neural network design or a set of labelled data with several layers may both be used to train deep learning models. They occasionally perform better than human beings. These architectures remove the need for manual feature extraction by learning features directly from the data.
 
After Training the model for 100 epochs
 

Evaluation:
 
Hyper Parameter Tuning
The model has improved drastically after hyper Parameter Tuning.
 
After Hyper Parameter Tuning 
We found out that the Accuracy has Increased 
The other parameters like Precision, Recall and F1 Score also improved 
The Best Parameters were {'activation': 'sigmoid', 'batch_size': 64, 'epochs': 50}

Results: After ParamGrid and Cross validation:
	Logistic Regression	KNN	Deep Learning
Without Smote	0.89	0.90	0.88
Smote/hyperparameter	0.82	0.82	0.91


Conclusion:
As we can see that ROC Score for Boosting Algorithm is slightly higher than the other as it converts weak learner to strong learners.
Therefore, we can Move ahead with the Gradient Boosting with the Best Params which we have found using Grid Search and validated using Cross Validation.


References:
[1].	“Marketing analytics: Methods, practice, implementation, and links to other fields,” Expert Systems with Applications, 2018. S. L. France and S. Ghose,
[2].	S. Palaniappan, A. Mustapha, C. F. M. Foozy, and R. Atan, “Customer profiling using classification approach for bank telemarketing,” JOIV: International Journal on Informatics Visualization, vol. 1, no. 4-2, pp. 214–217, 2017.
[3].	Deal Banking Marketing Campaign Dataset with Machine Learning, Nutan, December 15th, 2021 https://medium.com/@nutanbhogendrasharma/deal-banking-marketing-campaign-dataset-with-machine-learning-9c1f84ad285d
[4].	Analysis on Bank Marketing Campaign for Portuguese Bank, Hegde Chaitra, May 26th,2018 
https://github.com/HegdeChaitra/Bank-Marketing-Campaign-Analysis

In the Above references has use either only basic models or have done some statistical analysis.
We have looking into the data distribution up sampled the data and trained the Ensemble models with Boosting to achieved best results and get higher data understanding and accuracy.
