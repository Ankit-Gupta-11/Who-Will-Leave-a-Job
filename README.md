
# Who will Leave the Job?

<p align="justify">
Who will leave the job is a machine learning project which predicts who will not accept the job or who will not be eligible for the job by considering a few trends of the intern during the training process. 
</p>

## Table of Content

* [About The Project](#1)
* [About Data](#2)
    - [Sub-tasks]()
* [Methodlogy](#3)
* [Evaluation Criteria](#4)
* [Deployment](#5)
* [Reference](#9)

## About The Project

<p align="justify">
A company which is active in Big Data and Data Science wants to 
hire data scientists among people who successfully pass some 
courses which conduct by the company. Many people signup for 
their training. Company wants to know which of these candidates 
are really wants to work for the company after training or looking
for a new employment **because it helps to reduce the cost and time
as well as the quality of training or planning the courses** 
and categorization of candidates. 
Information related to demographics, education, experience are 
in hands from candidates signup and enrollment.
</p>

## About The Data

The dataset used in the project is Tabular data taken from a Kaggle dataset.  
[HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)

<p align="justify">
This dataset was designed to understand the factors that lead a person
to leave a current job for HR research too. By model(s) that
uses the current credentials,demographics,experience data
you will predict the probability of a candidate looking for
a new job or will work for the company, as well as interpreting 
affected factors on employee decision.
</p>

![](images\Data-Info.png)

Note few points 

- The dataset is imbalanced.
- Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality.
- Missing imputation can be a part of your pipeline as well.

<p align="justify">
To overcome the above challenges various approaches were used(see the notebook for more deep understanding). 
</p>
## Methodology

![](images\Flow-Chart-Who-Will-Leave-The-Job.jpg)

<p align="justify">
For the detail Understanding of the methodology please visit the notebook present in the master repository. 
</p>

## Results

### Learning Curve

It Help us determine if model is overfitting or underfitting.

![Learning Curve](images\Learning-Curve.png)


### AUC Curve

Evaluation Results of the model.

![AUC Curve](images\AUC.png)



<p align="justify">

<p align="justify">
<p align="justify">
<p align="justify">
<p align="justify">