

<h1 align = 'center'> Stroke Prediction </h1>
<p align = 'center'>
  
 <a href = 'https://www.python.org/downloads/release/python-396/'>
   <img src = 'https://img.shields.io/badge/python-v3.9-blue'>
 </a>

 <a href="https://github.com/orkunaran/Stroke-Prediction/issues">
  <img alt="GitHub issues" src="https://img.shields.io/github/issues/orkunaran/Stroke-Prediction">
 </a>
 
 <a href="https://github.com/orkunaran/Stroke-Prediction/blob/main/LICENSE.md">
  <img alt="GitHub license" src="https://img.shields.io/github/license/orkunaran/Stroke-Prediction">
 </a>
 
 <img src = 'https://badges.pufler.dev/visits/orkunaran/Stroke-Prediction'>
  
 <img src = 'https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/asset/c6bf82ef-2745-4d3d-8ff3-2fb2fe114a8d/gr1.jpg'>

</p>


In this notebook I tried to predict whether a person would have a stroke or not with given parameters. The most eye catching part of this dataset that it has imbalanced target class. 

I followed a general approach on the dataset;

1. Data Cleaning and EDA
2. Scaling the data
3. Dealing with Class Imbalance with SMOTEENN method
4. Comparing Machine Learning models 
5. Hyperparameter tuning with Randomized Search CV
6. Finding best parameters with Grid Search CV

In addition to these, I also tried Future Selection and tried to implement the model with selected features. However, feature selection didn't changed the model accuracy significantly.

## Update v2.0

Recently I learned that resampling is not a good idea in terms of preseving data integrity, that it might change the variance in the data. Then I started to look for option to deal with CI, found books - medium articles and kaggle notebooks. In this notebook I tried class weights in sklearn and scale_pos_weight in xgboost classifiers. Also, I calculated tpr, fpr and gmeans scores to determine prediction probability treshholds. Last, I created an APP with streamlit to deploy the model. 


