# Kaggle-Yelp-Restaurant-Photo-Classification
Past Kaggle competition : Yelp Restaurant Photo Classification  
(https://www.kaggle.com/c/yelp-restaurant-photo-classification) 

By. Seokju Hahn / https://www.kaggle.com/ggouaeng / sjhahn11512@naver.com
  
## Requirements
<pre>
python==3.6.5 (Intel distribution for Python)
numpy==1.14.3
pandas==0.22.0
tqdm==4.23.4
Keras==2.1.6
scikit-learn==0.19.1
xgboost==0.72
</pre>

## Steps to solve the problem
1. Extract bottleneck features using a pre-trained model (ResNet50)
2. Manipulate business features
3. Construct classifiers
  * Support Vector Machine
  * XGBoost
  * Multi-Layered Perceptron
4. Predict labels

## Results
*Best scores of each classifier*  

Model        | Private Score | Public Score 
------------ | ------------ | ------------ 
SVM | 0.80254 | 0.79200 
XGBoost | 0.81438 | 0.80144 
MLP | 0.82029 | 0.81399   

## Development environment
* CPU : Intel Xeon 2 Cores
* RAM : 8GB
* GPU : Tesla K80 12GB
