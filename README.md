# Kaggle-Yelp-Restaurant-Photo-Classification
Past Kaggle competition : Yelp Restaurant Photo Classification  
(https://www.kaggle.com/c/yelp-restaurant-photo-classification) 

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
* Extract bottleneck features using a pre-trained model (ResNet50)
* Manipulate business features
* Construct classifiers
  * Support Vector Machine
  * XGBoost
  * Multi-Layered Perceptron
* Predict labels

## Results
*Best scores of each classifier*  

Model        | Private Score | Public Score 
------------ | ------------ | ------------ 
SVM | 0.80254 | 0.79200 
MLP | 0.82029 | 0.81399  
XGBoost | 0.81438 | 0.80144 
 

## Development environment
* CPU : Intel Xeon 2 Cores
* RAM : 8GB
* GPU : Tesla K80 12GB

## Things to do
* [ ] Model pipelining
* [ ] Application of unsuperivsed learning (mainly clustering), instead of averaging features,  
on a feature engineering step to find a representative feature for each business id
* [ ] Constructing an end-to-end neural network model

<hr>
By. Seokju Hahn / https://www.kaggle.com/ggouaeng / sjhahn11512@naver.com
