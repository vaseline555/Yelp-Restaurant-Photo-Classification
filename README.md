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
* [ ] Ensembled model
* [ ] Application of unsuperivsed learning (mainly clustering), instead of averaging features,  
on a feature engineering step to find a representative feature for each business id
* [ ] Constructing an end-to-end neural network model

<hr>
By. Seokju Hahn / https://www.kaggle.com/ggouaeng / sjhahn11512@naver.com
<pre>
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNmdddhhyysssoooo+++++++++++++++++oooosssyyhhddmmNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNmddhysso++++++++++++++++++++++++++++++++++++++++++++++++++++++oosyyhdmNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNmdhysso++++++++++++++++++++oooosssyyyyyhhhhhhddddddddddddddddhhhhhyyyysssoooo++++osyydmNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMNmdhys++++++++++++++++++oosssyyhhdddddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmddddddddhhyysooo+osydmNMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMNmhyso+++++++++++++++oossyyhhdddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddddhyysoosyhmNMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMmdhso++++++++++++++oosyyhhdddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddhhyssyhdNMMMMMMMMMMMMM
MMMMMMMmdys++++++++++++++ossyhhhddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddhyyhdmNMMMMMMMM
MMNmhso++++++++++++oosyhhhdddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddhhdmNMMMM
ms++++++++++++ossyhhdddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddddmM
++++++++++osyhhdddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdm
+++++osyhhdddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
+osyhhdddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
hhdddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
dddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmddddddddddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmddddddddddddddddddddddddddddddddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddddddddddddddddddddddddddddddddddddddddddddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmdo++oshmmmmmmmmmmmmmmmhso+oddddddddddddddddddddddddddddddddddddddddddddddddddddddddyoooohdddsoshdddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmm+    `smmmmmmmmmmmmms`   odddddddddddddddddddddddddhhhhhhhhhhhhhhhhhhhhhhhhhdddddd/    ydd/   .dddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmmd:    `ymmmmmmmmmmdh`   /ddddddddddddddddddhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhd/    yddo.``:ddddddddddddddmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
mmmmmmd.    .dmmmmmmdddd-   -ddddhyyyyyyyhhhhhhhhhhhyysssssyhhhhhhhhhhhhhyssssyyhhhhhhd/    yhdhyyyhdddhhddddddhyyyyhdmmmmmmmmmmmdhhyyyhdmmmmmmmmmmmmm
mmmmmmmy`    :dmmdddddd/   `hds/-.```````.:ohhhhhs/-.``` ```.:oyhhhhhyo:.```````./shhhd/    yhd/....hdd-..:ss:.``````./ymmmmmmho:.``````.-+ymmmmmmmmmm
mmmmmmmms     oddddddds   `sh-   `/syys/`   -yhho`   -osss/.   -yhhy+`   ./++/-`   -yhd/    yhd:    hhd.   ` .:+o+/.   `sdmmd/`   -/++/-   `/dmmmmmmmm
mmmmmmmmm/    `ydddddh`   +d+....yhhhhhd+    /hd`   `yhhhhhh-...+hh:    +yhhhhdy`   .yd/    yhd:    hhd.    /hdddddh.   `ddd-   `sdmmmmds`   -dmmmmmmm
mmmmmmmmmd-    .hdddd-   :hdhhhhyyyyssss:    :hd:    .:+oosyyyyyyhs    -oooooooy/    :d/    shd:    hhd.   `hddddddd:    hd+    /yyyyyyyy-    smmmmmmm
mmmmmmmmmdh.    :ddd/   .hhhhs+:..```````    :hhh+.       ``.:+syh/                  .h/    shd:    hhd.   .hhhddddd/    hd.                  /mmmmmmm
mmmmmmmddddy`    +do   `yhhy-`   -/osssy+    :yyhhhyo/:-.``    `+h/    :oooooooooooo++h/    syd:    yhd.   .hhhhhhdd/    hd.    /oooooooooooooymmmmmmm
mmmmmdddddddo    `o`   ohhd:    /hhhhhhh+    :yo++++yhhhyyyo`   `hs    .yyyyyyyyysssssh/    syd:    yhd.   .hhhhhhhd/    hd+    /ddmmmmmmdhhhhdmmmmmmm
mmmmddddddddd/        +hhhd:    :yhhyyyo`    -ho    -syyyyyh.   `yd/    -ossysso-```.oh/    syd:    yhd.   .hhhhhhhd/    hdd:    /yddddhs.```-ddhhdmmm
mmdddddddddddd-      :hhhhhh-    .:::-. .`   `ydo.   `-:::-.   .+syds-`   `...`   `-oyh/    syd:    yyd.   .hhhhhhhd/    hdddo.   `.---.   .+dhy++yhmm
mdddddddddddddh::::::yhhhhhhhs/-.....-/ohy/--.oyhhs/-.......-/+sssssyhs+:-.....-/+oyyyho:::-syh+:::-yyd/::::hhhhhhhdo:::-hddddds+:......:+shmmdyssydmm
dddddddddddddhdddhhhhhhhhhhhhhhhhyyyyyyyyyyyyyyyyyyyyyyyyyyyyssssssssssyyyyyyyyyssssssyyyyyyyyyyyyyyyyhhhhhyhhhhhhhhhhhdhhhddddddddddddddmmmmmmmddmmmm
ddddddddddddhhhhhhhhhhhhhhyyyyyyyyyyyyyyyyyyyssssssssssssssssssssssssssssssssssssssssssssssyyyyyyyyyyyyyyyyyyyhhhhhhhhhhhhhhhdddddddddddddmmmmmmmmmmmm
ddddddddddhhhhhhhhhhhhhhyyyyyyyyyyyyyyyyyysssssssssssssssssssssssssssssssssssssssssssssssssssssyyyyyyyyyyyyyyyyyhhhhhhhhhhhhhhdddddddddddddmmmmmmmmmmm
dddddddddhhhhhhhhhhhhhhyyyyyyyyyyyyyyyyssssssssssssssssssssssssssssssssssssssssssssssssssssssssssyyyyyyyyyyyyyyyyyhhhhhhhhhhhhhdddddddddddddmmmmmmmmmm
ddddddddhhhhhhhhhhhhhyyyyyyyyyyyyyyyysssssssssssssssssssssssoooooooooooooooossssssssssssssssssssssssyyyyyyyyyyyyyyyhhhhhhhhhhhhhhddddddddddddmmmmmmmmm
dddddddhhhhhhhhhhhhhyyyyyyyyyyyyyyysssssssssssssssssssoooooooooooooooooooooooooooossssssssssssssssssssyyyyyyyyyyyyyyyhhhhhhhhhhhhhddddddddddddmmmmmmmm
ddddddhhhhhhhhhhhhyyyyyyyyyyyyyyysssssssssssssssssoooooooooooooooooooooooooooooooooooossssssssssssssssssyyyyyyyyyyyyyyhhhhhhhhhhhhhddddddddddddmmmdddd
dddddhhhhhhhhhhhhyyyyyyyyyyyyyyssssssssssssssssoooooooooooooooooooooooooooooooooooooooooossssssssssssssssyyyyyyyyyyyyyyhhhhhhhhhhhhhddddddddddddddddhy
ddddhhhhhhhhhhhhyyyyyyyyyyyyyysssssssssssssssooooooooooooooooooooooooooooooooooooooooooooooosssssssssssssssyyyyyyyyyyyyyhhhhhhhhhhhhddddddddddddhhyo++
dddhhhhhhhhhhhhyyyyyyyyyyyyyyssssssssssssssooooooooooooooooooooooooooooooooooooooooooooooooooossssssssssssssyyyyyyyyyyyyyhhhhhhhhhhhhddddddhhyso++++++
ddhhhhhhhhhhhhyyyyyyyyyyyyyssssssssssssssooooooooooooooooooo+++++++++++++++++oooooooooooooooooossssssssssssssyyyyyyyyyyyyyhhhhhhhhhddddhhyso++++++++++
Nmdhhhhhhhhhhhyyyyyyyyyyyysssssssssssssooooooooooooooooo+++++++++++++++++++++++++oooooooooooooooosssssssssssssyyyyyyyyyyyyhhhhhhdhhhyso++++++++++++oym
MMMMNmdhyyhhhhyyyyyyyyyyysssssssssssssooooooooooooooo+++++++++++++++++++++++++++++++oooooooooooooosssssssssssssyyyyyyyyhhhhdhhhyso+++++++++++++oydmMMM
MMMMMMMMMmdhysyyyyyyyyyyyssssssssssssoooooooooooooo+++++++++++++++++++++++++++++++++++oooooooooooooossssssssssyyyhhhhhhhyysoo+++++++++++++oshdNMMMMMMM
MMMMMMMMMMMMMNmhyssssyyyyyssssssssssooooooooooooo+++++++++++++++++++++++++++++++++++++++ooooooooooosssssyyyhhhhhhhysso+++++++++++++++oyhdNMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMmdhsoosssyyyyyssssooooooooooo++++++++++++++++////////++++++++++++++++oooooossyyyhhhhhhhyssoo++++++++++++++++oshdmNMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMNmdhys++oosssyyyyyyysssoooo++++++++++++//////////+++++oooosssyyyyhhhhhhyyyssoo++++++++++++++++++osyhdNMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNmdhyso+++++oooosssyyyyyyyyyyyyyyyhhhhhhhhhhhhhyyyyyssssoooo+++++++++++++++++++++osyhdmNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNmmdhyyso+++++++++++++++++++++++++++++++++++++++++++++++++++++oosyyhdmNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNmmddhhyyysssooooo+++++++++++ooooosssyyyhhdddmNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
</pre>
