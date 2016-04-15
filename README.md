Average Perceptron for NER
==========================

##Requirements: Python2.7

##Running
In the scripts folder run:  
`python train_ner.py -trainfile ../testdata/esp.train -testfile ../testdata/esp.testa -iter 2 -cf n`  
`train_ner.py` takes the following command line arguments:  
-trainfile path to file used to train model  
-testfile path to file used for evaluation  
-iter number of iterations  
-cf {y,n} print confusion matrix  

##Example Run
###To train with 2 iterations and do not print CF  
`python train_ner.py -trainfile ../testdata/esp.train -testfile ../testdata/esp.testa -iter 2 -cf n`  

###To train with 5 iterations and  print CF  
`python train_ner.py -trainfile ../testdata/esp.train -testfile ../testdata/esp.testa -iter 5 -cf y`  

###Shell Script to plot convergence
`./plot_convergence.sh ../testdata/esp.train ../testdata/esp.testa`
