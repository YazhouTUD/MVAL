# MVAL: Maximizing Variance for Active Learning
The Matlab code of "A Variance Maximization Criterion for Active Learning". 

Reference: Yang, Yazhou, and Marco Loog. "A variance maximization criterion for active learning." Pattern Recognition 78 (2018): 358-370.

1. Installation
We already include all the libaries in the folder "lib". These libaries are downloaded from libsvm (https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and liblinear (https://www.csie.ntu.edu.tw/~cjlin/liblinear/). We provided the pre-compiled libaries on Mac, Windows and Ubuntu systems. 

If you have error with these libaries, please download the libsvm and liblinear packages and compile them on your own computer. Subsequently, rename these compiled libaries as follows:

(Libsvm):     svmpredict.mexmaci64 --> mysvmpredict.mexmaci64
(Libsvm):     svmtrain.mexmaci64 --> mysvmtrain.mexmaci64
(Liblinear):  predict.mexmaci64 --> linearpredict.mexmaci64
(Liblinear):  train.mexmaci64 --> lineartrain.mexmaci64

The file-name extensions are .mexmaci64 (on Apple Mac (64-bit)), .mexw64 (on Windows (64-bit)), or .mexa64 (on Linux (64-bit)).

2. Usage

Just run "Example_MVAL.m" to see how it works.

Just choose "choice" from {1,2,3} depending on the used classifier and datasets:

choice:  choose the classifier and type of datasets
    1 -- MVAL using logistic regression as the classifier on binary datasets (MVAL_logistic_binary.m).
    2 -- MVAL using SVM as the classifier on binary datasets (MVAL_SVM_binary.m).
    3 -- MVAL using logistic regression as the classifier on multi-class datasets (MVAL_logistic_multi.m).

If you have any questions, please feel free to connect with me (yazhouy@gmail.com).

If you are using this code, please consider citing the following reference:

Bibtex:
@article{yang2018variance,
  title={A variance maximization criterion for active learning}, \\
  author={Yang, Yazhou and Loog, Marco},
  journal={Pattern Recognition},
  volume={78},
  pages={358--370},
  year={2018},
  publisher={Elsevier}
}
