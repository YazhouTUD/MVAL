function [LR_acc, SVM_acc] = single_run_MVAL(X,Y, num_query, choice)
% % choice:   choose the classifier and type of datasets
%     1 -- MVAL using logistic regression as the classifier on binary datasets
%     2 -- MVAL using SVM as the classifier on binary datasets
%     3 -- MVAL using logistic regression as the classifier on multi-class datasets


% split the data for training and test
num_X = size(X,1);
num_train = floor(num_X/2); % half for training and half for test

rand_num = randperm(num_X);
train_list = rand_num(1: num_train);
test_list = rand_num(num_train+1:end);

Xtr = X(train_list,:);
Ytr = Y(train_list,:);
Xte = X(test_list,:);
Yte = Y(test_list,:);

%% randomly select one instance from each category as the initial labeled set
mi = unique(Y);
train_label = Y(train_list);
ini_set = zeros(1,length(mi));
for i=1:length(mi)
    index = find(train_label==mi(i));
    ini_set(i)=index(randi(length(index),1));
end

%% set up Dl and Du
% Dl  - the index of labeled instances
% Du  - the index of unlabeled instances
Dl = ini_set;
Du = 1:num_train;
Du(Dl) = [];
%% query num_query data points
for i = 1: num_query
    fprintf('iteration: %i  \n', i);
    switch choice        
        case 1
            [next_index, fnum] = MVAL_logistic_binary(Xtr, Ytr, Dl, Du);
        case 2
            [next_index, fnum] = MVAL_SVM_binary(Xtr, Ytr, Dl, Du);
        case 3
            [next_index, fnum] = MVAL_logistic_multi(Xtr, Ytr, Dl, Du);
        otherwise
            error('There are error about "choice". Please choose it from {1,2,3}!')
    end
    Dl = [Dl next_index];
    Du(fnum) = [];
end
%% compute the performance on test set

LR_acc = zeros(1,num_query);
SVM_acc = zeros(1,num_query);

for i=1:num_query
    xtr = Xtr(Dl(1:i+2),:);
    ytr = Ytr(Dl(1:i+2));
    logmodel = lineartrain(ytr, sparse(xtr), '-s 0 -c 100 -B 1 -q');
    [~, acc, ~] = linearpredict(Yte, sparse(Xte), logmodel,'-b 1 -q');
    LR_acc(i) = acc(1);
    
    model = mysvmtrain(ytr, xtr,'-t 0 -c 10 -q 1');
    [~, acc, ~] = mysvmpredict(Yte, Xte, model,'-q');
    SVM_acc(i) = acc(1);
end

end
