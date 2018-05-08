% The Matlab code of "A Variance Maximization Criterion for Active Learning".
%  If you are using this code, please consider citing the following reference:
%
%  Yang, Yazhou, and Marco Loog. "A variance maximization criterion for active learning." Pattern Recognition 78 (2018): 358-370.
%
%  (C) Yazhou Yang, 2018
%  Email: yazhouy@gmail.com
%  Delft University of Technology
%


%% choice:   choose the classifier and type of datasets
%     1 -- MVAL using logistic regression as the classifier on binary datasets
%     2 -- MVAL using SVM as the classifier on binary datasets
%     3 -- MVAL using logistic regression as the classifier on multi-class datasets

choice = 1;

rng('default');
addpath(genpath(pwd));

% load binary data, please set choice = 1 or 2;
[Y, X] =libsvmread('australian_scale_real');

%% load multi-class data, please set choice = 3;
% 
% choice = 3;
% file = importdata('heart-cleveland_R.dat');
% X = file.data(:,2:end-1);
% Y = file.data(:,end);
% mi = unique(Y);
% for i=1:length(Y)
%     Y(i) =find(mi==Y(i));
% end
%%

num_repeat = 10; % repeat 10 times
num_query = 30; % budget of queried samples

average_LR = zeros(num_repeat,num_query);
average_SVM = zeros(num_repeat,num_query);

for i=1:num_repeat
    fprintf('repetition number: %i \n \n', i);
    [LR_acc, SVM_acc] = single_run_MVAL(X,Y,num_query, choice);
    
    average_LR(i,:) = LR_acc;
    average_SVM(i,:) = SVM_acc;
    fprintf('\n');
end

% plot the average performance
figure,
plot(mean(average_LR));
xlabel('Number of Labeled Instances')
ylabel('Average Accuracy (Logistic Regression)');

figure,
plot(mean(average_SVM));
xlabel('Number of Labeled Instances')
ylabel('Average Accuracy (SVM)');
