function [next_index, fnum] = MVAL_logistic_multi(X, Y, Dl, Du)
% This code refers to MVAL using logistic regression as the classifier on multi-class tasks
% 
%  Syntax
%       next_index = MVAL_logistic_multi(X, Y, Dl, Du)
%    
%  Description
%       Setting: We have a data set X with n instances, n1 of them are labeled (queried) and the others are unlabeled. The i-th instance is stored in X(i,:);
%       Task:   choose the next queried data point.
%         
%  Input: 
%       X     - A n x D array, the pool of availabel data, where N is the number of samples and D is the feature dimensionality;
%       Y     - A n x 1 vector, the true labels of samples in X, (multi-class)
%       Dl    - A 1 x n1 vector, the index of labeled instances
%       Du    - A 1 x (n-n1) vector, the index of unlabeled instances          
%       
% 
%  Output:
%        next_index - An integer, the index of the selected instance for query, i.e., the label of X(next_index,:) is to be queried;
%   
%  If you are using this code, please consider citing the following reference:
% 
%  Reference: Yang, Yazhou, and Marco Loog. "A variance maximization criterion for active learning." Pattern Recognition 78 (2018): 358-370.
%  
%  (C) Yazhou Yang, 2018
%  Email: yazhouy@gmail.com
%  Delft University of Technology
%%

% get the labeled data and unlabeled data
Ldata = X(Dl',:);
Llabel = Y(Dl',:);

Udata = X(Du',:);
Ulabel = Y(Du',:);

% train the model using liblinear, logistic regression, and calculate the posterior probability
model = lineartrain(Llabel, sparse(Ldata), '-s 0 -c 100 -B 1 -q');
[~, ~, dec_values] = linearpredict(Ulabel, sparse(Udata), model,'-b 1 -q');

% the number of classes
K = length(unique(Y));

if K<=2
   error('This function is for multi-class datasets! Please consider using MVAL_logistic_binary.m or MVAL_SVM_binary.m!');
end


sto = zeros(size(Du,2),size(Y,1),K^2);
ny = unique(Y);
%% generate the Retraining information matrices (RIMs)

% goes over all the unlabeled data 
for num = 1:size(Du,2)
    % add Du(num) to the labeled data 
    newDl = [Dl Du(num)];    
    newLdata = X(newDl',:);
            
    prb =[];
    for n_label = 1: K
        newLlabel = [Llabel; model.Label(n_label)];
        newmodel = lineartrain(newLlabel, sparse(newLdata),'-s 0 -c 100 -B 1 -q');
        % one-vs-all strategy for logistic regressioin
        [~, ~, prob1] = linearpredict(Y,sparse(X), newmodel,'-q');
        prob = 1./(1+exp(-prob1));        
        prb = [prb prob];                
    end
    sto(num,:,:) = prb;       
end
 
%% compute the variance of V1 and V2 for multi-class dataset
[N,M, ~]=size(sto);

Amx = zeros(N,M,K,K); 
for i=1:K    
    Amx(:,:,i,:) = sto(:,:, (i-1)*K+1:i*K);
end

L_mx = Amx(:,Du,:,:); 
clear Amx;

ABmx = zeros(N*K,N,K);
for i=1:K
    ABmx((i-1)*N+1:i*N,:,:)=squeeze(L_mx(:,:,i,:));
end
var11 = squeeze(var(ABmx));
var1 = sum(var11,2);
if N==1
   var1 =sum(var11); 
end

clear Abmx;

CDmx = zeros(N,N*K,K);
for i=1:K-1
    CDmx(:,(i-1)*N+1:i*N,:)= squeeze(L_mx(:,:,i+1,:))-squeeze(L_mx(:,:,i,:));
end
CDmx(:,(K-1)*N+1:K*N,:)= squeeze(L_mx(:,:,1,:))-squeeze(L_mx(:,:,K,:));

clear L_mx;

var22 = squeeze(var(CDmx,0,2));
var2 = sum(var22,2);
if N==1
   var2 =sum(var22); 
end

lsort = sort(dec_values,2,'descend');
bsb = lsort(:,1)-lsort(:,2);
ent = exp(-bsb);

Weight = repmat(ent',length(Du),K,K);

wCDmx = Weight.*CDmx;
clear CDmx Weight;
var44 = squeeze(var(wCDmx,0,2));
V2 = sum(var44,2);
if N==1
   V2 = sum(var44);
end

clear wCDmx;

var33 = repmat((ent.^2),1,K).*var11; 
if N==1
   var33 = repmat((ent.^2),1,K).*var11'; 
end

V1 = sum(var33,2);
if N==1
   V1 = sum(var33);
end

%% select the next point
value = V1.*V2;

[~,ind] = sort(value,'descend');

fnum = ind(1);
next_index = Du(fnum);

%% Dl and Du can be updated as follows:
% Dl = [Dl next_index];
% Du(fnum) = [];
end
