clc
clearvars
% Load the dataset
data = readtable('dataset3 edit.csv');
% Extract the features and target variable
features = table2array(data(:,1:8)); % extract columns 1 to 6 as features
target = table2array(data(:,9:11)); % extract columns 7 to 10 as target variable
% Split the data into training, validation, and test sets
[trainInd,valInd,testInd] = dividerand(size(features,1),0.7,0.15,0.15);
Xtrain = features(trainInd,:);
Ytrain = target(trainInd,:);
Xval = features(valInd,:);
Yval = target(valInd,:);
Xtest = features(testInd,:);
Ytest = target(testInd,:);
% Train an ANFIS model on the training set
opt = anfisOptions('InitialFis',4,'EpochNumber',20,'ValidationData',[Xval Yval]);
fis = anfis([Xtrain Ytrain],opt);
% Evaluate the model on the test set and compute the errors
Ypred = evalfis(Xtest,fis);
MSE = mean((Ytest - Ypred).^2);
RMSE = sqrt(MSE);
% Display the errors
disp(['Mean Squared Error: ' num2str(MSE)]);
disp(['Root Mean Squared Error: ' num2str(RMSE)]);