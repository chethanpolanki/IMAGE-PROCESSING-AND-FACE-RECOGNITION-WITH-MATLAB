clc
clear all
close all
warning off
g=alexnet;%used in training upto 1000 images, predefined function
layers=g.Layers;
layers(23)=fullyConnectedLayer(3);
layers(25)=classificationLayer;
allImages=imageDatastore('database','IncludeSubfolders',true, 'LabelSource','foldernames');
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);%stochastic gradient descent with momentum
myNet=trainNetwork(allImages,layers,opts);
save myNet;