%example for creating artificial images
%the new imagese will be stored in TEMPimg  and TEMPmask (see below)
%notice that then you should concatenate these folders with ones
%that store original images/mask

clear all
warning off

%address where the folder with the new images will be built,
% and that contains folders with the original images/masks (see below)
address='D:\c\Lavoro\DATA\DATA\kvasir\AltriPolipi';

%the new imagese will be stored in TEMPimg  and TEMPmask
%notice that then you should concatenate these folders with ones
%that store original images/mask
mkdir(strcat(address,'\TEMPimg\'))
mkdir(strcat(address,'\TEMPmask\'))
imageTempDir = strcat(address,'\TEMPimg\');
labelTempDir = strcat(address,'\TEMPmask\');


%read address of the training images and their masks
imageDir = strcat(address,'\NewTRimage\');
labelDir = strcat(address,'\NewTRmask\');


% Create a |pixelLabelDatastore| holding the ground truth pixel labels for 
% the training images.
imds = imageDatastore(imageDir);
pxds = imageDatastore(labelDir);
trainingData = combine(imds,pxds);
data=readall(trainingData);
sizeInput=352; %input size, notice that it varies among the different datasets (see the dataset description in the paper), in some dataset 352 in other 513

aug(1:15)=0;%aug(i)=1 if you want to use that data augmentation approach
aug([ 14 15])=1;%imagesTrasformation_new + demo_new is used in the tests of this chapter

numberIT=1;
DimT=size(data,1);%the new images will be named with a number higher than the number of images of the training data

if aug(14)
    k=demo_new(data,imageTempDir,labelTempDir,numberIT,sizeInput);
    numberIT=(k/size(data,1))+1;
end
if aug(15)
    [data1,data2,k]=imagesTrasformation_new(data,imageTempDir,labelTempDir,numberIT,sizeInput);
    numberIT=(k/size(data,1))+1;
    saveImLb(data1,numberIT*DimT,imageTempDir,labelTempDir);numberIT=numberIT+1;
    saveImLb(data2,numberIT*DimT,imageTempDir,labelTempDir);numberIT=numberIT+1;
end