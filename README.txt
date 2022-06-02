Paper:
An empirical study on ensemble of segmentation approaches


The loss functions are available in the folder LossFunctions, for using them:
LGD,  dicePixelClassificationLayer('Name','labels'); %this an official matlab functin
LT, tverskyPixelClassificationLayer('Name',0.3,0.7);
Comb1, Test2PixelClassificationLayer('Name',0.3,0.7,4/3)
Comb2, Test3PixelClassificationLayer('Name',0.3,0.7,4/3)
Comb3, SSimDicePixelClassificationLayer('Name');
Comb4, MultiSSimFocaldicePixelClassificationLayer('Name',4/3);
LSTR, StructureLossPixelClassificationLayer('Name');
LBoundExpS, BoundExpStructurePixelClassificationLayer('Name',0.3,0.3,0.6,0.4);
LDiceBES, DiceBEStructurePixelClassificationLayer('Name',1,0.01);
LCS, CAStructurePixelClassificationLayer('Name',5);

The code (it explains also how to replace loss function) of the pretrained DeepLabV3+ (resnet101 as backbone), 
used in this paper is available at:
https://github.com/LorisNanni/Deep-ensembles-in-bioimage-segmentation

The file functions\README.txt explains different data augmentation approaches
see the file "ExampleDataAugmentationDA2.m"  to replicate the data augmentation named DA2 

The method named DA1 is very simple (matlab):
im=data{i,1};   	%consider the image i
lb=logical(data{i,2}); %consider the mask i
im=fliplr(im);
lb=fliplr(lb);
%save the new image and mask
    
im=data{i,1};   	%consider the image i
lb=logical(data{i,2}); %consider the mask i
im=flipud(im);
lb=flipud(lb);
%save the new image and mask
    
im=data{i,1};   	%consider the image i
lb=logical(data{i,2}); %consider the mask i
im=rot90(im);
lb=rot90(lb);
%save the new image and mask



Notice that we have removed images created by data augmentation that doesn't contain foreground images.
Simply, we delete the training images whose mask have less than 100 foreground pixels


Link for HardNet-mseg: https://github.com/james128333/HarDNet-MSEG
Link for PVT-Transformer:  https://github.com/DengPingFan/Polyp-PVT
we have used those toolboxes without any modification 


DATASETS:
POLYP:
The test sets for the polyp segmentation problem are available here: 
https://zenodo.org/record/5579392#.YdzC1f7MKUk

The training sets, coupled with data augmentation, for the polyp segmentation problem are available  here: 
https://zenodo.org/record/5579392#.YdzC1f7MKUk
the method named, in the chapter, DA2 (of the chapter) is "NewTRimageDA" of the above zenodo link

SKIN:
https://zenodo.org/record/5834916#.YkYQCShBw2x

LEUKOCYTE:
http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm

ButterFly:
http://www.josiahwang.com/dataset/leedsbutterfly/

Emicro:
https://figshare.com/articles/dataset/EMDS-6/17125025/1






