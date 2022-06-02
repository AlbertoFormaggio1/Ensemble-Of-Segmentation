function TrainingData= resizemix(data,siz)

for i=1:size(data,1)
    im1=data{i,1};
    lb1=data{i,2};
    k=round(1+rand*(size(data,1)-1));
    im2=data{k,1};
    lb2=data{k,2};
    im1=imresize(im1, [siz siz]);
    lb1=imresize(lb1, [siz siz]);
    dim=round(50+rand*90);
    im2=imresize(im2, [dim dim]);
    lb2=imresize(lb2, [dim dim]);
    x=ceil(1+rand*(224-dim));
    y=ceil(1+rand*(224-dim));
    im1(y:y+dim-1,x:x+dim-1,:)=im2;
    lb1(y:y+dim-1,x:x+dim-1)=lb2;
    TrainingData{i,1}=im1;
    TrainingData{i,2}=lb1;
end
TrainingData=imagesAugmentation_new(TrainingData);
end