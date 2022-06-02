function TrainingData= ModifiedAttentiveCutMix(data,siz)

TrainingData=data;
for i=1:size(data,1)
    im1=double(data{i,1});
    lb1=logical(data{i,2});
    k=i;
    while k==i
        k=round(1+rand*(size(data,1)-1));
    end
    im2=double(data{k,1});
    lb2=logical(data{k,2});
    
    im1 =imresize(im1,[siz siz]);
    lb1 =imresize(lb1,[siz siz]);
    im2 =imresize(im2,[siz siz]);
    lb2 =imresize(lb2,[siz siz]);
    
    dim=7; %dim*dim grid map
    N=round(3+rand*14);   %number of patches
    mask_im=zeros(siz,siz,3);
    mask_lb=zeros(siz,siz);
    SIZ=siz/dim; %dim of a square
    m_im=ones(SIZ,SIZ,3);
    m_lb=ones(SIZ,SIZ);
    ones_im=ones(siz,siz,3);
    ones_lb=ones(siz,siz);
    for j=1:N
        x=round(1+rand*(dim-1));
        y=round(1+rand*(dim-2));
        r1=(y-1)*SIZ+1;
        r2=y*SIZ;
        c1=(x-1)*SIZ+1;
        c2=x*SIZ;
        mask_im(r1:r2,c1:c2,:)=m_im;
        mask_lb(r1:r2,c1:c2)=m_lb;    
    end
    
    new_im=(mask_im.*im2)+(ones_im-mask_im).*im1;
    new_lb=(mask_lb.*lb2)+(ones_lb-mask_lb).*lb1;
    TrainingData{i,1}=new_im;
    TrainingData{i,2}=new_lb;
end
end
