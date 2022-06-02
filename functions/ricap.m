function TrainingData=ricap(data,siz)

for i = 1:size(data,1)   %Loop for each training image i pass to the function
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
end
TrainingData=data;
for j=1:size(data,1)
    %% step 1
    % randomly select four images k ∈ {1, 2, 3, 4}
    a=1+floor(rand*(size(data,1)-1));
    b=1+floor(rand*(size(data,1)-1));
    c=1+floor(rand*(size(data,1)-1));
    d=1+floor(rand*(size(data,1)-1));
    
    a_im=data{a,1};
    b_im=data{b,1};
    c_im=data{c,1};
    d_im=data{d,1};
    
    a_lb=logical(data{a,2});
    b_lb=logical(data{b,2});
    c_lb=logical(data{c,2});
    d_lb=logical(data{d,2});
    %% step 2
    % crop the images separately
    %corner-RICAP restricts the boundary position (w, h) within ranges close to the four corners.
    k=rand;
    u=0.437;
    if k>=0 && k<0.25  % corner up-left
        w=1+floor(rand*u*(siz-1));
        h=1+floor(rand*u*(siz-1));
    elseif k>=0.25 && k<0.5  % corner up-right
        w=1+floor((1-u)*siz+rand*(siz-((1-u)*siz)));
        h=1+floor(rand*u*(siz-1));
    elseif k>=0.5 && k<0.75  % corner down-left
        w=1+floor(rand*u*(siz-1));
        h=1+floor((1-u)*siz+rand*(siz-((1-u)*siz)));
    else  % corner down-right
        w=1+floor((1-u)*siz+rand*(siz-((1-u)*siz)));
        h=1+floor((1-u)*siz+rand*(siz-((1-u)*siz)));
    end
    
    wa=w;
    ha=h;
    wb=siz-w;
    hb=h;
    wc=w;
    hc=siz-h;
    wd=siz-w;
    hd=siz-h;
    
    xa=ceil(1+rand*((siz-1)-wa));
    ya=ceil(1+rand*((siz-1)-ha));
    xb=ceil(1+rand*((siz-1)-wb));
    yb=ceil(1+rand*((siz-1)-hb));
    xc=ceil(1+rand*((siz-1)-wc));
    yc=ceil(1+rand*((siz-1)-hc));
    xd=ceil(1+rand*((siz-1)-wd));
    yd=ceil(1+rand*((siz-1)-hd));
    
    a_crop_im=imcrop(a_im,[xa ya wa-1 ha-1]);
    b_crop_im=imcrop(b_im,[xb yb wb-1 hb-1]);
    c_crop_im=imcrop(c_im,[xc yc wc-1 hc-1]);
    d_crop_im=imcrop(d_im,[xd yd wd-1 hd-1]);
    
    a_crop_lb=imcrop(a_lb,[xa ya wa-1 ha-1]);
    b_crop_lb=imcrop(b_lb,[xb yb wb-1 hb-1]);
    c_crop_lb=imcrop(c_lb,[xc yc wc-1 hc-1]);
    d_crop_lb=imcrop(d_lb,[xd yd wd-1 hd-1]);

    
    %% step 3
    % patch the cropped images to construct a new image
    new_im=zeros(siz,siz,3);
    new_lb=zeros(siz,siz);

    new_im(1:ha,1:wa,:)=a_crop_im;
    new_im(1:hb,wa+1:end,:)=b_crop_im;
    new_im(ha+1:end,1:wc,:)=c_crop_im;
    new_im(ha+1:end,wc+1:end,:)=d_crop_im;
    
    new_lb(1:ha,1:wa)=a_crop_lb;
    new_lb(1:hb,wa+1:end)=b_crop_lb;
    new_lb(ha+1:end,1:wc)=c_crop_lb;
    new_lb(ha+1:end,wc+1:end)=d_crop_lb;
    
    
    TrainingData{j,1}=new_im;
    TrainingData{j,2}=new_lb;
end
end
