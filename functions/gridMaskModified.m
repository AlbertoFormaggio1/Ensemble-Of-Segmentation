function data= gridMaskModified(data,siz)

for i=1:size(data,1)
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
    im=double(data{i,1});
    lb=logical(data{i,2});
    mask_im=ones(siz,siz,3);
    mask_lb=ones(siz,siz);
    d=20+rand*70; %length of one unit
    deltaX=rand*(d-1); %distances between the first intact unit and boundary of the image.
    deltaY=rand*(d-1); %distances between the first intact unit and boundary of the image.
    r=0.6;
    l=round(d-d*r); % length  of the square of zeros
    square_im=zeros(l,l,3);
    square_lb=zeros(l,l);
    x=deltaX-l;
    if x<=0
        x=1;
    end
    y=deltaX-l;
    if y<=0
        y=1;
    end
    M=floor((siz-x)/d); %number of units
    N=floor((siz-y)/d); %number of units
    for t=0:M
        for s=0:N
            r1=ceil(y+s*d);
            r2=ceil(y+s*d+l-1);
            c1=ceil(x+t*d);
            c2=ceil(x+t*d+l-1);
            if r2>siz | c2>siz
                break
            end
            mask_im(r1:r2,c1:c2, :)=square_im;
            mask_lb(r1:r2 , c1:c2)=square_lb;
        end
    end    
    data{i,1}=im.*mask_im;
    data{i,2}=lb.*mask_lb;
end
end