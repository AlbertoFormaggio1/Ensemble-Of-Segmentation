function data=occlusion(data,siz)

row_im=zeros(1,siz,3);
row_lb=zeros(1,siz);
column_im=zeros(siz,1,3);
column_lb=zeros(siz,1);
for i = 1:size(data,1)   %Loop for each training image i pass to the function
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
    im=data{i,1};
    lb=data{i,2};
    t=rand;
    k=round(15+rand*15); %distance between two black lines
    if t>=0 && t<0.5  %rows
        j=1;
        while k*j<=siz
            im(j*k,:,:)=row_im;
            lb(j*k,:)=row_lb;
            j=j+1;
        end
    else   %columns
        j=1;
        while k*j<=siz
            im(:,j*k,:)=column_im;
            lb(:,j*k)=column_lb;
            j=j+1;
        end
    end
    data{i,1}=im;
    data{i,2}=lb;
end