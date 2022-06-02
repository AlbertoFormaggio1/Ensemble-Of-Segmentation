function data=occlusion_2(data,siz)

for i = 1:size(data,1)   %Loop for each training image i pass to the function
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
end
for i = 1:size(data,1)
    im=data{i,1};
    lb=data{i,2};
    % the size of the rectangles is chosen randomly
    m=round(4+rand*10);
    n=round(4+rand*10);
    rect_im=zeros(m,n,3);
    rect_lb=zeros(m,n);
    N=7+rand*7; %number of rectangles
    for j=1:N
        x=round(1+rand*(siz-n-1));
        y=round(1+rand*(siz-m-1));
        im(y:y+m-1,x:x+n-1,:)=rect_im;
        lb(y:y+m-1,x:x+n-1)=rect_lb;
    end
    data{i,1}=im;
    data{i,2}=lb;
end
end