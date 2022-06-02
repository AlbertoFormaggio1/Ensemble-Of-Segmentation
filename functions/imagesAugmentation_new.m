function data=imagesAugmentation_new(data,siz)

for i = 1:size(data,1)   %Loop for each training image i pass to the function
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
    im=data{i,1};    	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    im = jitterColorHSV(im,'Saturation',[-0.3 -0.1]); %Saturation
    im = jitterColorHSV(im,'Brightness',[-0.3 -0.1]); %Brightness
    im = jitterColorHSV(im,'Contrast',[1.2 1.4]); %Contrast
    sigma = 1+2*rand;
    im = imgaussfilt(im,sigma);    %Gauss_noise
    im = imnoise(im,'gaussian');
    angles = 0:90:270;
    tform = randomAffine2d('Rotation',@() angles(randi(4))); %rotation
    outputView = affineOutputView(size(im),tform);
    data{i,1}=imwarp(im,tform,'OutputView',outputView);
    data{i,2}=imwarp(lb,tform,'OutputView',outputView);
end
end