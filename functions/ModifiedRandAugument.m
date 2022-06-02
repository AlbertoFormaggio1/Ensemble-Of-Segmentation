function data=ModifiedRandAugument(data,siz)

P=0.7; %probability that the chosen transformation is actually performed
for i = 1:size(data,1)
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
    im=data{i,1};    	   %consider the image i
    lb=logical(data{i,2}); %consider the label i
    %I randomly choose one of the 13 transformations present in the color category
    color=rand;
    tras_col=rand;
    if color>=0 && color<1/13 %Fusion
        ind=floor(1+rand*size(data,1));
        im2=imresize(data{ind,1},[siz siz]);
        lb2=imresize(data{ind,2},[siz siz]);
        im=imfuse(im,im2,'blend');
        lb=imfuse(lb,lb2,'blend');
    elseif color>=1/13 && color<2/13 %Gauss_noise
        sigma = 0.2*rand;
        im = imgaussfilt(im,sigma);
        im = imnoise(im,'gaussian');
    elseif color>=2/13 && color<3/13 %Saturation
        im = jitterColorHSV(im,'Saturation',[0.6 1]); 
    elseif color>=3/13 && color<4/13 %Contrast
        im = jitterColorHSV(im,'Contrast',[0.6 1]);
    elseif color>=4/13 && color<5/13 %Brightness
        im = jitterColorHSV(im,'Brightness',[0.6 1]);
    elseif color>=5/13 && color<6/13 %Sharpness
        im=imsharpen(im,'Radius',2,'Amount',1);
    elseif color>=6/13 && color<7/13 %Motion
        H = fspecial('motion',20,45);
        im = imfilter(im,H,'replicate');
    elseif color>=7/13 && color<8/13 %Equalize
        im=histeq(im);
    elseif color>=8/13 && color<9/13 %Equalize_yuv
        R=im(:,:,1);
        G=im(:,:,2);
        B=im(:,:,3);
        Y = 0.299 * R + 0.587 * G + 0.114 * B;
        U = -0.14713 * R - 0.28886 * G + 0.436 * B;
        V = 0.615 * R - 0.51499 * G - 0.10001 * B;
        im(:,:,1)=Y;
        im(:,:,2)=U;
        im(:,:,3)=V;
        im=histeq(im);
    elseif color>=9/13 && color<10/13 %disk_filter
        H = fspecial('disk',10);
        im = imfilter(im,H,'replicate');
    elseif color>=10/13 && color<11/13 %salt&pepper_noise
        im = imnoise(im,'salt & pepper');
    elseif color>=11/13 && color<12/13 %hue
        J1 = jitterColorHSV(im,'Hue',[0.6 1]);
    else %local_contrast
        edgeThreshold = 0.4;
        amount = 0.5;
        im = localcontrast(im, edgeThreshold, amount);
    end
    
    %I randomly choose one of the 8 transformations present in the shape category
    tras_shape=rand;
    if tras_shape>=P
        shape=rand;
        if shape>=0 && shape<1/8 %Rotate
            tform = randomAffine2d('Rotation',[-40 40]);
            outputView = affineOutputView(size(im),tform);
            im = imwarp(im,tform,'OutputView',outputView);
            lb = imwarp(lb,tform,'OutputView',outputView);
        elseif shape>=1/8 && shape<2/8  %Flip
            im=flip(im);
            lb=flip(lb);
        elseif shape>=2/8 && shape<3/8  %ShearX
            tform = randomAffine2d('XShear',[-15 15]);
            outputView = affineOutputView(size(im),tform);
            im = imwarp(im,tform,'OutputView',outputView);
            lb = imwarp(lb,tform,'OutputView',outputView);
        elseif shape>=3/8 && shape<4/8  %ShearY
            tform = randomAffine2d('YShear',[-15 15]);
            outputView = affineOutputView(size(im),tform);
            im = imwarp(im,tform,'OutputView',outputView);
            lb = imwarp(lb,tform,'OutputView',outputView);
        elseif shape>=4/8 && shape<5/8  %XTranslation
            tform = randomAffine2d('XTranslation',[-50 50]);
            outputView = affineOutputView(size(im),tform);
            im = imwarp(im,tform,'OutputView',outputView);
            lb = imwarp(lb,tform,'OutputView',outputView);
        elseif shape>=5/8 && shape<6/8  %Scale
            tform = randomAffine2d('Scale',[0.8,1.2]);
            outputView = affineOutputView(size(im),tform);
            im = imwarp(im,tform,'OutputView',outputView);
            lb = imwarp(lb,tform,'OutputView',outputView);
        elseif shape>=6/8 && shape<7/8  %YTranslation
            tform = randomAffine2d('YTranslation',[-50 50]);
            outputView = affineOutputView(size(im),tform);
            im = imwarp(im,tform,'OutputView',outputView);
            lb = imwarp(lb,tform,'OutputView',outputView);
        else  %Cutout
            targetSize = [200 100];
            win = centerCropWindow2d(size(im),targetSize); 
            im = imcrop(im,win);
            lb= imcrop(lb,win);
        end
    end
    data{i,1} =imresize(im,[siz siz]);
    data{i,2} =imresize(lb,[siz siz]);
end
end