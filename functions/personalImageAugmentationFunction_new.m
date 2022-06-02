function data = personalImageAugmentationFunction_new(data,siz)

for i = 1:size(data,1)   %Loop for each training image i pass to the function
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    lab_he = rgb2lab(im);  %I convert the image to LAB image
    ab = lab_he(:,:,2:3);
    ab = im2single(ab);
    nColors = 3;
    pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);    %segment the image in 3 colors -> I will have 3 clusters
    mask1 = pixel_labels==1;
    cluster1 = im .* uint8(mask1);
    mask2 = pixel_labels==2;
    cluster2 = im .* uint8(mask2);
    mask3 = pixel_labels==3;
    cluster3 = im .* uint8(mask3);
    
    %sum the total black pixels for each cluster
    totBlackPixel1=sum(cluster1(:) == 0);
    totBlackPixel2=sum(cluster2(:) == 0);
    totBlackPixel3=sum(cluster3(:) == 0);
    
    
    %fprintf("NUMBER OF BLACK PIXEL OF CLUSTER 1= %d \n",totBlackPixel1);
    %fprintf("NUMBER OF BLACK PIXEL OF CLUSTER 2= %d \n",totBlackPixel2);
    %fprintf("NUMBER OF BLACK PIXEL OF CLUSTER 3= %d \n",totBlackPixel3);
    
    %I compose the image as the sum of the images with more black pixels
    %composedImage=cluster1+cluster2+cluster3;
    if(totBlackPixel1>totBlackPixel2 && totBlackPixel1>totBlackPixel3)
        if(totBlackPixel2>totBlackPixel3)
            cluster3 = jitterColorHSV(cluster3,'Brightness',[-0.20 -0.15]);
            %composedImage=cluster1+cluster2;
        else
            cluster2 = jitterColorHSV(cluster2,'Brightness',[-0.20 -0.15]);
            %composedImage=cluster1+cluster3;
        end
    end
    if(totBlackPixel2>totBlackPixel1 && totBlackPixel2>totBlackPixel3)
        if(totBlackPixel1>totBlackPixel3)
            cluster3 = jitterColorHSV(cluster3,'Brightness',[-0.20 -0.15]);
            %composedImage=cluster2+cluster1;
        else
            cluster1 = jitterColorHSV(cluster1,'Brightness',[-0.20 -0.15]);
            %composedImage=cluster2+cluster3;
        end
    end
    if(totBlackPixel3>totBlackPixel2 && totBlackPixel3>totBlackPixel1)
        if(totBlackPixel2>totBlackPixel1)
            cluster1 = jitterColorHSV(cluster1,'Brightness',[-0.20 -0.15]);
            %composedImage=cluster3+cluster2;
        else
            cluster2 = jitterColorHSV(cluster2,'Brightness',[-0.20 -0.15]);
            %composedImage=cluster3+cluster1;
        end
    end
    composedImage=cluster1+cluster2+cluster3;
    angles = 0:90:270;
    tform = randomAffine2d('Rotation',@() angles(randi(4)));
    outputView = affineOutputView(size(composedImage),tform);
    data{i,1} = imwarp(composedImage,tform,'OutputView',outputView); %I create a copy of the i-th rotated by + -90 degrees
    data{i,2} = imwarp(lb,tform,'OutputView',outputView);
end
end