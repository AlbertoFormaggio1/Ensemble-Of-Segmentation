function saveImLb(trainingData,n,imageDir,labelDir)
    for i=1:size(trainingData,1)
        outI=strcat(imageDir,num2str(i+n),'.png');
        outL=strcat(labelDir,num2str(i+n),'.bmp');
        imwrite(uint8(trainingData{i,1}),outI);
        imwrite(logical(trainingData{i,2}(:,:,1)),outL);
    end
end

