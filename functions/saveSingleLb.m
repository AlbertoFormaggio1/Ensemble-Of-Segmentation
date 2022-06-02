
function k=saveSingleLb(imageDir,labelDir,k,im,lb)
outI=strcat(imageDir,num2str(k),'.png');
outL=strcat(labelDir,num2str(k),'.bmp');
imwrite(uint8(im),outI);
imwrite(lb(:,:,1),outL);
k=k+1;
end