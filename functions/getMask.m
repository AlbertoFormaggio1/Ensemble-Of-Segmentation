%to save binary mask
function [bMask,mask]=getMask(net, img,siz)
    origSize=size(img);
    
    img=imresize(img,[siz siz]);
    
    [C,~,allScores]  = semanticseg(img, net);
    mask=1-allScores(:,:,1);
    mask=imresize(mask, [origSize(1) origSize(2)]);
    %save BMP in the output dir
    bMask=(C=='one');
    bMask=imresize(bMask, [origSize(1) origSize(2)]);
    
end