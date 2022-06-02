classdef CAStructurePixelClassificationLayer < nnet.layer.ClassificationLayer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        K = 5;
    end
    
    methods
        function layer = CAStructurePixelClassificationLayer(name,k)
            %Layer name
            layer.Name = name;
            
            layer.K = k;
            
            %Layer description
            layer.Description = "Contour-Aware Loss + Structure";
        end
        
        function loss = forwardLoss(layer,Y,T)
            %Versione senza dlarray: ho già testato io l'equivalenza tra
            %questa e la versione sotto
            se = strel('square',5);
            erodedT1 = imerode(logical(T),se);
            dilatedT1 = imdilate(logical(T),se);
            border = layer.K * (dilatedT1 - erodedT1);        
            M = dlarray(imgaussfilt(border,'FilterSize',5) + 1);
            %}
            
             T = dlarray(T);
%             
%             dilatedT = maxpool(T,[5,5],'padding','same','DataFormat','SSTU');
%             erodedT= 1 - maxpool(1-T,[5,5],'padding','same','DataFormat','SSTU'); 
%             border = layer.K * (dilatedT - erodedT); 
%             kernel = fspecial('gaussian', [5 5], 0.5);
%             bias = 0;
%             M = dlconv(border,kernel,bias,'padding','same','DataFormat','SSTU') + 1;
            
            bce = - (T.*log(Y+1e-8));  

            %Not normalized (come nel paper
            numObservations=size(Y,1)*size(Y,2);
            CLoss = sum(sum(sum(M.*bce,3),2),1)./numObservations;
            %{
            %Weight-normalized
            
            num = sum(sum(M.*bce,1),2);
            den = sum(sum(M,1),2);
            CLossc = num./den;
            CLoss = CLossc(:,:,1,:);
            %}
            

            weit = dlarray(zeros(size(T)));
            for Nimg=1:size(Y,4)
                P = Y(:,:,:,Nimg); %Predictions
                M = T(:,:,:,Nimg); %Mask                
                P = avgpool(P,[31,31],'Padding',15,'DataFormat','SST'); %It could also be Padding = same
                weit(:,:,:,Nimg) = 1+5*abs(P-M);                
            end  
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiou = 1 - (inter + 1)./(tot-inter+1);
            wiou = sum(wiou,3);
            
            N=size(Y,4);
            loss = sum(CLoss + wiou)/N;            
        end
    end
end

