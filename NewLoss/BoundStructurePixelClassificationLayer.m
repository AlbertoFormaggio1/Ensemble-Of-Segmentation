classdef BoundStructurePixelClassificationLayer < nnet.layer.ClassificationLayer
    %STRUCTURELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    %Epsilon value is used in order to prevent divisions by 0 which
    %generates nan values.
    properties (Constant)
        Epsilon = 1e-8;
    end
    
    methods
        function layer = BoundStructurePixelClassificationLayer(name)
            %Set layer name
            layer.Name = name;
            
            %Set layer description
            layer.Description = 'Weighted BCE + Weighted IoU';
        end
        
        function loss = forwardLoss(layer,Y,T)
            T = dlarray(T);
            %iterate through the images inside the batch
            %maybe i have to consider classes too
            
            for Nimg=1:size(Y,4)
                P = Y(:,:,:,Nimg); %Predictions
                M = T(:,:,:,Nimg); %Mask                
                P = avgpool(P,[31,31],'Padding',15,'DataFormat','SST'); %It could also be Padding = same
                weit(:,:,:,Nimg) = 1+5*abs(P-M);
                P1=((T(:,:,1,Nimg)));
                P2=((Y(:,:,1,Nimg)));
                P1=abs(avgpool(P1,6,'Stride',1,'DataFormat','SST')-0.5);
                P2=abs(avgpool(P2,6,'Stride',1,'DataFormat','SST')-0.5);
                DistBOUND(Nimg)=1-ssim(P1,P2);
            end
            DistBOUND=mean(DistBOUND);
            
            %P = avgpool(Y,[31,31],'Padding',15,'DataFormat','SST');
            %weit = 1+5.*abs(P-T);
            
            %Weighted binary crossed entropy
               
            
            wbce = -(T.*log(sigmoid(Y)));
            den = sum(sum(weit,1),2);
            num = sum(sum(wbce.*weit,1),2);
            wbce = num./den;
            wbce = sum(wbce,3);
            
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiou = 1 - (inter + 1)./(tot-inter+1);   
            wiou = sum(wiou,3);
            
            N = size(Y,4);
            loss = sum(wbce + wiou)/N; 
            loss = loss+DistBOUND;
        end
    end
end

