classdef SensSpecStructurePixelClassificationLayer < nnet.layer.ClassificationLayer
    %STRUCTURELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    %Epsilon value is used in order to prevent divisions by 0 which
    %generates nan values.
    properties
        Epsilon = 1e-8;
        R = 0.1;
    end
    
    methods
        function layer = SensSpecStructurePixelClassificationLayer(name,r)
            %Set layer name
            layer.Name = name;
            layer.R = r;
            
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
            end
            
            %P = avgpool(Y,[31,31],'Padding',15,'DataFormat','SST');
            %weit = 1+5.*abs(P-T);
            
            %Weighted binary crossed entropy
            Tcnot = 1-T;
            Ycnot = 1-Y;     
            
            wbcec = -(T.*log(Y+1e-8) + Tcnot.*log(Ycnot+1e-8));
            den = sum(sum(weit,1),2);
            num = sum(sum(wbcec.*weit,1),2);
            wbcec = num./den;            
            wbce = sum(wbcec,3);
            
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiou = 1 - (inter + 1)./(tot-inter+1);
            wiou = sum(wiou,3);
            
            N = size(Y,4);
            lossStr = sum(wbce + wiou)/N;  
            
            %SensSpec
            Ycnot = 1-Y;
            Tcnot = 1-T;
            TP = sum(sum(Y.*T,1),2);
            FP = sum(sum(Y.*Tcnot,1),2);
            FN = sum(sum(Ycnot.*T,1),2); 
            TN = sum(sum(Ycnot.*Tcnot,1),2);
            
            lossSens = layer.R*TP./(TP + FN) + (1 - layer.R)*TN./(TN+FP);
            lossSens = lossSens(1,1,1,:);
            lossSens = sum(lossSens)/N;

            loss = lossSens + 0.5*lossStr; 
        end
    end
end

