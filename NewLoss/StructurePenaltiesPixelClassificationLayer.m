classdef StructurePenaltiesPixelClassificationLayer < nnet.layer.ClassificationLayer
    %STRUCTURELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    %Epsilon value is used in order to prevent divisions by 0 which
    %generates nan values.
    properties (Constant)
        Epsilon = 1e-8;
    end
    
    properties
        Alpha=0.6;
        Beta=1.4;
    end
    
    methods
        function layer = StructurePenaltiesPixelClassificationLayer(name,alpha,beta)
            %Set layer name
            layer.Name = name;
            
            layer.Alpha = alpha;
            layer.Beta = beta;
            
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
            wbce = -(T.*log(sigmoid(Y)) + Tcnot.*log(sigmoid(Ycnot)));
            den = sum(sum(weit,1),2);
            num = sum(sum(wbce.*weit,1),2);
            wbce = num./den;
            
            TP = sum(sum(Y.*T.*weit,1),2);
            FN = sum(sum(Ycnot.*T.*weit,1),2);
            FP = sum(sum(Y.*Tcnot.*weit,1),2);
            num = TP + layer.Epsilon;
            den = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;
            wiou = 1 - num ./ den;
            
            wbce = sum(wbce,3);
            wiou = sum(wiou,3);
            
            N = size(Y,4);
            loss = sum(wbce + wiou)/N;            
        end
    end
end

