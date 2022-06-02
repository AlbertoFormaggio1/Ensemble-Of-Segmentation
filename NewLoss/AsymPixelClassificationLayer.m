classdef AsymPixelClassificationLayer < nnet.layer.ClassificationLayer
    %STRUCTURELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    %Epsilon value is used in order to prevent divisions by 0 which
    %generates nan values.
    properties (Constant)
        Epsilon = 1e-8;
    end
    
    properties
        %Asym is based on the idea of https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9140793
        Alpha = 0.6;
        Beta = 1.6;
    end
    
    methods
        function layer = AsymPixelClassificationLayer(name,alpha,beta)
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
            wbcec = -(T.*log(sigmoid(Y)));
            den = sum(sum(weit,1),2);
            num = sum(sum(wbcec.*weit,1),2);
            wbcec = num./den;
            wbce = sum(wbcec,3);
            
            %asym
            TP = T.*Y;
            FN = T.*Ycnot;
            FP = Tcnot.*Y;
            num = (1+layer.Beta^2)*sum(sum(TP,1),2);
            den = sum(sum(num + layer.Beta^2*FN + FP,1),2);
            asymc = 1 - num./den;
            asym = sum(asymc,3);
            
            N = size(Y,4);
            loss = sum(layer.Alpha*wbce + asym)/N;
            if isnan(loss)
                keyboard
            end
        end
    end
end

