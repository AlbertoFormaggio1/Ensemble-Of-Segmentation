classdef comb4PixelClassificationLayer < nnet.layer.ClassificationLayer
    %STRUCTURELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    %Epsilon value is used in order to prevent divisions by 0 which
    %generates nan values.
    properties (Constant)
        Epsilon = 1e-8;
    end
    
    properties
        Alpha=0.3;
        Beta=0.7;
        Gamma=4/3;
    end
    
    methods
        function layer = comb4PixelClassificationLayer(name,alpha,beta,gamma)
            %Set layer name
            layer.Name = name;
            
            layer.Alpha = alpha;
            layer.Beta = beta;
            layer.Gamma = gamma;
            
            %Set layer description
            layer.Description = 'Weighted BCE + Weighted IoU';
        end
        
        function loss = forwardLoss(layer,Y,T)
            T = dlarray(T);
            %iterate through the images inside the batch
            %maybe i have to consider classes too
            
            for Nimg=1:size(Y,4)
                P = Y(:,:,1,Nimg); %Predictions
                M = T(:,:,1,Nimg); %Mask                
                P = avgpool(P,[31,31],'Padding',15,'DataFormat','SST'); %It could also be Padding = same
                weit(:,:,1,Nimg) = 1+5*abs(P-M);
            end
            
            %P = avgpool(Y,[31,31],'Padding',15,'DataFormat','SST');
            %weit = 1+5.*abs(P-T);
            Ycnot = 1-Y;
            Tcnot = 1-T;
            
            %Weighted binary crossed entropy
            TP = sum(sum(Y.*T,1),2);
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;
            
            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);

            % Compute focaldice index
            lossTIc = (1 - numer./denom).^(1./layer.Gamma);
            lossTI = sum(lossTIc,3);
            
            FN = sum(sum(Ycnot.*T.*weit,1),2);
            FP = sum(sum(Y.*Tcnot.*weit,1),2);
            num = TP + layer.Epsilon;
            den = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;
            wiouc = (1 - num ./ den).^(1./layer.Gamma);           
            wiou = sum(wiouc,3);
            
            N = size(Y,4);
            loss = sum(lossTI + wiou)/N;            
        end
    end
end

