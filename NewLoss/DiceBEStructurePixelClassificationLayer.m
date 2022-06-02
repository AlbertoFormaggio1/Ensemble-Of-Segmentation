classdef DiceBEStructurePixelClassificationLayer < nnet.layer.ClassificationLayer
    %STRUCTURELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    %Epsilon value is used in order to prevent divisions by 0 which
    %generates nan values.
    properties (Constant)
        Epsilon = 1e-8;
    end

    properties
        Lambda1 = 1;
        Lambda2 = 0.01;
    end
    
    methods
        function layer = DiceBEStructurePixelClassificationLayer(name,lambda1,lambda2)
            %Set layer name
            layer.Name = name;
           
            layer.Lambda1 = lambda1;
            layer.Lambda2 = lambda2;

            %Set layer description
            layer.Description = 'Dice + BE + Structure';
        end
        
        function loss = forwardLoss(layer,Y,T)
            T = dlarray(T);            
            %iterate through the images inside the batch
            %maybe i have to consider classes too
            
            T = dlarray(T);
            %iterate through the images inside the batch
            %maybe i have to consider classes too
            
            weit = dlarray(zeros(size(T)));
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
            
            wbcec = -(T.*log(sigmoid(Y)) + Tcnot.*log(sigmoid(Ycnot)));
            den = sum(sum(weit,1),2);
            num = sum(sum(wbcec.*weit,1),2);
            wbcec = num./den;            
            wbce = sum(wbcec,3);
            
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiou = 1 - (inter + 1)./(tot-inter+1);
            wiou = sum(wiou,3);
            
            N = size(Y,4);
            lossStructure = sum(wbce + wiou)/N;
            
            %P = avgpool(Y,[31,31],'Padding',15,'DataFormat','SST');
            %weit = 1+5.*abs(P-T);
            
            %Soft dice loss
            TP = sum(sum(Y.*T,1),2);            
            
            numer =2*sum(TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(denom,3);
            % Compute dice_new index
            softD = 1 - numer./denom;            
            
            %Boundary enhancement (BE)
            %3x3 kernel. Single channel. 3 Layers
            bias = 0;
            %{
            kernel(1:3,1:3,1,1:3) = 1/9;
            kernel(1:3,1:3,1,4) = -1;
            kernel(2,2,1,4) = 8;
            kernel = dlarray(kernel);
            Y1 = dlconv(Y,kernel,bias,'Padding','same','DataFormat','SSC');
            %}
            
            Y1 = Y;
            %Threefold smoothing kernel (Laplacian tends to sharpen the
            %image so we blurry it a bit before applying the laplacian mask)
            kernel(1:3,1:3) = 1/9;            
            for i=1:3
                Y1 = dlconv(Y1,kernel,bias,'Padding','same','DataFormat','SSTU');
            end

            %Laplacian filter
            laplacian_mask = [0 -1 0,-1 4 -1,0 -1 0];            
            Y1 = dlconv(Y1,laplacian_mask,bias,'Padding','same','DataFormat','SSTU');
            T1 = dlconv(T,laplacian_mask,bias,'Padding','same','DataFormat','SSTU');  
            
            %L2 norm            
            l2Norm = sqrt(sum(sum((T1-Y1).^2,1),2));            
            lossBE = sum(l2Norm,3);
            lossBE = mean(lossBE);
            %lossBE = l2loss(Y1,T,'DataFormat','SSTU');      

            N = size(Y,4);
            %loss = sum(layer.Lambda1*softD + layer.Lambda2*lossBE + wiou)/N;    
            lossD = sum(layer.Lambda1*softD,4)/N;                        
            loss = lossD + layer.Lambda2*lossBE + lossStructure;            
        end
    end
end

