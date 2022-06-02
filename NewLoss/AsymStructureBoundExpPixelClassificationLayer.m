classdef AsymStructureBoundExpPixelClassificationLayer < nnet.layer.ClassificationLayer
    %STRUCTURELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    %Epsilon value is used in order to prevent divisions by 0 which
    %generates nan values.
    properties (Constant)
        Epsilon = 1e-8;
    end
    
    properties
        Alpha = 0.6;
        Beta = 1.6;
        LambdaDice = 0.3;
        LambdaCE = 0.3;
        WeightDice = 0.6;
        WeightCE = 0.4;
    end
    
    methods
        function layer = AsymStructureBoundExpPixelClassificationLayer(name,lambdaD,lambdaC,wD,wC,alpha,beta)
            %Set layer name
            layer.Name = name;
            
            layer.LambdaDice = lambdaD;
            layer.LambdaCE = lambdaC;
            layer.WeightCE = wC;
            layer.WeightDice = wD;
            layer.Alpha = alpha;
            layer.Beta = beta;
            
            %Set layer description
            layer.Description = 'Weighted BCE + Weighted IoU';
        end
        
        function loss = forwardLoss(layer,Y,T)
            N = size(Y,4);
            
            T=dlarray(T);
            weit=dlarray(zeros(size(T)));
            for Nimg=1:size(Y,4)
                P1=((T(:,:,1,Nimg)));
                P2=((Y(:,:,1,Nimg)));
                P1=abs(avgpool(P1,6,'Stride',1,'DataFormat','SST')-0.5);
                P2=abs(avgpool(P2,6,'Stride',1,'DataFormat','SST')-0.5);                
                DistBOUND(Nimg)=1-ssim(P1,P2);
                P = Y(:,:,:,Nimg); %Predictions
                M = T(:,:,:,Nimg); %Mask                
                P = avgpool(P,[31,31],'Padding',15,'DataFormat','SST'); %It could also be Padding = same
                weit(:,:,:,Nimg) = 1+5*abs(P-M);
            end
            DistBOUND=mean(DistBOUND);
            
            %P = avgpool(Y,[31,31],'Padding',15,'DataFormat','SST');
            %weit = 1+5.*abs(P-T);
            
            %Dice
            Ycnot = 1-Y;
            Tcnot = 1-T;            
            TP = sum(sum(Y.*T,1),2);                        
            numer =2*TP;
            denom = sum(sum(Y+T,1),2);                   
            lossDc = (numer+1)./(denom+1);
            lossDc = (-log(lossDc)).^(layer.LambdaDice);
            lossD = sum(lossDc,3)/size(lossDc,3);    
            

            %Log wce           
            fk = sum(sum(T,1),2);
            f = sum(fk,3);
            %Calcolo pesi come indicato
            w = sqrt(f./fk);            
            %Only one between Y*T or Tcnot * Ycnot is active            
            logCE = -(T.*log(sigmoid(Y)) + Tcnot.*log(sigmoid(Ycnot)));            
            %Moltiplico i pesi (nella dimensione 3)            
            wbcec = weit.*(logCE);     
            wbceExpc = wbcec.^(layer.LambdaCE);
            wbceExp = sum(w.*wbceExpc,3);
            %Pixel totali
            P = size(Y,1)*size(Y,2);
            %Valor medio E
            wbceExp = sum(sum(wbceExp,1),2)/P;  
            %wbce            
            lossExp = sum(layer.WeightDice*lossD + layer.WeightCE*wbceExp,4)/N;
            lossBoundExp = lossExp + DistBOUND;
            
            
            
            
            %wiou
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiou = 1 - (inter + 1)./(tot-inter+1);
            wiou = sum(wiou,3);   
            wbcec = sum(sum(wbcec,1),2)./sum(sum(weit,1),2);
            wbce = sum(wbcec,3);

            lossStructure = sum(wbce+wiou)/N;  
            
            
            
            
            %asym
            TP = T.*Y;
            FN = T.*Ycnot;
            FP = Tcnot.*Y;
            num = sum(sum((1+layer.Beta^2)*TP,1),2);
            den = sum(sum((1+layer.Beta^2)*TP + layer.Beta^2*FN + FP,1),2);
            asymc = 1 - num./den;
            asym = sum(asymc,3);            
            
            lossAsym = sum(layer.Alpha*wbce + asym)/N;
            
            loss = lossBoundExp + lossStructure + lossAsym;
            keyboard
        end
    end
end

