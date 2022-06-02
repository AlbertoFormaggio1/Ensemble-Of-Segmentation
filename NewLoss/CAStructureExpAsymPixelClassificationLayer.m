classdef CAStructureExpAsymPixelClassificationLayer < nnet.layer.ClassificationLayer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        K = 5;
        LambdaDice = 0.3;
        LambdaCE = 0.3;
        WeightDice = 0.6;
        WeightCE = 0.4;
        Alpha = 0.6;
        Beta = 1.6;
    end
    
    methods
        function layer = CAStructureExpAsymPixelClassificationLayer(name,k,lambdaD,lambdaC,wD,wC,alpha,beta)
            %Layer name
            layer.Name = name;
            
            layer.K = k;
            layer.LambdaDice = lambdaD;
            layer.LambdaCE = lambdaC;
            layer.WeightCE = wC;
            layer.WeightDice = wD;
            layer.Alpha = alpha;
            layer.Beta = beta;
            
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
            
            bce = - (T.*log(T+1e-8));  

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
            lossCAStructure = sum(CLoss + wiou)/N;
            
            %Dice
            Ycnot = 1-Y;
            Tcnot = 1-T;            
            TP = sum(sum(Y.*T,1),2);                        
            numer =2*TP;
            denom = sum(sum(Y.*Y + T.*T,1),2);                   
            lossDc = (numer+1)./(denom+1);
            lossDc = (-log(lossDc)).^(layer.LambdaDice);
            lossD = sum(lossDc,3)/size(lossDc,3);    
            
            %Log wce (weighted classes)          
            fk = sum(sum(T,1),2);
            f = sum(fk,3);
            %Calcolo pesi come indicato
            w = sqrt(f./fk);            
            %Only one between Y*T or Tcnot * Ycnot is active            
            logCE = -(T.*log(sigmoid(Y))+Tcnot.*log(sigmoid(Ycnot)));
            %Moltiplico i pesi (nella dimensione 3)            
            wbce = (logCE).^(layer.LambdaCE);            
            wbce = sum(w.*wbce,3);
            %Pixel totali
            P = size(Y,1)*size(Y,2);
            %Valor medio E
            wbce = sum(sum(wbce,1),2)/P;               
            N = size(Y,4);
            lossExp = sum(layer.WeightDice*lossD + layer.WeightCE*wbce,4)/N;
            
            %asym
            TP = T.*Y;
            FN = T.*Ycnot;
            FP = Tcnot.*Y;
            num = sum(sum((1+layer.Beta^2)*TP,1),2);
            den = sum(sum((1+layer.Beta^2)*TP + layer.Beta^2*FN + FP,1),2);
            asymc = 1 - num./den;
            asym = sum(asymc,3); 
                        
            den = sum(sum(weit,1),2);
            num = sum(sum(bce.*weit,1),2);
            wbcec = num./den;
            wbce = sum(wbcec,3);
            
            lossAsym = sum(layer.Alpha*wbce + asym)/N;
            
            
             
            loss = lossCAStructure + lossAsym + lossExp;
        end
    end
end

