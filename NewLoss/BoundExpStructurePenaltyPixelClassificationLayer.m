classdef BoundExpStructurePenaltyPixelClassificationLayer < nnet.layer.ClassificationLayer

    % The idea is to sum the three loss function: logCoshDiceloss,
    % focalDiceloss and logCoshFocalTverskiloss


    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end

    properties
        %Reference https://arxiv.org/pdf/1809.00076.pdf
        LambdaDice = 0.3;
        LambdaCE = 0.3;
        WeightDice = 0.6;
        WeightCE = 0.4;
        Alpha = 0.6;
        Beta = 1.4;
    end

    methods

        function layer = BoundExpStructurePenaltyPixelClassificationLayer(name,lambdaD,lambdaC,wD,wC,alpha,beta)
            % layer =  tverskyPixelClassificationLayer(name) creates a
            % Tversky pixel classification layer with the specified name.

            % Set layer name.
            layer.Name = name;
            
            layer.LambdaDice = lambdaD;
            layer.LambdaCE = lambdaC;
            layer.WeightCE = wC;
            layer.WeightDice = wD;
            layer.Alpha = alpha;
            layer.Beta = beta;

            % Set layer description.
            layer.Description = 'boundary + dice';
        end


        function loss = forwardLoss(layer, Y, T)
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
            wbce = (logCE).^(layer.LambdaCE);            
            wbce = sum(w.*wbce,3);
            %Pixel totali
            P = size(Y,1)*size(Y,2);
            %Valor medio E
            wbce = sum(sum(wbce,1),2)/P;   
            
            %wbce
            
            den = sum(sum(weit,1),2);
            num = sum(sum(weit.*logCE,1),2);
            wbcec2 = num./den;
            wbce2 = sum(wbcec2,3);
            
            %wiou Pen
            TP = sum(sum(Y.*T.*weit,1),2);
            FN = sum(sum(Ycnot.*T.*weit,1),2);
            FP = sum(sum(Y.*Tcnot.*weit,1),2);
            num = TP + layer.Epsilon;
            den = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;
            wiou = 1 - num ./ den;
            wiou = sum(wiou,3);
            
            
            %Compute final loss 
            N = size(Y,4);
            lossExp = sum(layer.WeightDice*lossD + layer.WeightCE*wbce,4)/N;
            lossBoundExp = lossExp + DistBOUND;            
            lossStructure = sum(wbce2 + wiou)/N;
            
            loss = lossBoundExp + lossStructure;
            if isnan(loss)
                keyboard
            end
            keyboard
        end

    end
end

