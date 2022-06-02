classdef BoundExpPixelClassificationLayer < nnet.layer.ClassificationLayer

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
    end

    methods

        function layer = BoundExpPixelClassificationLayer(name,lambdaD,lambdaC,wD,wC)
            % layer =  tverskyPixelClassificationLayer(name) creates a
            % Tversky pixel classification layer with the specified name.

            % Set layer name.
            layer.Name = name;
            
            layer.LambdaDice = lambdaD;
            layer.LambdaCE = lambdaC;
            layer.WeightCE = wC;
            layer.WeightDice = wD;

            % Set layer description.
            layer.Description = 'boundary + dice';
        end


        function loss = forwardLoss(layer, Y, T)
            T=dlarray(T);
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
            %{
            TP = sum(sum(Y.*T,1),2);                        
            numer =2*TP;
            denom = sum(sum(Y+T,1),2);                   
            lossDc = (numer+1)./(denom+1);
            lossDc = (-log(lossDc)).^(layer.LambdaDice);
            lossD = sum(lossDc,3)/size(lossDc,3);            
            %}
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiouc = (inter + 1)./(tot-inter+1);
            wiouc = (-log(wiouc)).^(layer.LambdaDice);
            wiou = sum(wiouc)/size(wiouc,3);
            
            %Log wce           
            %fk = sum(sum(T,1),2);
            %f = sum(fk,3);
            %Calcolo pesi come indicato
            %w = sqrt(f./fk);            
            %Only one between Y*T or Tcnot * Ycnot is active            
            logCE = -(T.*log(sigmoid(Y)) + Tcnot.*log(sigmoid(Ycnot)));            
            %Moltiplico i pesi (nella dimensione 3)            
            wbce = weit.*(logCE).^(layer.LambdaCE);            
            wbce = sum(wbce,3);
            %Pixel totali
            P = size(Y,1)*size(Y,2);
            %Valor medio E
            wbce = sum(sum(wbce,1),2)/P;
            
            
            %Compute final loss 
            N = size(Y,4);
            loss = sum(layer.WeightDice*wiou + layer.WeightCE*wbce,4)/N;
            loss = loss + DistBOUND;
            
            if isnan(loss)
                keyboard
            end
        end

    end
end

