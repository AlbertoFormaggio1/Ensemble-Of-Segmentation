classdef BoundComboPixelClassificationLayer < nnet.layer.ClassificationLayer

    % The idea is to sum the three loss function: logCoshDiceloss,
    % focalDiceloss and logCoshFocalTverskiloss


    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end

    properties
        % Default weighting coefficients for False Positives and False
        % Negatives
        Alpha = 0.5;
        Beta = 0.7;
        Gamma=4/3;
    end

    methods

        function layer = BoundComboPixelClassificationLayer(name,alpha,beta)
            % layer =  tverskyPixelClassificationLayer(name) creates a
            % Tversky pixel classification layer with the specified name.

            % Set layer name.
            layer.Name = name;

            layer.Alpha = alpha;
            layer.Beta = beta;

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
            
            %Weighted binary crossed entropy
            Tcnot = 1-T;
            Ycnot = 1-Y;     
            
            %We use Beta=0.7 (>0.5) as explained in Combo loss: Handling input and output imbalance in multi-organ segmentation
            %In order to penalize more false Negatives
            wbce = -(layer.Beta*T.*log(sigmoid(Y)) + (1-layer.Beta)*Tcnot.*log(sigmoid(Ycnot)));
            den = sum(sum(weit,1),2);
            num = sum(sum(wbce.*weit,1),2);
            wbce = num./den;
            wbce = sum(wbce,3);
            
            %We evaluate now Generalized Dice Loss
            TP = sum(sum(Y.*T,1),2);
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;            
            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);

            % Compute dice index.
            %We use epsilon as said in https://arxiv.org/pdf/1805.02798.pdf            
            lossDc = 1 - (numer+layer.Epsilon)./(denom+layer.Epsilon);
            lossD = sum(lossDc,3);
            
            %Moreover, still in https://arxiv.org/pdf/1805.02798.pdf it is
            %said that alpha = 0.5 gives the best performances
            N = size(Y,4);
            loss = sum(layer.Alpha*wbce + (1-layer.Alpha)*lossD)/N;
            loss = loss + DistBOUND;
        end

    end
end

