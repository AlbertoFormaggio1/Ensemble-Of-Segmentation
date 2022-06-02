classdef BoundaryDicePixelClassificationLayer < nnet.layer.ClassificationLayer

    % The idea is to sum the three loss function: logCoshDiceloss,
    % focalDiceloss and logCoshFocalTverskiloss


    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end

    properties
        % Default weighting coefficients for False Positives and False
        % Negatives
        Alpha = 0.3;
        Beta = 0.7;
        Gamma=4/3;
    end

    methods

        function layer = BoundaryDicePixelClassificationLayer(name)
            % layer =  tverskyPixelClassificationLayer(name) creates a
            % Tversky pixel classification layer with the specified name.

            % Set layer name.
            layer.Name = name;

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
            end
            DistBOUND=mean(DistBOUND);


            % loss = forwardLoss(layer, Y, T) returns the dice_new loss
            % between the predictions Y and the training targets T.
            TP = sum(sum(Y.*T,1),2);
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;

            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);
            % Compute dice_new index
            lossTIc = 1 - numer./denom;
            lossTI = sum(lossTIc,3);

            % Return average dice_new index loss.
            N = size(Y,4);
            loss = sum(lossTI)/N;

            % Return the sum of the  loss function
            loss = loss + DistBOUND;

           
            if isnan(loss) || loss==0
                keyboard
            end
        end

    end
end

