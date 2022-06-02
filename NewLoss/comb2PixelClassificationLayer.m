classdef comb2PixelClassificationLayer < nnet.layer.ClassificationLayer

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
        Beta = 0.5;
        Gamma=4/3;
    end
    
    methods
        
        function layer = comb2PixelClassificationLayer(name, alpha, beta,gamma)
            % layer =  tverskyPixelClassificationLayer(name) creates a
            % Tversky pixel classification layer with the specified name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer properties.
            layer.Alpha = alpha;
            layer.Beta = beta;
            layer.Gamma = gamma;
            
            % Set layer description.
            layer.Description = 'logCoshDice + focalDice + logCoshFocal Tverski';
        end
        
        
        function loss = forwardLoss(layer, Y, T) 
           
            %LogCosh Dice Loss
            TP = sum(sum(Y.*T,1),2);
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;
            
            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);
            % Compute Tversky index
            lossTIc = log(cosh(1 - numer./denom));
%             lossTIc(find(extractdata(isnan(gather(lossTIc)))))=0;

            lossTI = sum(lossTIc,3);
            
            % Compute logCosh dice loss
            N = size(Y,4);
            logCoshDiceloss = sum(lossTI)/N;
            
            %---------------------------
            
            % Focal Dice loss
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;
            
            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);
            % Compute Tversky index
            lossTIc = (1 - numer./denom).^(1./layer.Gamma);
%             lossTIc(find(extractdata(isnan(gather(lossTIc)))))=0;
            lossTI = sum(lossTIc,3);
            
            % Compute focal dice loss.
            N = size(Y,4);
            focalDiceloss = sum(lossTI)/N;
            
            %---------------------------
            
            % logCosh focal tversky loss
            Ycnot = 1-Y;
            Tcnot = 1-T;
            TP = sum(sum(Y.*T,1),2);
            FP = sum(sum(Y.*Tcnot,1),2);
            FN = sum(sum(Ycnot.*T,1),2); 
            
            numer = TP + layer.Epsilon;
            denom = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;
            
            % Compute Tversky index
            lossTIc = log(cosh((1 - numer./denom).^(1./layer.Gamma)));
%             lossTIc(find(extractdata(isnan(gather(lossTIc)))))=0;
            lossTI = sum(lossTIc,3);
            
            % Compute logCosh focal tversky loss.
            N = size(Y,4);
            logCoshFocalTverskiloss = sum(lossTI)/N;
            
            % Return the sum of the three loss function
            loss = logCoshDiceloss + focalDiceloss + logCoshFocalTverskiloss;

            if isnan(loss)
                keyboard
            end
        end
        
    end
end

