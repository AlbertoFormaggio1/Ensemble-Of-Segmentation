classdef comb1PixelClassificationLayer < nnet.layer.ClassificationLayer

    % The idea is to sum the two loss function: focal tversky loss and
    % focal dice loss
    
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
        
        function layer = comb1PixelClassificationLayer(name, alpha, beta,gamma)        
            % Set layer name.
            layer.Name = name;
            
            % Set layer properties.
            layer.Alpha = alpha;
            layer.Beta = beta;
            layer.Gamma = gamma;
            
            % Set layer description.
            layer.Description = 'focal dice loss + focal tversky loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T) 
            Ycnot = 1-Y;
            Tcnot = 1-T;
            TP = sum(sum(Y.*T,1),2);
            FP = sum(sum(Y.*Tcnot,1),2);
            FN = sum(sum(Ycnot.*T,1),2); 
            
            numer = TP + layer.Epsilon;
            denom = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;
            
            % Compute Focal Tversky index
            lossFTIc = (1 - numer./denom).^(1./layer.Gamma);
            lossFTI = sum(lossFTIc,3);
            
            %Compute focal tversky loss
            N = size(Y,4);
            focal_tverski= sum(lossFTI)/N;
            
            %--------------------------------------------
            
            %comute focal dice loss
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;
            
            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);
            % Compute Tversky index

            lossTIc = (1 - numer./denom).^(1./layer.Gamma);
            lossTI = sum(lossTIc,3);
            
            % Compute focal dice loss.
            N = size(Y,4);
            focal_dice = sum(lossTI)/N;
            
            % Return the sum of focal tversky and focal dice loss
            loss = focal_dice + focal_tverski;
            
            
        end
        
    end
end

