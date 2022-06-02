classdef focaltverskyPixelClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the Focal Tversky loss function for training semantic
    % segmentation networks.
    %
    
    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end
    
    properties
        % Default weighting coefficients for False Positives and False
        % Negatives
        Alpha = 0.3;
        Beta = 0.7;
        Gamma= 4/3;
    end
    
    methods
        
        function layer = focaltverskyPixelClassificationLayer(name, alpha, beta, gamma)
            % layer =  focaltverskyPixelClassificationLayer(name) creates a
            % Focal Tversky pixel classification layer with the specified name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer properties.
            layer.Alpha = alpha;
            layer.Beta = beta;
            layer.Gamma = gamma;

            % Set layer description.
            layer.Description = ' Focal Tversky loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Focal Tversky loss
            % between the predictions Y and the training targets T.
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
            
            % Return average Focal Tversky index loss.
            N = size(Y,4);
            loss = sum(lossFTI)/N;
        end
        
    end
end