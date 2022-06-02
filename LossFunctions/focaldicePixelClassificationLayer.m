classdef focaldicePixelClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the focaldice loss function for training semantic
    % segmentation networks.
    
    properties(Constant)

    end
    
    properties
        Gamma= 4/3;
    end
    
    methods
        
        function layer = focaldicePixelClassificationLayer(name, gamma)
            % layer =  focaldicePixelClassificationLayer(name) creates a
            % focaldice pixel classification layer with the specified name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer properties.
            layer.Gamma = gamma;
            
            % Set layer description.
            layer.Description = 'focaldice loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the focaldice loss
            % between the predictions Y and the training targets T.

            TP = sum(sum(Y.*T,1),2);
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;
            
            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);

            % Compute focaldice index
            lossTIc = (1 - numer./denom).^(1./layer.Gamma);
            lossTI = sum(lossTIc,3);
            
            % Return average focaldice index loss.
            N = size(Y,4);
            loss = sum(lossTI)/N;
        end
        
    end
end