classdef logcoshdicePixelClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the logcoshdice loss function for training semantic
    % segmentation networks.

    properties(Constant)
    end
    
    properties
    end
    
    methods
        
        function layer = logcoshdicePixelClassificationLayer(name)
            % layer =  logcoshdicePixelClassificationLayer(name) creates a
            % logcoshdice pixel classification layer with the specified name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'logcoshdice loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the logcoshdice loss
            % between the predictions Y and the training targets T.
            Ycnot = 1-Y;
            Tcnot = 1-T;
            TP = sum(sum(Y.*T,1),2);
            a=(sum(sum(T.*T,1),2)).^2;
            w=1./a;
            
            numer =2*sum(w.*TP,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom=sum(w.*denom,3);
            % Compute logcoshdice index
            lossTIc = log(cosh(1 - numer./denom));
            lossTI = sum(lossTIc,3);
            
            % Return average logcoshdice index loss.
            N = size(Y,4);
            loss = sum(lossTI)/N;
        end
        
    end
end