classdef MultiSSimFocaldicePixelClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the focaldice loss function for training semantic
    % segmentation networks.
    
    properties(Constant)

    end
    
    properties
        Gamma= 4/3;
    end
    
    methods
        
        function layer = MultiSSimFocaldicePixelClassificationLayer(name, gamma)
            % layer =  focaldicePixelClassificationLayer(name) creates a
            % focaldice pixel classification layer with the specified name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer properties.
            layer.Gamma = gamma;
            
            % Set layer description.
            layer.Description = 'MultiSSim + focaldice loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the focaldice loss
            % between the predictions Y and the training targets T.

            T=dlarray(T);
            for Nimg=1:size(Y,4)
                P1=(T(:,:,1,Nimg));
                P2=(Y(:,:,1,Nimg));
                DistMS(Nimg)=1-multissim(dlarray(P1),P2);
            end
            DistMS=mean(DistMS);

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
            
            loss=loss+DistMS;
        end
        
    end
end