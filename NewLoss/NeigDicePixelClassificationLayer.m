classdef NeigDicePixelClassificationLayer < nnet.layer.ClassificationLayer
       
    properties
        
    end
    
    methods
        function layer = NeigDicePixelClassificationLayer(name)
            layer.Name = name;
            
            layer.Description = "NeighbordBoundaryDetection + Dice";
        end
        
        function loss = forwardLoss(layer,Y,T)
            T=dlarray(T);
            Nimg = size(Y,4);
            for i=1:Nimg
                T1 = T(:,:,1,i);
                Y1 = Y(:,:,1,i);
                %Prendo il maxPool
                P1 = maxpool(T1,[5,5],'padding','same','DataFormat','SS');
                %Se il pixel è diverso dal max nel pool fatto in
                %precedenza, significa che nel 2-neighborhood è presente un
                %pixel diverso da quello centrale.
                P1 = single(P1 ~= T1);
                P2 = maxpool(Y1,[5,5],'padding','same','DataFormat','SS');
                P2 = P2.*P1;
                Bound(i) = 1-ssim(P1,P2);
            end
            
            lossBound = mean(Bound);
            
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
            loss = loss + 10*lossBound;        
            
            %{
            %To see that the boundaries reveal is almost correct:
            
            b = extractdata(Bound);
            figure
            imshow(b(:,:,1));
            figure
            T1 = extractdata(T);
            imshow(T1(:,:,1));
            %}
            
        end
    end
end