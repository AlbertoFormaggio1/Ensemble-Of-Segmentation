classdef BoundTopKDicePixelClassificationLayer < nnet.layer.ClassificationLayer

    % The idea is to sum the three loss function: logCoshDiceloss,
    % focalDiceloss and logCoshFocalTverskiloss


    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end

    properties
        % Default weighting coefficients for False Positives and False
        % Negatives        
    end

    methods

        function layer = BoundTopKDicePixelClassificationLayer(name)
            % layer =  tverskyPixelClassificationLayer(name) creates a
            % Tversky pixel classification layer with the specified name.

            % Set layer name.
            layer.Name = name;        

            % Set layer description.
            layer.Description = 'boundary + dice';
        end


        function loss = forwardLoss(layer, Y, T)

            T=dlarray(T); 
            %{
            for Nimg=1:size(Y,4)
                P1=((T(:,:,1,Nimg)));
                P2=((Y(:,:,1,Nimg)));
                P1=abs(avgpool(P1,6,'Stride',1,'DataFormat','SST')-0.5);
                P2=abs(avgpool(P2,6,'Stride',1,'DataFormat','SST')-0.5);
                DistBOUND(Nimg)=1-ssim(P1,P2);
            end
            DistBOUND=mean(DistBOUND);
            %}
            
            %P = avgpool(Y,[31,31],'Padding',15,'DataFormat','SST');
            %weit = 1+5.*abs(P-T);
            
            
            %Pixels with probability below threshold will have 1, otherwise 0
            classSize = 224^2*2;            
            %Where Y==T, I'm setting the value to 1 so it isn't involved in
            %the evaluation of the min value.            
            inter = Y.*T;
            inter(inter==0) = 1;

            Y1 = reshape(inter,classSize,20);
            for i=1:512                
                [val(i,:),idx] = min(Y1,[],1,'linear');   
                Y1(idx) = 1;
            end           
            
            %{
            %Nonfunziona perchè non è in grado di calcolare gradiente da
            solo dopo aver fatto extractdata
            x = extractdata(Y1);    
            x = reshape(x,classSize,20); 
            val = mink(x,2500);
            %}

            %Hard pixel (probability is lower than lambda but they should
            %be considered because T=1)                 
            numer = -sum(log(val+1e-10),1);
            denom = size(val,1);            
            lossK = (numer+layer.Epsilon)/(denom+layer.Epsilon);    
            lossK = mean(lossK);            
            
            %We evaluate now Generalized Dice Loss
            Tcnot = 1-T;
            Ycnot = 1-Y;   
            %Soft Dice
            inter = sum(sum(Y.*T,1),2);
            numer = 2*sum(inter,3);
            denom = sum(sum(Y.*Y+T.*T,1),2);
            denom = sum(denom,3);
            softDc = 1 - numer./denom;
            softD = sum(softDc,3);             
            softD = mean(softD);

            %Compute final loss            
            loss = lossK+softD;
            %loss = loss + DistBOUND;  
        end
    end
end

