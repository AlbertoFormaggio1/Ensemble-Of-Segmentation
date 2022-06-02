classdef NeigStructureExpPixelClassificationLayer < nnet.layer.ClassificationLayer
       
    properties
        LambdaDice = 1;
        LambdaCE = 1;
        WeightDice = 0.6;
        WeightCE = 0.4;
    end
    
    methods
        function layer = NeigStructureExpPixelClassificationLayer(name,lambdaD,lambdaC,wD,wC)
            layer.Name = name;
            
            layer.LambdaDice = lambdaD;
            layer.LambdaCE = lambdaC;
            layer.WeightCE = wC;
            layer.WeightDice = wD;

            layer.Description = "NeighbordBoundaryDetection + Dice";
        end
        
        function loss = forwardLoss(layer,Y,T)
            T=dlarray(T);
            weit=dlarray(zeros(size(T)));

            Nimg = size(Y,4);
            for i=1:Nimg
                T1 = T(:,:,1,i);
                Y1 = Y(:,:,1,i);
                %Prendo il maxPool
                Gboundary = maxpool(T1,[5,5],'padding','same','DataFormat','SS');
                %Se il pixel è diverso dal max nel pool fatto in
                %precedenza, significa che nel 2-neighborhood è presente un
                %pixel diverso da quello centrale.
                Gboundary = single(Gboundary ~= T1);     
                %Size of column (= number of pixel in a column)
                csz = size(Y,1);
                %Indexes in the 1-neigborhood (have to change that)
                %hoodIdxs = [-rsz-1,-1,rsz-1;-rsz,0,rsz;-rsz+1,1,rsz+1];
                hoodIdxs = [-csz,-1,1,csz];
                Gb1 = extractdata(Gboundary);
                idG = find(Gb1==1);
                Gb5 = zeros(size(Gb1));
                for j=1:length(hoodIdxs)
                    idx = idG + hoodIdxs(j);
                    if(idx >= 1)
                        if(idx <= csz^2)
                            Gb5(idx) = 0.5;
                        end
                    end
                end                
                Gb25 = zeros(size(Gb1));
                hoodIdxs = [-2,-csz-1,csz-1,-2*csz,2*csz,-csz+1,csz+1,2];
                for j=1:length(hoodIdxs)
                    if(idx >= 1)
                        if(idx <= csz^2)
                            Gb25(idx) = 0.25;
                        end
                    end
                end
                Gb5(Gb1~=0) = 0;
                Gb25(Gb1~=0) = 0;
                Gb25(Gb5~=0) = 0;
                Gb1 = Gb1 + Gb5 + Gb25;   
                Gboundary = dlarray(Gb1);                
                
                n = sum(sum(Gboundary,1),2); %Number of ones in the matrix
                %P2 = Y1.*P1 Mantengon solo la prediction del border
                R = dlconv(Y1,Gboundary,0,'DataFormat','SS');
                Bound(i) = R/n;
                %Bound(i) = 1-ssim(P1,P2);

                P = Y(:,:,:,i); %Predictions
                M = T(:,:,:,i); %Mask                
                P = avgpool(P,[31,31],'Padding',15,'DataFormat','SST'); %It could also be Padding = same
                weit(:,:,:,i) = 1+5*abs(P-M);     
            end
            
            lossBound = mean(Bound);
            
            %Dice
            Ycnot = 1-Y;
            Tcnot = 1-T;            
            TP = sum(sum(Y.*T,1),2);                        
            numer =2*TP;
            denom = sum(sum(Y.*Y + T.*T,1),2);                   
            lossDc = (numer+1)./(denom+1);
            lossDc = (-log(lossDc)).^(layer.LambdaDice);
            lossD = sum(lossDc,3)/size(lossDc,3);    
            
            %Log wce (weighted classes)          
            fk = sum(sum(T,1),2);
            f = sum(fk,3);
            %Calcolo pesi come indicato
            w = sqrt(f./fk);            
            %Only one between Y*T or Tcnot * Ycnot is active            
            logCE = -(T.*log(sigmoid(Y))+Tcnot.*log(sigmoid(Ycnot)));
            wbce = (logCE).^(layer.LambdaCE);  
            %Multiplying everything by T because otherwise I would consider
            %the binary cross entropy twice (once for each label). Hence, I
            %would be multiplying the bce for both weights instead of
            %multiplying it for the weight of the corresponding class...
            wbce = sum(T.*w.*wbce,3);            
            %Pixel totali
            P = size(Y,1)*size(Y,2);
            %Valor medio E
            wbce = sum(sum(wbce,1),2)/P;   
            
            %wbce (weighted pixels)
            logCE = -(T.*log(sigmoid(Y)));
            den = sum(sum(weit,1),2);
            num = sum(sum(weit.*logCE,1),2);
            wbcec2 = num./den;
            wbce2 = sum(wbcec2,3);
            
            %wiou
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiou = 1 - (inter + 1)./(tot-inter+1);
            wiou = sum(wiou,3);
            N = size(Y,4);
            lossStructure = sum(wbce2 + wiou)/N;
            
            
            %Compute final loss 
            N = size(Y,4);
            lossExp = sum(layer.WeightDice*lossD + layer.WeightCE*wbce,4)/N;
            loss = lossExp + lossStructure + lossBound;
            %loss = lossStructure + lossBound;

            
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