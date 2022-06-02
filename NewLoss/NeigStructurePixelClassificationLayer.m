classdef NeigStructurePixelClassificationLayer < nnet.layer.ClassificationLayer
       
    properties
        
    end
    
    methods
        function layer = NeigStructurePixelClassificationLayer(name)
            layer.Name = name;
            
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
            
            wbcec = -(T.*log(sigmoid(Y)));
            den = sum(sum(weit,1),2);
            num = sum(sum(wbcec.*weit,1),2);
            wbcec = num./den;            
            wbce = sum(wbcec,3);
            
            inter = sum(sum(Y.*T.*weit,1),2);
            tot = sum(sum((Y+T).*weit,1),2);
            wiou = 1 - (inter + 1)./(tot-inter+1);
            wiou = sum(wiou,3);
            
            N = size(Y,4);
            loss = sum(wbce + wiou)/N;
            loss = loss + lossBound;
            
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