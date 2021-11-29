clear
close all
clc

%% Pre-Processing
%Initializations
letters = char(65:90);
N = 26;
sigma = 1.0;
backgroundIm = imread('BackgroundImages(New)/IMG_0257.JPG');
smoothedBackground = GaussianSmoothing(backgroundIm, sigma);
T = .2;

covMatrices = zeros(3, 3, N);

%Only train if no training file saved
if ~isfile('covMatrices.txt')
    
    %for ecah letter
    for i = 1:N
        %find directory for that letter
        currentDirectory = strcat('LetterImages(New)/', letters(i));
        imageFiles = dir(fullfile(currentDirectory, '*.jpg'));
        sz = length(imageFiles);

        sumMatrix = zeros(3, 3);

        %For each image of that letter
        for j = 1:sz
            %read image and smooth
            filename = imageFiles(j).name;
            letterIm = imread(strcat(currentDirectory, '/', filename));
            blurredIm = GaussianSmoothing(letterIm, sigma);

            % Perform background subtraction
            region = BackgroundSubtraction(blurredIm, smoothedBackground, T);
            region = bwmorph(region, 'dilate');
            [L, num] = bwlabel(region, 8);
            region = bwareaopen(L, 500, 8);

            % OLD REGION FINDING
            % Extract window using background subtracted image
            %{
            [rows, columns] = find(region);
            windowLeft = min(columns);
            windowRight = max(columns);
            windowBottom = max(rows);
            windowTop = min(rows);
            

            rowThreshold = 25;
            columnThreshold = 25;
            imageSz = size(blurredIm);
            

            left = max(1, windowLeft-columnThreshold);
            right = min(imageSz(2), windowRight+columnThreshold);
            top = max(1, windowTop-rowThreshold);
            bottom = min(imageSz(1), windowBottom+rowThreshold);

            handRegion = blurredIm(top:bottom, left:right, :);
            %}
            
            %CALCULATION
            %Using features, calculate covariance matrix C model, add result
            %Overlay region to image, convert to grayscale
            handIm = rgb2gray(blurredIm.*region);
            %figure();
            %imshow(handIm);
            
            %get feature vectors of image
            features = GetWindowFeatureVectors(handIm);
            
            %Mean feature vector
            ufkx = mean(features, 1);
            ufk = mean(ufkx,2);
            
            %Calculate covariance
            [R,C] = size(features(:,:,3));
            sum = 0;
            for y = 1:R
                 for x = 1:C
                     sub = features(y,x,:)-ufk;
                     sub = sub(:);
                     sum = sum + (sub)*(sub.');
                 end
             end
             %bias sum, divide by #pixels
             sum = sum/(R*C);

             % add to sumMatrix  
             sumMatrix = sumMatrix + sum;
        end
        %average sumMatrix by #images
        covMatrices(:, :, i) = sumMatrix / sz;
    end
    %Save to file
    writematrix(covMatrices, 'covMatrices.txt');
else
    covMatrices = readmatrix('covMatrices.txt');
end




%% Matching
%Initialization
covMatrices = readmatrix('covMatrices.txt');
covMatrices(isnan(covMatrices))=0;
letters = char(65:90);
N = 26;
sigma = 1.0;
T = 0.2;

backgroundIm = imread('BackgroundImages(New)/IMG_0257.JPG');
smoothedBackground = GaussianSmoothing(backgroundIm, sigma);

total = 0;
correct = 0;
%For each letter
for i = 1:N
    currentDirectory = strcat('TestImages/', letters(i));
    imageFiles = dir(fullfile(currentDirectory, '*.jpg'));
    sz = length(imageFiles);
    
    %For each test image
    %Same Process as during Training
    for j = 1:sz
        filename = imageFiles(j).name;
        concat = strcat(currentDirectory, '/', filename);
        letterIm = imread(concat,'jpg');
        blurredIm = GaussianSmoothing(letterIm, sigma);
        
        % Perform background subtraction
        region = BackgroundSubtraction(blurredIm, smoothedBackground, T);
        region = bwmorph(region, 'dilate');
        [L, num] = bwlabel(region, 8);
        region = bwareaopen(L, 150, 8);
        %figure();
        %imshow(region);
        
        % Extract window using background subtracted image
        %{
        [rows, columns] = find(region);
        windowLeft = min(columns);
        windowRight = max(columns);
        windowBottom = max(rows);
        windowTop = min(rows);
        
        rowThreshold = 25;
        columnThreshold = 25;
        sz = size(blurredIm);
        
        left = max(1, windowLeft-columnThreshold);
        right = min(sz(2), windowRight+columnThreshold);
        top = max(1, windowTop-rowThreshold);
        bottom = min(sz(1), windowBottom+rowThreshold);
        
        handRegion = blurredIm(top:bottom, left:right, :);
        %}
        %CALCULATION
        %Using features, calculate covariance matrix C model, add result
        %Overlay region to image, convert to grayscale
        
        %uses side cropping to region
        %handIm = rgb2gray(handRegion);
        
        %only looks at pixels flagged by region
        handIm = rgb2gray(blurredIm.*region);
        %figure();
        %imshow(handIm);
        
        %get feature vectors of image
        features = GetWindowFeatureVectors(handIm);
        
        %Mean feature vector
        ufkx = mean(features, 1);
        ufk = mean(ufkx,2);
        
        [R,C] = size(features(:,:,3));
        sum = 0;
        for y = 1:R
             for x = 1:C
                 sub = features(y,x,:)-ufk;
                 sub = sub(:);
                 sum = sum + (sub)*(sub.');
             end
        end
        %bias sum, divide by #pixels
        covar = sum/(R*C);

        %CLOSEST MATCH
        distances = []; 
        for l = 1:N
            %for each letter covariance
            %get eigenvalues
            low = l*3-2;
            high = l*3;
            covMatrix = covMatrices(:,low:high);
            vals = eig(covar,covMatrix);
            %get rid of zeros so don't ln(0)
            vals(vals==0)=NaN;
            len = size(vals);
            sum = 0;
            %ln() each eigval, square, then sum
            for p = 1:len(1)
                sum = sum + log(vals(p))^2;
            end
            sum = sqrt(sum);
            %save sqrt'd total
            distances(l) = sum;
        end
        %find minimum distance
        closest = min(distances);
        %index of that minimum distance is letter classification
        result = find(distances==closest);
        fprintf('Actual: %c, Classified: %c\n', letters(i), letters(result));
        if letters(i) == letters(result)
            correct = correct + 1;
        end
        
        total = total + 1;
    end
    
end

fprintf('Classification accuracy: %u/%u\n', correct, total);

%% Functions

% Performs basic background subtraction using a foreground image, a
% background image, and a threshold (T).
function region = BackgroundSubtraction(foregroundIm, backgroundIm, T)

    % Need to convert images to grayscale
    grayForeground = rgb2gray(foregroundIm);
    grayBackground = rgb2gray(backgroundIm);

    region = abs(grayForeground - grayBackground) > T;

end

% PULLED FROM HW2 CODE
% Performs Gaussian Smoothing.
function smoothedIm = GaussianSmoothing(im, sigma)
    
    % Get smoothing mask
    G = fspecial('gaussian', 2*ceil(3*sigma)+1, sigma);
    
    % Convert image to double type
    im = double(im);
    
    % Apply smoothing mask
    smoothedIm = imfilter(im, G, 'replicate');
    smoothedIm = smoothedIm / 255;
end

% Extracts feature vectors from each pixel within the specified window.
% x - x coordinate of the window center
% y - y coordinate of the window center
function features = GetWindowFeatureVectors(im)
    [R, C] = size(im);
    features = zeros(R, C, 3);
    
    %generate column numbers
    col = 1:C;
    repeat = repelem(col,R);
    repeat = reshape(repeat,[R,C]);
    features(:,:,1) = repeat;
    
    %generate row numbers
    rows = 1:R;
    repeat = repelem(rows,C);
    repeat = transpose(reshape(repeat,[C,R]));
    features(:,:,2) = repeat;
    
    %Generate Greyscale Values
    features(:,:,3) = im;
    
    %I think the below code was slow
    %{
    for x = 1:C
        for y = 1:R
            features(x, y, 1) = x;
            features(x, y, 2) = y;
            features(:, :, 3) = im(x,y);
        end    
    end
    %}
    
end
