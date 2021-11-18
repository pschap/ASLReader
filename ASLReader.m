clear
close all
clc

%% Pre-Processing
letters = char(65:90);
N = 26;
sigma = 1.0;
backgroundIm = imread('BackgroundImages/IMG_0049.JPG');
smoothedBackground = GaussianSmoothing(backgroundIm, sigma);
T = .5;

covMatrices = zeros(3, 3, N);

for i = 1:N
    currentDirectory = strcat('LetterImages/', letters(i));
    imageFiles = dir(fullfile(currentDirectory, '*.jpg'));
    sz = length(imageFiles);
    
    sumMatrix = zeros(3, 3);
    
    for j = 1:sz
        filename = imageFiles(j).name;
        letterIm = imread(strcat(currentDirectory, '/', filename));
        blurredIm = GaussianSmoothing(letterIm, sigma);

        % Perform background subtraction
        region = BackgroundSubtraction(blurredIm, smoothedBackground, T);
        region = bwmorph(region, 'dilate');
        [L, num] = bwlabel(region, 8);
        region = bwareaopen(L, 150, 8);
        
        
        
        %CALCULATION
        %Using features, calculate covariance matrix C model, add result
        %Overlay region to image, convert to grayscale
        handIm = rgb2gray(blurredIm.*region);
        
        %get feature vectors of image
        features = GetWindowFeatureVectors(handIm);
        
        %only pulls coordinate info from region
        %(not sure if this will help?)
        features(:,:,1) = features(:,:,1).*region;
        features(:,:,2) = features(:,:,2).*region;
        
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
         sum = sum/(R*C)

         % to sumMatrix  
         sumMatrix = sumMatrix + sum;
    end
    
    covMatrices(:, :, i) = sumMatrix / sz;
end




%% Matching

letters = char(65:90);
N = 26;
sigma = 1.0;
T = 0.5;

backgroundIm = imread('ASLPics(Ian)/Background/IMG-0405.JPG');
smoothedBackground = GaussianSmoothing(backgroundIm, sigma);

for i = 1:N
    currentDirectory = strcat('ASLPics(Ian)/Letters/', letters(i))
    imageFiles = dir(fullfile(currentDirectory, '*.HEIC'))
    sz = length(imageFiles);
    
    for j = 1:sz
        filename = imageFiles(j).name;
        letterIm = imread(strcat(currentDirectory, '/', filename));
        blurredIm = GaussianSmoothing(letterIm, sigma);

        % Perform background subtraction
        region = BackgroundSubtraction(blurredIm, smoothedBackground, T);
        region = bwmorph(region, 'dilate');
        [L, num] = bwlabel(region, 8);
        region = bwareaopen(L, 150, 8);
        
        
        
        %CALCULATION
        %Using features, calculate covariance matrix C model, add result
        %Overlay region to image, convert to grayscale
        handIm = rgb2gray(blurredIm.*region);
        
        %get feature vectors of image
        features = GetWindowFeatureVectors(handIm);
        
        %only pulls coordinate info from region
        %(not sure if this will help?)
        features(:,:,1) = features(:,:,1).*region;
        features(:,:,2) = features(:,:,2).*region;
        
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
        covar = sum/(R*C)

         
        
        
        %CLOSEST MATCH
        distances = []; 
        for l = 1:N
            %for each letter covariance
            %get eigenvalues
            vals = eig(covar,covMatrix(:,:,l));
            %get rid of zeros so don't ln(0)
            vals(vals==0)=NaN;
            length = size(vals);
            sum = 0;
            %ln() each eigval, square, then sum
            for p = 1:length(1)
                sum = sum + log(vals(p))^2;
            end
            sum = sqrt(sum);
            %save sqrt'd total
            distances(l) = sum;
        end
        
        %find minimum distance
        closest = min(distances);
        %index of that minimum distance is index of patch
        result = find(distances==closest);
        printf('Actual: %s, Classified: %s', letters(i), letters(result));
    end
    
end




%% Reference
% FOLLOWING CODE IS PULLED FROM MY HW6 COVARIACE MATCHING
% modelCovMatrix = [47.917, 0, -146.636, -141.572, -123.269;
% 0, 408.250, 68.487, 69.828, 53.479;
% -146.636, 68.487, 2654.285, 2621.672, 2440.381;
% -141.572, 69.828, 2621.672, 2597.818, 2435.368;
% -123.269, 53.479, 2440.381, 2435.368, 2404.923];
% 
% 
% image = imread('target.jpg');
% figure();
% imagesc(image);
% axis('image');
% 
% window = zeros(70,24);
% [wr, wc] = size(window);
% [R,C,L] = size(image);
% page = zeros(R,C);
% fk = page;
% features = 5;
% 
% for i = 2:features
%     fk(:,:,i) = page;
% end
% 
% %{ 
%     (:,:,1) = x
%     (:,:,2) = y
%     (:,:,3) = R
%     (:,:,4) = G
%     (:,:,5) = B
% %}
% 
% %creating feature vectors
% for r = 1:R
%     for c = 1:C
%         %matlab row-column reverse of x-y
%         fk(r,c,1) = c;
%         fk(r,c,2) = r;
%     end
% end
% 
% fk(:,:,3) = image(:,:,1);
% fk(:,:,4) = image(:,:,2);
% fk(:,:,5) = image(:,:,3);
% 
% patch_cov = zeros(5,5,1);
% patch_num = 1;
% patches = zeros(1,2);
% 
% r = 1;
% while r < (R-wr)
%     c = 1;
%     while c < (C-wc)
%         %Grab Patch
%         patch = fk(r:(r+wr-1),c:(c+wc-1),:);
%         %store starting pixel coordinate
%         %(1,1,1) is x, (1,1,2) is y
%         patches(patch_num,:) = patch(1,1,1:2);
%         %Average features across pixels
%         ufkx = mean(patch, 1);
%         ufk = mean(ufkx,2);
%         sum = 0;
%         
%         %for each pixel in window, subtract mean vector
%         %then multiply by transpose
%         for i = 1:wr
%             for j = 1:wc
%                 sub = patch(i,j,:)-ufk;
%                 sub = sub(:);
%                 sum = sum + (sub)*(sub.');
%             end
%         end
%         %bias sum, divide by #pixels
%         sum = sum/(wr*wc);
%         
%         %save 5x5 covariance matrix
%         patch_cov(:,:,patch_num) = sum;
%         %patch_cov(:,:,patch_num) = cov(sum,1);
%         patch_num = patch_num + 1;
%         %shift window over so only 1 pixel overlaps prior
%         c = c + wc - 1;
%     end
%     %shift window down so only 1 pixel overlaps
%     r = r + wr - 1;
% end
% 
% distances = [];
% patch_num = patch_num - 1;
% 
% %for each patch covariance
% for i = 1:patch_num
%     %get eigenvalues
%     vals = eig(patch_cov(:,:,i),modelCovMatrix);
%     %get rid of zeros so don't ln(0)
%     vals(vals==0)=NaN;
%     length = size(vals);
%     sum = 0;
%     %ln() each eigval, square, then sum
%     for j = 1:length(1)
%         sum = sum + log(vals(j))^2;
%     end
%     sum = sqrt(sum);
%     %save sqrt'd total
%     distances(i) = sum;
% end
% 
% %find minimum distance
% closest = min(distances);
% %index of that minimum distance is index of patch
% result = find(distances==closest);
% %patches array stores starting pixel of that patch
% start_coord = patches(result,:);
% %1=x, 2=y
% c = start_coord(1);
% r = start_coord(2);
% 
% %grab patch from image and display
% section = image(r:r+wr-1, c:c+wc-1, :);
% figure();
% imagesc(section);
% axis('image');
% title('Closest Patch');
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %FOLLOWING CODE IS PULLED FROM HW10 NCC TEMPLATE MATCHING
% left = double(imread('left.png'));
% right = double(imread('right.png'));
% 
% S = 256;
% disparity = zeros([S,S]);
% 
% %I'm assuming the epipolar lines are already paralled
% %they look that way from the images
% %going straight to bottom bullet points on slide 26
% window = 11;
% radius = (window-1)/2;
% 
% %for each pixel in left
% for c = 1+radius:S-radius
%     for r = 1+radius:S-radius
%         xl = left(r,c);
%         lpatch = left(r-radius:r+radius,c-radius:c+radius);
%         left_av = mean(mean(lpatch));
%         left_std = std(lpatch,0,'all');
%         
%         search = min(50,c-(1+radius));
%         best_ncc = 0;
%         best_index = 0;
%         for c2 = 0:search
%             %only need to search to left of rc in right image
%             xr = right(r,c-c2);
%             rpatch = right(r-radius:r+radius,(c-c2)-radius:(c-c2)+radius);
%             right_av = mean(mean(rpatch));
%             right_std = std(rpatch,0,'all');
%             
%             %use ncc to find best pixel in right
%             sum = 0;
%             for x = 1:window
%                 for y = 1:window
%                     sum = sum + ((lpatch(x,y)-left_av)*(rpatch(x,y)-right_av)) / (left_std * right_std);
%                 end
%             end
%             
%             %record largest ncc value
%             ncc = sum/(window*window-1);
%             if ncc > best_ncc
%                 best_ncc = ncc;
%                 best_index = c2;
%             end
%             
%         end
%         
%         match = c-best_index;
%         disparity(r,c) = c-match;
%     end
% end
% 
% figure();
% imagesc(disparity, [0 50]);  
% axis equal; 
% colormap gray; 
% title 'disparity map';


%% Functions

% Performs basic background subtraction using a foreground image, a
% background image, and a threshold (T).
function region = BackgroundSubtraction(foregroundIm, backgroundIm, T)

    % Need to convert images to grayscale
    grayForeground = rgb2gray(foregroundIm);
    grayBackground = rgb2gray(backgroundIm);

    region = abs(grayForeground - grayBackground) > T;

end

% Given a binary region, calculates the average X and Y coordinates
% in the region.
function [X, Y] = CalculateRegionCenter(region)

    X = 0;
    Y = 0;
    count = 0;
    
    sz = size(region);
    for x = 1:sz(2)
        for y = 1:sz(1)
            
            if region(y,x) == 1
                X = X + x;
                Y = Y + y;
                count = count + 1;
            end
            
        end
    end
    
    X = X / XCount;
    Y = Y / YCount;
    
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
