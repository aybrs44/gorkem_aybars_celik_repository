clc; clear; close all;

%% 1. MODEL TRAINING (FAST AND STABLE)
posFolder = 'Real_Person';
negFolder = 'Fake_People';

posFiles = dir(fullfile(posFolder, '*.jpg'));
negFiles = dir(fullfile(negFolder, '*.jpg'));

X = []; Y = [];
fprintf('Training model, please wait...\n');

for i = 1:length(posFiles)
    img = imread(fullfile(posFolder, posFiles(i).name));
    X = [X; my_simple_features(img)];
    Y = [Y; 1];
end

for i = 1:length(negFiles)
    img = imread(fullfile(negFolder, negFiles(i).name));
    X = [X; my_simple_features(img)];
    Y = [Y; -1];
end

% Normalization and SVM (RBF Kernel)
mu = mean(X); sigma = std(X); sigma(sigma == 0) = 1;
X_norm = (X - mu) ./ sigma;
SVMModel = fitcsvm(X_norm, Y, 'KernelFunction', 'rbf', 'Standardize', true);

% Threshold Setting (Safe zone for recognition)
[~, scores] = predict(SVMModel, X_norm);
posScores = scores(Y==1, 2);
safe_threshold = mean(posScores) - 1.8 * std(posScores); 

fprintf('System Ready! Starting 10-frame analysis...\n');

%% 2. 10-FRAME STEP-BY-STEP TEST (NO LAG)
try
    cam = webcam;
catch
    delete(webcamlist); cam = webcam;
end

for i = 1:10
    fprintf('\nPrepare for Frame %d... (3 seconds)', i);
    pause(3); % Time to adjust pose
    
    % Capture snapshot
    img_live = snapshot(cam);
    
    % Analysis
    feat_live = my_simple_features(img_live);
    feat_norm = (feat_live - mu) ./ sigma;
    [label, score] = predict(SVMModel, feat_norm);
    
    % Decision Mechanism
    if label == 1 && score(2) > safe_threshold
        fprintf('\n>>> RESULT: ACCESS GRANTED (Score: %.2f)\n', score(2));
    else
        fprintf('\n>>> RESULT: ACCESS DENIED (Score: %.2f)\n', score(2));
    end
end

clear cam;
fprintf('\n10-frame test completed.\n');

%% --- LIGHTWEIGHT FEATURE FUNCTION ---
function feat = my_simple_features(img)
    if size(img, 3) == 3, img = rgb2gray(img); end
    img = imresize(img, [64 64]);
    img = im2double(img);

    % Sobel - Edge
    Gx = conv2(img, [1 0 -1; 2 0 -2; 1 0 -1], 'same');
    Gy = conv2(img, [1 2 1; 0 0 0; -1 -2 -1], 'same');
    Gmag = sqrt(Gx.^2 + Gy.^2);
    
    % FFT - Texture
    F = abs(fftshift(fft2(img)));
    F_log = log(1 + F);

    feat = [mean(Gmag(:)), mean(F_log(:)), std(F_log(:))];
end
