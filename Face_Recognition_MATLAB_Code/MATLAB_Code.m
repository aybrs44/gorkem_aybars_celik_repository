clc; clear; close all;

%% 1. MODEL TRAINING (STABLE & ROBUST)
positive_folder = 'Real_Person';
negative_folder = 'Fake_People';

pos_files = dir(fullfile(positive_folder, '*.jpg'));
neg_files = dir(fullfile(negative_folder, '*.jpg'));

X = []; Y = [];
fprintf('Status: Training the SVM model. Please wait...\n');

% Process Positive Samples
for i = 1:length(pos_files)
    img = imread(fullfile(positive_folder, pos_files(i).name));
    X = [X; get_verification_features(img)];
    Y = [Y; 1];
end

% Process Negative Samples
for i = 1:length(neg_files)
    img = imread(fullfile(negative_folder, neg_files(i).name));
    X = [X; get_verification_features(img)];
    Y = [Y; -1];
end

% Normalization
mu = mean(X); 
sigma = std(X); 
sigma(sigma == 0) = 1; % Avoid division by zero
X_norm = (X - mu) ./ sigma;

% Train SVM with RBF Kernel for non-linear decision boundaries
SVMModel = fitcsvm(X_norm, Y, 'KernelFunction', 'rbf', 'Standardize', true);

% Calculate Security Threshold
[~, scores] = predict(SVMModel, X_norm);
pos_scores = scores(Y==1, 2);
security_threshold = mean(pos_scores) - 1.8 * std(pos_scores); 

fprintf('Status: System Ready! Security Threshold: %.4f\n', security_threshold);

%% 2. STEP-BY-STEP LIVE VERIFICATION (10 CYCLES)
try
    cam = webcam;
catch
    delete(webcamlist); 
    cam = webcam;
end

fprintf('\n--- Starting 10-Frame Verification Process ---\n');

for i = 1:10
    fprintf('\nFrame [%d/10]: Get ready... (3 seconds)', i);
    pause(3); % Time to adjust your pose or test with your hand
    
    % Capture Snapshot
    live_frame = snapshot(cam);
    
    % Extract and Normalize Features
    live_feat = get_verification_features(live_frame);
    live_feat_norm = (live_feat - mu) ./ sigma;
    
    % Prediction
    [label, score] = predict(SVMModel, live_feat_norm);
    current_score = score(2);
    
    % Decision Logic
    if label == 1 && current_score > security_threshold
        fprintf('\n>>> RESULT: ACCESS GRANTED (Score: %.2f)\n', current_score);
        fprintf('User: GORKEM AYBARS CELIK\n');
    else
        fprintf('\n>>> RESULT: ACCESS DENIED (Score: %.2f)\n', current_score);
        fprintf('Identity: UNKNOWN / BLOCKED\n');
    end
end

clear cam;
fprintf('\n--- Verification process completed. ---\n');

%% --- FEATURE EXTRACTION FUNCTION ---
function feat = get_verification_features(img)
    % Pre-processing
    if size(img, 3) == 3, img = rgb2gray(img); end
    img = imresize(img, [64 64]);
    img = im2double(img);

    % Sobel Edge Analysis (Structural features)
    Gx = conv2(img, [1 0 -1; 2 0 -2; 1 0 -1], 'same');
    Gy = conv2(img, [1 2 1; 0 0 0; -1 -2 -1], 'same');
    gradient_magnitude = sqrt(Gx.^2 + Gy.^2);
    
    % FFT Magnitude Analysis (Texture/Frequency features)
    F = abs(fftshift(fft2(img)));
    F_log = log(1 + F);

    % Return stable feature vector
    feat = [mean(gradient_magnitude(:)), mean(F_log(:)), std(F_log(:))];
end
