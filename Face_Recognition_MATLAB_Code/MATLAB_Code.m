clc; clear; close all;
%% 1. MODEL TRAINING (CALIBRATED)
positive_folder = 'Real_Person';
negative_folder = 'Fake_People';
pos_files = dir(fullfile(positive_folder, '*.jpg'));
neg_files = dir(fullfile(negative_folder, '*.jpg'));
X = []; Y = [];
fprintf('Status: Calibrating and training model...\n');
for i = 1:length(pos_files)
    img = imread(fullfile(positive_folder, pos_files(i).name));
    X = [X; my_stable_features(img)];
    Y = [Y; 1];
end
for i = 1:length(neg_files)
    img = imread(fullfile(negative_folder, neg_files(i).name));
    X = [X; my_stable_features(img)];
    Y = [Y; -1];
end
% Normalization
mu = mean(X); 
sigma = std(X); 
sigma(sigma == 0) = 1; 
X_norm = (X - mu) ./ sigma;
% SVM Training
SVMModel = fitcsvm(X_norm, Y, 'KernelFunction', 'rbf', 'Standardize', true);
% RE-CALIBRATED THRESHOLD
[~, scores] = predict(SVMModel, X_norm);
pos_scores = scores(Y==1, 2);
% Çarpanı 1.7 yaparak güvenli alanı genişlettik
security_threshold = mean(pos_scores) - 1.7 * std(pos_scores);
fprintf('Status: System Calibrated. Threshold: %.4f\n', security_threshold);

%% 2. VERIFICATION TEST (FIRST FRAME SKIP)
try
    cam = webcam;
    % Kamerayı uyandır ve ilk hatalı kareyi çöpe at
    fprintf('Stabilizing camera sensor...\n');
    snapshot(cam); 
    pause(1); 
catch
    delete(webcamlist); cam = webcam;
end

for i = 1:10
    fprintf('\nFrame %d: Adjusting... (3s)', i);
    pause(3); 
    
    img_live = snapshot(cam);
    feat_live = my_stable_features(img_live);
    feat_norm = (feat_live - mu) ./ sigma;
    [label, score] = predict(SVMModel, feat_norm);
    
    % Decision Logic
    if label == 1 && score(2) > security_threshold
        fprintf('\n>>> RESULT: ACCESS GRANTED (Score: %.2f)\n', score(2));
    else
        fprintf('\n>>> RESULT: ACCESS DENIED (Score: %.2f)\n', score(2));
    end
end
clear cam;

%% --- STABLE FEATURE FUNCTION ---
function feat = my_stable_features(img)
    if size(img, 3) == 3, img = rgb2gray(img); end
    img = imresize(img, [64 64]);
    img = im2double(img);
    % Sobel - Structural
    Gx = conv2(img, [1 0 -1; 2 0 -2; 1 0 -1], 'same');
    Gy = conv2(img, [1 2 1; 0 0 0; -1 -2 -1], 'same');
    Gmag = sqrt(Gx.^2 + Gy.^2);
    
    % FFT - Texture
    F = abs(fftshift(fft2(img)));
    F_log = log(1 + F);
    % Sadece en kararlı 3 veriyi gönderiyoruz
    feat = [mean(Gmag(:)), mean(F_log(:)), std(F_log(:))];
end
