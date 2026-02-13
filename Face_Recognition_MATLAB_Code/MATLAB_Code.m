%--------------------------------------
% ana_script.m

positive_folder = 'Ben_2025-07-07_10-38-07';
negative_folder = 'Negatives'; % Tam yolunu yaz

positive_files = dir(fullfile(positive_folder, '*.jpg'));
negative_files = dir(fullfile(negative_folder, '*.jpeg'));

X = [];
Y = [];

% Pozitif örnekler
for i = 1:length(positive_files)
    img = imread(fullfile(positive_folder, positive_files(i).name));
    img_gray = rgb2gray(img);
    img_resized = imresize(img_gray, [64 64]);
    img_double = im2double(img_resized);

    feat = extract_features_hog_sobel(img_double);
    feat = reshape(feat, 1, []);
    X = [X; feat];
    Y = [Y; 1];
end

% Negatif örnekler
for i = 1:length(negative_files)
    img = imread(fullfile(negative_folder, negative_files(i).name));
    img_gray = rgb2gray(img);
    img_resized = imresize(img_gray, [64 64]);
    img_double = im2double(img_resized);

    feat = extract_features_hog_sobel(img_double);
    feat = reshape(feat, 1, []);

    fprintf('Negatif örnek %d min: %.5f, max: %.5f, NaN sayısı: %d\n', i, min(feat), max(feat), sum(isnan(feat)));

    X = [X; feat];
    Y = [Y; -1];
end

% Negatif örneklerde benzersiz satır sayısı
unique_neg = unique(X(Y==-1, :), 'rows');
fprintf('Negatif örneklerde benzersiz örnek sayısı: %d\n', size(unique_neg,1));

% Ortalama ve standart sapma
mu = mean(X, 1);
sigma = std(X, [], 1);

% NaN ve 0 varyans düzeltmesi
sigma(isnan(sigma) | sigma == 0) = 1;

% Normalize et
X_norm = (X - mu) ./ sigma;

% SVM parametreleri
C = 1;
lr = 0.01;
epochs = 10000;

[w, b] = train_svm(X_norm, Y, C, lr, epochs);

% Eğitim skorları
scores_train = X_norm * w' + b;
disp('Pozitif örnek skorları:');
disp(scores_train(Y==1));
disp('Negatif örnek skorları:');
disp(scores_train(Y==-1));

mean_pos = mean(scores_train(Y==1));
mean_neg = mean(scores_train(Y==-1));
threshold = (mean_pos + mean_neg) / 2;

fprintf('Pozitif skor ortalaması: %.4f\n', mean_pos);
fprintf('Negatif skor ortalaması: %.4f\n', mean_neg);
fprintf('Belirlenen threshold: %.4f\n', threshold);

% Canlı görüntü al ve özellik çıkar
cam = webcam;
img_live = snapshot(cam);
img_live_gray = rgb2gray(img_live);
img_live_resized = imresize(img_live_gray, [64 64]);
img_live_double = im2double(img_live_resized);

feat_live = extract_features_hog_sobel(img_live_double);
feat_live = reshape(feat_live, 1, []);

% NaN'ları sıfırla (güvenlik)
feat_live(isnan(feat_live)) = 0;

feat_live_norm = (feat_live - mu) ./ sigma;

score = dot(w, feat_live_norm) + b;
fprintf('Canlı görüntü skoru: %.4f\n', score);
fprintf('Threshold: %.4f\n', threshold);

if score >= threshold
    disp('Bu sensin!');
else
    disp('Sen değilsin.');
end

clear cam;
disp(length(negative_files)) 
%--------------------------------------------------------

function y_pred = predict_svm(X, w, b)
    y_pred = sign(X * w' + b);
end

%-------------------------------------------------------
function [w, b] = train_svm(X, Y, C, lr, epochs)
    [N, D] = size(X);
    w = zeros(1, D);
    b = 0;

    for epoch = 1:epochs
        for i = 1:N
            xi = X(i, :);
            yi = Y(i);
            margin = yi * (w * xi' + b);
            if margin >= 1
                grad_w = w;
                grad_b = 0;
            else
                grad_w = w - C * yi * xi;
                grad_b = -C * yi;
            end
            w = w - lr * grad_w;
            b = b - lr * grad_b;
        end

        if mod(epoch, 1000) == 0
            margins = Y .* (X * w' + b);
            loss = 0.5 * (w * w') + C * sum(max(0, 1 - margins));
            fprintf('Epoch %d, Loss: %.6f\n', epoch, loss);
        end
    end
end

%-------------------------------------------------------------------------

function features = extract_features_hog_sobel(img)
    % img: 64x64 gri tonlu, double [0 1] arası görüntü
    
    % Sobel filtreleri (x ve y için)
    sobel_x = [1 0 -1; 2 0 -2; 1 0 -1];
    sobel_y = sobel_x';
    
    % Gradyanları hesapla (convolution)
    Gx = conv2(img, sobel_x, 'same');
    Gy = conv2(img, sobel_y, 'same');
    
    % Magnitüd ve açı
    Gmag = sqrt(Gx.^2 + Gy.^2);
    Gdir = atan2d(Gy, Gx);
    
    % Açıları 0-180 derece aralığına getir
    Gdir(Gdir < 0) = Gdir(Gdir < 0) + 180;
    
    % Hücre boyutu ve sayısı
    cellSize = 8;
    numCells = size(img,1) / cellSize; % 64/8=8
    
    % Histogram ayarları
    binEdges = linspace(0, 180, 10); % 9 binli histogram
    
    features = [];
    
    % Hücre bazında histogramları hesapla
    for i = 1:numCells
        for j = 1:numCells
            rows = (i-1)*cellSize+1 : i*cellSize;
            cols = (j-1)*cellSize+1 : j*cellSize;
            cellDirs = Gdir(rows, cols);
            cellMags = Gmag(rows, cols);
            
            histVals = zeros(1,length(binEdges)-1);
            for k = 1:length(histVals)
                binMask = cellDirs >= binEdges(k) & cellDirs < binEdges(k+1);
                histVals(k) = sum(cellMags(binMask));
            end
            
            % L2 norm ile normalize et
            normVal = norm(histVals, 2);
            if normVal > 0
                histVals = histVals / normVal;
            end
            
            features = [features, histVals];
        end
    end
end