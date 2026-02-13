clc; clear; close all;

%% 1. MODEL EĞİTİMİ (HIZLI VE STABİL)
posFolder = 'Real_Person';
negFolder = 'Fake_People';

posFiles = dir(fullfile(posFolder, '*.jpg'));
negFiles = dir(fullfile(negFolder, '*.jpg'));

X = []; Y = [];
fprintf('Model eğitiliyor, lütfen bekleyin...\n');

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

% Normalizasyon ve SVM (RBF Kernel)
mu = mean(X); sigma = std(X); sigma(sigma == 0) = 1;
X_norm = (X - mu) ./ sigma;
SVMModel = fitcsvm(X_norm, Y, 'KernelFunction', 'rbf', 'Standardize', true);

% Eşik Ayarı (Seni tanıması için güvenli alan)
[~, scores] = predict(SVMModel, X_norm);
posScores = scores(Y==1, 2);
safe_threshold = mean(posScores) - 1.8 * std(posScores); 

fprintf('Sistem Hazır! 10 karelik analiz başlıyor...\n');

%% 2. 10 KARELİK ADIM ADIM TEST (KASMA YAPMAZ)
try
    cam = webcam;
catch
    delete(webcamlist); cam = webcam;
end

for i = 1:10
    fprintf('\n%d. Kare için hazırlan... (3 saniye)', i);
    pause(3); % Poz vermen için zaman tanır
    
    % Fotoğrafı çek ve kamerayı o anlık dondur
    img_live = snapshot(cam);
    
    % Analiz et
    feat_live = my_simple_features(img_live);
    feat_norm = (feat_live - mu) ./ sigma;
    [label, score] = predict(SVMModel, feat_norm);
    
    % Karar Mekanizması
    if label == 1 && score(2) > safe_threshold
        fprintf('\n>>> SONUÇ: BU SENSİN (Skor: %.2f)\n', score(2));
        % Görsel onay istersen imshow(img_live) buraya eklenebilir ama şart değil
    else
        fprintf('\n>>> SONUÇ: SEN DEĞİLSİN (Skor: %.2f)\n', score(2));
    end
end

clear cam;
fprintf('\n10 karelik test tamamlandı.\n');

%% --- HAFİF ÖZNİTELİK FONKSİYONU ---
function feat = my_simple_features(img)
    if size(img, 3) == 3, img = rgb2gray(img); end
    img = imresize(img, [64 64]);
    img = im2double(img);

    % Sobel - Kenar
    Gx = conv2(img, [1 0 -1; 2 0 -2; 1 0 -1], 'same');
    Gy = conv2(img, [1 2 1; 0 0 0; -1 -2 -1], 'same');
    Gmag = sqrt(Gx.^2 + Gy.^2);
    
    % FFT - Doku
    F = abs(fftshift(fft2(img)));
    F_log = log(1 + F);

    feat = [mean(Gmag(:)), mean(F_log(:)), std(F_log(:))];
end
