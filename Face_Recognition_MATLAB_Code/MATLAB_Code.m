import sys
import os

# 1. Kütüphane Bypass
sys.path = [p for p in sys.path if "anaconda3" not in p.lower()]
user_site = os.path.expanduser("~\\AppData\\Roaming\\Python\\Python311\\site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

from sklearn import svm
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import time

# --- ÖZNİTELİK ÇIKARMA ---
def extract_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    f_log = np.log(1 + np.abs(fshift))
    return [np.mean(gmag), np.mean(f_log), np.std(f_log)]

# --- 2. EĞİTİM ---
pos_path = 'Real_Person'
neg_path = 'Fake_People'

X, Y = [], []

if not os.path.exists(pos_path) or not os.path.exists(neg_path):
    print(f"!!! KRİTİK HATA: Klasörler bulunamadı!")
else:
    print("Durum: Model eğitiliyor (Dengeli Mod)...")
    for folder, label in [(pos_path, 1), (neg_path, -1)]:
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for file in files:
            img = cv2.imread(os.path.join(folder, file))
            if img is not None:
                X.append(extract_features(img))
                Y.append(label)

    X = np.array(X)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # Class weight='balanced' ile 20 resminin ağırlığını 100 resme eşitledik
    clf = svm.SVC(kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_norm, Y)

    pos_scores = clf.decision_function(X_norm[np.array(Y) == 1])
    
    # Eşiği 3.5 yaparak açılardan kaynaklanan hata payını genişlettik
    security_threshold = np.mean(pos_scores) - 3.5 * np.std(pos_scores)
    print(f"Sistem Hazır! Hedef Eşik: {security_threshold:.4f}")

    # --- 3. TEST ---
    cap = cv2.VideoCapture(0)
    print("Kamera açılıyor, lütfen hazırlanın...")
    time.sleep(2)
    
    for i in range(1, 11):
        ret, frame = cap.read()
        if not ret: break
        
        feat = np.array([extract_features(frame)])
        feat_norm = scaler.transform(feat)
        score = clf.decision_function(feat_norm)[0]
        
        # Karar
        is_granted = score > security_threshold
        status = "ONAYLANDI" if is_granted else "REDDEDİLDİ"
        color = (0, 255, 0) if is_granted else (0, 0, 255)
        
        # Ekrana Skor ve Durum yazdır
        cv2.putText(frame, f"{status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Skor: {score:.2f} (Esik: {security_threshold:.2f})", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Python Yuz Tanima - Test', frame)
        print(f"Kare {i}/10: {status} | Skor: {score:.2f}")
        
        if cv2.waitKey(1000) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
