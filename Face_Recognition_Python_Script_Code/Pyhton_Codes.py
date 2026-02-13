#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import time
import os

# Resimlerin kaydedileceği klasörü oluştur
folder = 'sahte_insanlar'
if not os.path.exists(folder):
    os.makedirs(folder)

print("İndirme başlıyor...")

for i in range(100):
    try:
        # Siteye istek at (Header eklemek bazen bot korumasını aşmak için gerekebilir)
        response = requests.get("https://thispersondoesnotexist.com", headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            with open(f"{folder}/person_{i+1}.jpg", 'wb') as f:
                f.write(response.content)
            print(f"Resim {i+1} indirildi.")
        
        # Siteyi yormamak ve banlanmamak için kısa bir ara (Opsiyonel)
        time.sleep(0.5) 
        
    except Exception as e:
        print(f"Hata oluştu: {e}")

print("İşlem tamamlandı!")


# In[ ]:




