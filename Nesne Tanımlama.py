

import time
import cv2
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import sys
import time



# Metin renkleri
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

    # Arka plan renkleri
BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

# Metin stilleri
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
BLINK = "\033[5m"
REVERSE = "\033[7m"
HIDDEN = "\033[8m"

# Renkleri sıfırlama
RESET = "\033[0m"
RESET_COLOR = "\033[39m"
RESET_BG = "\033[49m"



# Sabit prototxt ve model dosya yolları
prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
model_path = "MobileNetSSD_deploy.caffemodel"


# Diğer parametreler
confidence_threshold = 0.2
video_source = 0  # Video kaynağı, genellikle 0 kamerayı temsil eder


# Eğitilmiş modeller için derlenmiş bir CLASS bütününü çağırır
CLASSES = ["ARKAPLAN", "UCAK", "BISIKLET", "KUS", "TEKNE", 
	"SISE", "OTOBUS", "ARABA", "KEDI", "SANDALYE", "INEK", "MASA", 
	"KOPEK", "AT", "MOTOSIKLET", "INSAN", "SAKSI", "KOYUN", 
	"KANEPE", "TREN", "TELEVIZYON MONITORU"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))



# Modeli yükle
def loading_effect():
    loading_text = f"{RED}MODELLER YÜKLENİYOR{RESET}"
    for i in range(10):  # Döngü sayısını animasyon süresine göre ayarlayabilirsiniz
        sys.stdout.write("\r" + loading_text + "." * (i % 8))  # "." animasyonu oluşturur
        sys.stdout.flush()
        time.sleep(0.2)  # Her adım için 0.5 saniye bekler
    print("\r"+f"{GREEN}KAMERA BAŞLATILIYOR  {RESET}"   )  # Animasyonu tamamladıktan sonra mesaj yaz

loading_effect()
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


# Video akışını başlat
print(f"{YELLOW}ÇIKIŞ İÇİN 'q' TUŞUNA BASIN{RESET}")
vs = VideoStream(src=video_source).start()
time.sleep(2.0)
fps = FPS().start()


# Tam ekran pencere oluştur
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Video akışındaki her bir frame üzerinde döngü
while True:
	# Video akışından frame al
	frame = vs.read()

	# 720p çözünürlüğünde (1280x720) ve 16:9 oranında işleme
	frame = imutils.resize(frame, width=1280)  # 720p çözünürlük için genişliği 1280 yapıyoruz

	# Frame boyutlarını al ve blob'a dönüştür
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (200, 400)),
		0.007843, (200, 400), 127.5)

	# Blob'u ağdan geçir ve tahminleri al
	net.setInput(blob)
	detections = net.forward()

	# Tespitleri işle
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > confidence_threshold:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Tahmini ekrana çiz
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# Sonuçları tam ekran olarak göster
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q' tuşuna basılınca döngüden çık
	if key == ord("q") or key == ord("Q"):
		break

	# FPS sayacını güncelle
	fps.update()

# Zamanlayıcıyı durdur ve FPS bilgilerini göster
fps.stop()
print(f"{MAGENTA}GEÇEN SÜRE :  {fps.elapsed():.2f} {RESET}")
print(f"{MAGENTA}ORTALAMA FPS: {fps.fps():.2f} {RESET}")

# Temizlik işlemleri
cv2.destroyAllWindows()
vs.stop()


