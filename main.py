import numpy as np
import cv2, pafy, argparse, sys
from os import path
import time

# Parancssori argumentumok létrehozása
arg = argparse.ArgumentParser(description='Mozgáskövető biztonsági kamera program')
arg.add_argument("-c", "--cam", type=int, help="Kamera eszközkódja")
arg.add_argument("-v", "--video", help="Videófájl útvonala")
arg.add_argument("-s", "--stream", help="Videostream url-je")
arg.add_argument("-y", "--youtube", help="Youtube videó url-je")
arg.add_argument("-a", "--min-area", type=int, default=700, help=" Minimális területméret(nem kötelező, alapból 500 px)")
arg.add_argument("-d", "--distance", type=int, default=10, help=" Maximális távolság két kontúr között(nem kötelező, alapból 10 px)")
args = arg.parse_args()

# Parancssori argumentumok meglétének ellenőrzése
if args.cam is not None:
    inp = args.cam
elif args.video is not None:
    inp = args.video
    if path.exists(inp) is False:
        print("A fájl nem létezik!")
        sys.exit()
elif args.stream is not None:
    inp = args.stream           # EXCEPTION NINCS BENNE, ELFOGAD HIBÁS ÉRTÉKET IS
elif args.youtube is not None:
    url = args.youtube
    try:
        video = pafy.new(url)
    except ValueError:
        print("Nem érvényes youtube videó link")
        sys.exit()
    best = video.getbest(preftype="any")
    inp = best.url
else:
    arg.print_help()
    sys.exit()

# Video input
cap = cv2.VideoCapture(inp)

# változók alapértékének beállítása
x = y = cX = cY = 0

# kezdő frame beolvasása
_, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
ms = int(round(time.time() * 1000))
# program start
while cap.isOpened():
    frame2 = frame  # előző frame rögzítése
    ret, frame = cap.read()
    if (ret):
        # szürkeárnyalati konverzió
        f_gray_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(int)
        f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)
        # kivonás és abszolútérték vétel
        f_sub = f_gray_2 - f_gray
        f_abs = np.abs(f_sub)
        # Gauss-szűrő
        f_gauss = cv2.GaussianBlur(f_abs.astype(np.uint8), (9, 9), 0)
        # Küszöbölés
        f_mov = (f_gauss.astype(int) > 20) * 255
        # Nyitás
        kernel = np.ones((3, 3), np.uint8)
        f_open = cv2.morphologyEx(f_mov.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
        # kontúrok méretének megnövelése(dilatáció)
        f_dil = cv2.dilate(f_open, None, iterations=3)
        # kontúrok keresése
        contours, _ = cv2.findContours(f_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # megfelelő kontúr bekeretezése
        if len(contours) != 0:
            cns = sorted(contours, key=cv2.contourArea, reverse=True) # a kontúrok rendezése csökkenő sorrendbe
            c = max(contours, key=cv2.contourArea)  # a legnagyobb kontúr megtalálása
            for cn in cns:
                x1, y1, w1, h1 = cv2.boundingRect(cn)
                mom = cv2.moments(cn)
                cX1 = int(mom["m10"] / mom["m00"])
                cY1 = int(mom["m01"] / mom["m00"])
                # Sarkopontok és Középpontok vizsgálata
                if ((abs(x1 - x) <= args.distance & abs(y1 - y) <= args.distance) | (abs(cX1 - cX) <= args.distance & abs(cY1 - cY) <= args.distance)) & (
                        cv2.contourArea(cn) > args.min_area):
                    c = cn
                    break
            if cv2.contourArea(c) > args.min_area:  # ha a kontúr területe nagyobb mint min_area
                x, y, w, h = cv2.boundingRect(c)  # kontúr adatai(x,y sarok, w,h méretei)
                m = cv2.moments(c)
                cX = int(m["m10"] / m["m00"])
                cY = int(m["m01"] / m["m00"])
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # zöld négyzet rajzolása a frame2 képre
            else:
                pass
        # megjelenítés
        frame2rs = cv2.resize(frame2, (960, 540))  # frame átméretezése
        # a megjelenítést a videó fps-hez igazítása
        while int(round(time.time() * 1000)) < ms+(1000 / fps):
            pass
        cv2.imshow('Kimenet', frame2rs)
        ms = int(round(time.time() * 1000))
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q-val kilép
            break
    else:
        break
cap.release()  # capture bezárása
cv2.destroyAllWindows()  # ablakok bezárása
