import numpy as np
import cv2, pafy, argparse, sys

# Parancssori argumentumok tesztelése
arg = argparse.ArgumentParser(description='Motion detector security camera / Mozgáskövető biztonsági kamera program')
arg.add_argument("-v", "--video", help="path to the video file / Videófájl útvonala")
arg.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size / Minimális területméret")
arg.add_argument("-c", "--cam", type=int, help="camera device code/Kamera eszközkódja")
arg.add_argument("-s", "--stream", help="stream url/Videostream url-je")
args = vars(arg.parse_args())
if len(sys.argv) == 1:
    arg.print_help()
    sys.exit()


# TESZTELÉS VIDEO STREAMBÓL


url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
url = "https://www.youtube.com/watch?v=2dysaG-q6Lc"
url = "https://www.youtube.com/watch?v=xeuWNm72YRg"

url = "https://www.youtube.com/watch?v=mRe-514tGMg"

url = "https://www.youtube.com/watch?v=jjlBnrzSGjc"
url = "https://www.youtube.com/watch?v=lZsDve8_DkM"
url = "https://www.youtube.com/watch?v=CkVJyAKwByw"

video = pafy.new(url)
best = video.getbest(preftype="mp4")

# Video input módok megadása
# -('filenev.mp4') -> mp4 source
# -(0) -> main camera
# -(best.url) -> youtube video
x = y = cX = cY = 0
cap = cv2.VideoCapture(best.url)
# kezdő frame beolvasása
_, frame = cap.read()
# változók beállítása
# program start
while cap.isOpened():
    frame2 = frame  # előző frame rögzítése
    ret, frame = cap.read()
    # szürkeárnyalati konverzió
    f_gray_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(int)
    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)
    # kivonás és abszolútérték vétel
    f_sub = f_gray_2 - f_gray
    f_abs = np.abs(f_sub)
    # Gauss-szűrő
    f_gauss = cv2.GaussianBlur(f_abs.astype(np.uint8), (9, 9), 0)
    # Küszöbölés
    f_mov = (f_abs.astype(int) > 20) * 255
    # nyitás
    f_open = cv2.morphologyEx(f_mov.astype(np.uint8), cv2.MORPH_OPEN, (3, 3), iterations=2)
    # kontúrok méretének megnövelése(dilettáció)
    f_dil = cv2.dilate(f_open, None, iterations=3)
    # kontúrok keresése
    contours, _ = cv2.findContours(f_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # kontúr(ok) bekeretezése
    if len(contours) != 0:
       # cns = sorted(contours, key=cv2.contourArea, reverse=True) # a kontúrok rendezése csökkenő sorrendbe
        c = max(contours, key=cv2.contourArea)  # a legnagyobb kontúr megtalálása
        cns = sorted(contours, key=cv2.contourArea, reverse=True)
        for cn in cns:
            x1, y1, w1, h1 = cv2.boundingRect(cn)
            mom = cv2.moments(cn)
            cX1 = int(mom["m10"] / mom["m00"])
            cY1 = int(mom["m01"] / mom["m00"])
            if ((abs(x1 - x) <= 10 & abs(y1 - y) <= 10) | (abs(cX1 - cX) <= 10 & abs(cY1 - cY) <= 10)) & (
                    cv2.contourArea(cn) > 400):
                # Ha az előző kontúr és a jelenlegi kontúr sarka között 5 képpont vagy kevesebb van akkor ez a kontúr lesz a jó
                c = cn
                break
            else:
                continue
        if cv2.contourArea(c) > 400:  # ha a kontúr területe nagyobb mint 800
            x, y, w, h = cv2.boundingRect(c)  # kontúr adatai(x,y sarok, w,h méretei)
            m = cv2.moments(c)
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            ''' if(cv2.boundingRect(max(contours, key=cv2.contourArea))!=cv2.boundingRect(c)):
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 2)  # zöld négyzet rajzolása a frame2 képre
            else:'''
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # zöld négyzet rajzolása a frame2 képre
        else:
            pass
    # megjelenítés
    frame2rs = cv2.resize(frame2, (960, 540))  # frame átméretezése
    cv2.imshow('Kimenet', frame2rs)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q-val kilép, vagy ha vége a videóstreamnek
        break

cap.release()  # capture bezárása
cv2.destroyAllWindows()  # ablakok bezárása
