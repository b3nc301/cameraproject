import numpy as np
import cv2
import pafy

###TESZTELÉS VIDEO STREAMBÓL
url = "https://www.youtube.com/watch?v=xeuWNm72YRg"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

#Video input módok megadása
## ('filenev.mp4') -> mp4 source
## (0) -> main camera
## (video_url) -> youtube video
cap = cv2.VideoCapture(best.url)

#kezdő frame beolvasása
_, frame = cap.read()
#program start
while(cap.isOpened()):
    frame2=frame    # előző frame rögzítése
    ret, frame = cap.read() # következő frame beolvasás
    #szürkeárnyalati konverzió
    f_gray_2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY).astype(int)
    f_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(int)
    #kivonás és abszolútérték vétel
    f_sub = f_gray_2 - f_gray
    f_abs = np.abs(f_sub)
    #Gauss-szűrő
    f_gauss = cv2.GaussianBlur(f_abs.astype(np.uint8), (9, 9), 0)
    #Korrekció
    f_mov = (f_abs.astype(int) > 20) * 255

    #kontúrok méretének megnövelése
    f_dil= cv2.dilate(f_mov.astype(np.uint8), None, iterations=3)
    #kontúrok keresése
    contours,_ = cv2.findContours(f_dil,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #kontúr(ok) bekeretezése
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea) # a legnagyobb kontúr megtalálása
        if(cv2.contourArea(c)>800): #ha a kontúr területe nagyobb mint 800
            x, y, w, h = cv2.boundingRect(c) # kontúr adatai(x,y sarok, w,h méretei)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2) # zöld négyzet rajzolása a frame2 képre
        else:
            pass
    #megjelenítés
    frame2rs = cv2.resize(frame2, (960, 540))  # frame átméretezése
    cv2.imshow('MUKODJEL MAR', frame2rs)

    if cv2.waitKey(1) & 0xFF == ord('q'):   #q-val kilép, vagy ha vége a videóstreamnek
        break

cap.release() #capture bezárása
cv2.destroyAllWindows() #ablakok bezárása