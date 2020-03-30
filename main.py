import numpy as np
import cv2

cap = cv2.VideoCapture('vid.mp4')

_, frame = cap.read()

while(cap.isOpened()):
    frame2=frame    # előző frame rögzítése
    ret, frame = cap.read() # következő frame beolvasás
    ##szürkeárnyalati konverzió
    f_gray_2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY).astype(int)
    f_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(int)
    ##kivonás és abszolútérték vétel
    f_sub = f_gray_2 - f_gray
    f_abs = np.abs(f_sub)
    ##Korrekció
    f_mov = (f_abs > 20) * 255
    ##mediánszűrő
    f_med = cv2.medianBlur(f_mov.astype(np.uint8), 3)




    cv2.imshow('frame',frame)
    cv2.imshow('derivalt',f_med)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()