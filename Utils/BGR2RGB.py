import cv2

path = 'TestImgs/LocationColorTrans_I2.png'

I1 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
cv2.imwrite(path, I1)