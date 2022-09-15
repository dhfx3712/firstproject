import cv2 as cv
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys, os, json, random


class LPRAlg:
    maxLength = 700
    minArea = 2000

    def __init__(self, imgPath=None):
        if imgPath is None:
            print("Please input correct path!")
            return None

        self.imgOri = cv.imread(imgPath)
        if self.imgOri is None:
            print("Cannot load this picture!")
            return None

    # #cv.imshow("imgOri", self.imgOri)


    def plt_img(self,img,title):
        fig,ax = plt.subplots(figsize=(2,2))
        ax.set_title(title)
        ax.imshow(img)
        plt.show()



    def accurate_place(self, imgHsv, limit1, limit2, color):
        rows, cols = imgHsv.shape[:2]
        left = cols
        right = 0
        top = rows
        bottom = 0

        # rowsLimit = 21
        rowsLimit = rows * 0.8 if color != "green" else rows * 0.5  # 绿色有渐变
        colsLimit = cols * 0.8 if color != "green" else cols * 0.5  # 绿色有渐变
        for row in range(rows):
            count = 0
            for col in range(cols):
                H = imgHsv.item(row, col, 0)
                S = imgHsv.item(row, col, 1)
                V = imgHsv.item(row, col, 2)
                if limit1 < H <= limit2 and 34 < S:  # and 46 < V:
                    count += 1
            if count > colsLimit:
                if top > row:
                    top = row
                if bottom < row:
                    bottom = row
        for col in range(cols):
            count = 0
            for row in range(rows):
                H = imgHsv.item(row, col, 0)
                S = imgHsv.item(row, col, 1)
                V = imgHsv.item(row, col, 2)
                if limit1 < H <= limit2 and 34 < S:  # and 46 < V:
                    count += 1
            if count > rowsLimit:
                if left > col:
                    left = col
                if right < col:
                    right = col
        return left, right, top, bottom

    def findVehiclePlate(self):
        def zoom(w, h, wMax, hMax):
            # if w <= wMax and h <= hMax:
            # 	return w, h
            widthScale = 1.0 * wMax / w
            heightScale = 1.0 * hMax / h

            scale = min(widthScale, heightScale)

            resizeWidth = int(w * scale)
            resizeHeight = int(h * scale)

            return resizeWidth, resizeHeight

        def pointLimit(point, maxWidth, maxHeight):
            if point[0] < 0:
                point[0] = 0
            if point[0] > maxWidth:
                point[0] = maxWidth
            if point[1] < 0:
                point[1] = 0
            if point[1] > maxHeight:
                point[1] = maxHeight

        if self.imgOri is None:
            print("Please load picture frist!")
            return False

        # Step1: Resize
        img = np.copy(self.imgOri)
        h, w = img.shape[:2]
        imgWidth, imgHeight = zoom(w, h, self.maxLength, self.maxLength)#按self.maxLength调整resize方法
        print("resize 前后",w, h, imgWidth, imgHeight) # 717 543 700 530
        img = cv.resize(img, (imgWidth, imgHeight), interpolation=cv.INTER_AREA)
        #cv.imshow("imgResize", img)
        self.plt_img(img,"resize")

        # Step2: Prepare to find contours
        img = cv.GaussianBlur(img, (3, 3), 0)
        imgGary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #cv.imshow("imgGary", imgGary)
        self.plt_img(imgGary, "imgGary") #色彩转化

        kernel = np.ones((20, 20), np.uint8)
        imgOpen = cv.morphologyEx(imgGary, cv.MORPH_OPEN, kernel)
        #morphologyEx形态学函数。腐蚀是最基本的形态学操作之一，它能够将图像的边界点消除。膨胀是将与物体接触的所有背景点合并到该物体中，使边界向外部扩张的过程。可以用来填补物体中的空洞.
        #cv.imshow("imgOpen", imgOpen)
        self.plt_img(imgOpen, "open")

        imgOpenWeight = cv.addWeighted(imgGary, 1, imgOpen, -1, 0)
        #cv.imshow("imgOpenWeight", imgOpenWeight)
        self.plt_img(imgOpenWeight, "imgopenweight")

        ret, imgBin = cv.threshold(imgOpenWeight, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
        #cv.imshow("imgBin", imgBin)
        self.plt_img(imgBin, "imgopenweight--bin")

        imgEdge = cv.Canny(imgBin, 100, 200) #边缘 100, 200上下界限
        #cv.imshow("imgEdge", imgEdge)
        self.plt_img(imgEdge, "canny")

        kernel = np.ones((10, 19), np.uint8)
        #开运算：表示的是先进行腐蚀，再进行膨胀操作
        imgEdge = cv.morphologyEx(imgEdge, cv.MORPH_CLOSE, kernel)
        #闭运算：表示先进行膨胀操作，再进行腐蚀操作
        imgEdge = cv.morphologyEx(imgEdge, cv.MORPH_OPEN, kernel)
        #cv.imshow("imgEdgeProcessed", imgEdge)

        # Step3: Find Contours
        # image, contours, hierarchy = cv.findContours(imgEdge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(imgEdge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > self.minArea]

        print (f'contours_nums : {len(contours)}')
        # Step4: Delete some rects
        carPlateList = []
        imgDark = np.zeros(img.shape, dtype=img.dtype)
        self.plt_img(imgDark,"imgDark")
        for index, contour in enumerate(contours):
            rect = cv.minAreaRect(contour)  # [中心(x,y), (宽,高), 旋转角度]
            w, h = rect[1]
            if w < h:
                w, h = h, w
            scale = w / h
            print ( '%s--scale : %s'%(index,scale))
            if scale > 2 and scale < 4:
                # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color = (255, 255, 255) #白
                carPlateList.append(rect)
                cv.drawContours(imgDark, contours, index, color, 1, 8)
                self.plt_img(imgDark,"imgdark1-%s"%index)

                box = cv.boxPoints(rect)  # Peak Coordinate
                box = np.int0(box)
                # Draw them out
                cv.drawContours(imgDark, [box], 0, (0, 0, 255), 1) #(0, 0, 255)蓝
                self.plt_img(imgDark, "imgdark2-%s" % index)

        #cv.imshow("imgGaryContour", imgDark)
        print("carPlateList: ", len(carPlateList),carPlateList)

        # Step5: Rect rectify
        imgPlatList = []
        for index, carPlat in enumerate(carPlateList):
            print (f'carplat -- {carPlat}')
            if carPlat[2] > -1 and carPlat[2] < 1:
                angle = 1
            else:
                angle = carPlat[2]

            carPlat = (carPlat[0], (carPlat[1][0] + 5, carPlat[1][1] + 5), angle)
            print (f'fix_carplat -- {carPlat}')
            box = cv.boxPoints(carPlat) #获取矩形四个顶点，浮点型
            print (f'box--{index}, {box}')
            # Which point is Left/Right/Top/Bottom
            w, h = carPlat[1][0], carPlat[1][1]
            if w > h:
                LT = box[1]
                LB = box[0]
                RT = box[2]
                RB = box[3]
            else:
                LT = box[2]
                LB = box[1]
                RT = box[3]
                RB = box[0]

            for point in [LT, LB, RT, RB]:
                pointLimit(point, imgWidth, imgHeight) #区间限制

            # Do warpAffine
            newLB = [LT[0], LB[1]]
            newRB = [RB[0], LB[1]]
            print (f'newLB -- {newLB} , newRB -- {newRB}')
            # oldTriangle = np.float32([LT, LB, RB])
            # newTriangle = np.float32([LT, newLB, newRB])
            # print (f'oldtriangl :{oldTriangle} --- newtriangle :{newTriangle}')
            # 仿射变换
            # warpMat = cv.getAffineTransform(oldTriangle, newTriangle) #矩阵
            # imgAffine = cv.warpAffine(img, warpMat, (imgWidth, imgHeight)) #旋转和平移（放射变化）
            # #cv.imshow("imgAffine" + str(index), imgAffine)
            # self.plt_img(imgAffine,"affine-%s"%index)
            # imgPlat = imgAffine[int(LT[1]):int(newLB[1]), int(newLB[0]):int(newRB[0])]

            # 透视变化 (变化后坐标不对)
            oldTriangle = np.float32([LT, LB, RB ,RT])
            newTriangle = np.float32([LT, newLB, newRB ,RT])
            #缺少畸变处理
            # warpMat = cv.getPerspectiveTransform(box, np.array([box[0],box[1],[box[2][0],box[0][1]],[box[3][0],box[1][1]]]))
            warpMat = cv.getPerspectiveTransform(oldTriangle,newTriangle)
            imgPerspective = cv.warpPerspective(img, warpMat, (imgWidth, imgHeight)) #透视变化
            #cv.imshow("imgAffine" + str(index), imgAffine)
            self.plt_img(imgPerspective,"affine-%s"%index)
            imgPlat = imgPerspective[int(LT[1]):int(newLB[1]), int(newLB[0]):int(newRB[0])]

            imgPlatList.append(imgPlat)
            self.plt_img(imgPlat,"plat-%s"%index)
            #cv.imshow("imgPlat" + str(index), imgPlat)

        # print (f"imgPlatList : {len(imgPlatList)},{imgPlatList}")
        # Step6: Find correct place by color.
        colorList = []
        for index, imgPlat in enumerate(imgPlatList):
            green = yellow = blue = 0
            imgHsv = cv.cvtColor(imgPlat, cv.COLOR_BGR2HSV) #颜色空间转换
            self.plt_img(imgHsv,'imghsv')
            rows, cols = imgHsv.shape[:2]
            print (f'rows ,cols : {rows},{cols}')
            imgSize = cols * rows
            color = None

            for row in range(rows):
                for col in range(cols):
                    H = imgHsv.item(row, col, 0)
                    S = imgHsv.item(row, col, 1)
                    V = imgHsv.item(row, col, 2)

                    if 11 < H <= 34 and S > 34:
                        yellow += 1
                    elif 35 < H <= 99 and S > 34:
                        green += 1
                    elif 99 < H <= 124 and S > 34:
                        blue += 1

            limit1 = limit2 = 0
            if yellow * 3 >= imgSize:
                color = "yellow"
                limit1 = 11
                limit2 = 34
            elif green * 3 >= imgSize:
                color = "green"
                limit1 = 35
                limit2 = 99
            elif blue * 3 >= imgSize:
                color = "blue"
                limit1 = 100
                limit2 = 124

            print("Image Index[", index, '], Color：', color)
            colorList.append(color)
            print(blue, green, yellow, imgSize)

            if color is None:
                continue

            # Step7: Resize vehicle img.
            left, right, top, bottom = self.accurate_place(imgHsv, limit1, limit2, color)
            w = right - left
            h = bottom - top

            if left == right or top == bottom:
                continue

            scale = w / h
            if scale < 2 or scale > 4:
                continue

            needAccurate = False
            if top >= bottom:
                top = 0
                bottom = rows
                needAccurate = True
            if left >= right:
                left = 0
                right = cols
                needAccurate = True
            # imgPlat[index] = imgPlat[top:bottom, left:right] \
            # if color != "green" or top < (bottom - top) // 4 \
            # else imgPlat[top - (bottom - top) // 4:bottom, left:right]
            imgPlatList[index] = imgPlat[top:bottom, left:right]
            self.plt_img(imgPlatList[index],"find from color")
            #cv.imshow("Vehicle Image " + str(index), imgPlatList[index])


if __name__ == '__main__':
    L = LPRAlg("./car_license.jpeg")
    # L = LPRAlg("./1111.jpg")
    L.findVehiclePlate()
    # cv.waitKey(0)
    # cv.destroyAllWindows()


'''
车牌畸变与矫正
https://www.cnblogs.com/HL-space/p/10588423.html

'''