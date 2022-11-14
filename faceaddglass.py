#-- coding:UTF-8 --
import numpy as np
import cv2 as cv
import dlib
import math

# 做一个戴眼镜的滤镜效果

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/admin/Downloads/shape_predictor_68_face_landmarks.dat')


# 图像旋转，保持原来大小
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


def detect_face(camera_idx):
    # camera_idx: 电脑自带摄像头或者usb摄像头
    # cv.namedWindow('detect')
    cap = cv.VideoCapture(0)

    while (True):

        # cv.namedWindow('detect', cv.WINDOW_AUTOSIZE)
        cv.namedWindow('detect', 0)
        ok, frame = cap.read()

        # print(f'ok : {ok} , frame : {frame}')

        # 为摄像头的时候，翻转画面
        if camera_idx == 0 or camera_idx == 1:
            frame = cv.flip(frame, 1, dst=None)
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        
        rects = detector(gray, 0)
        print (f'rectes : {len(rects)}')
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rects[i]).parts()])
            # 脸轮廓：1~17
            # 眉毛：18~22, 23~27
            # 鼻梁：28~31
            # 鼻子：31~36
            # 眼睛：37~42, 43~48
            # 嘴唇：49~68
            # 左眼角和右眼角的位置
            pos_left = (landmarks[0][0, 0], landmarks[36][0, 1])
            pos_right = (landmarks[16][0, 0], landmarks[45][0, 1])
            face_center = (landmarks[27][0, 0], landmarks[27][0, 1])
            src = cv.imread('/Users/admin/Downloads/glasses.jpeg')
            #img.shape[0]的值为图片的高度,shape[1]的值为图片的宽度,shape[3]的值为图片的通道3
            print (f'src_shape : {src.shape}')
            g_h,g_w = src.shape[0],src.shape[1]
            print(f'src_shape : {g_h},{g_w}')
            # 433x187眼镜图片原始大小，按人脸比例缩放一下
            length = pos_right[0] - pos_left[0]
            # width = int(187/(433/length))
            width = int(g_h / (g_w / length))
            src = cv.resize(src, (length, width), interpolation=cv.INTER_CUBIC)

            # 角度旋转，通过计算两个眼角和水平方向的夹角来旋转眼镜
            sx = landmarks[36][0, 0] - landmarks[45][0, 0]
            sy = landmarks[36][0, 1] - landmarks[45][0, 1]
            # 夹角正切值
            r = sy/sx
            # 求正切角,弧度转为度
            degree = math.degrees(math.atan(r))
            # 调用旋转方法
            src = rotate_bound(src, degree)

            # mask处理，去掉旋转后的无关区域，初始化一个全0mask，用或运算处理mask
            src_mask = np.zeros(src.shape, src.dtype)
            src_mask = cv.bitwise_or(src, src_mask)
            # 泊松融合
            output = cv.seamlessClone(src, frame, src_mask, face_center, cv.MIXED_CLONE)
        cv.imshow('detect', output)

        c = cv.waitKey(5)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def test_mac_camre():


    # 创建VideoCapture，传入0即打开系统默认摄像头
    vc = cv.VideoCapture(0)

    while (True):
        # 读取一帧，read()方法是其他两个类方法的结合，具体文档
        # ret为bool类型，指示是否成功读取这一帧
        ret, frame = vc.read()
        # 就是个处理一帧的例子，这里转为灰度图
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 不断显示一帧，就成视频了
        # 这里没有提前创建窗口，所以默认创建的窗口不可调整大小
        # 可提前使用cv.WINDOW_NORMAL标签创建个窗口
        cv.imshow('frame', gray)
        # 若没有按下q键，则每1毫秒显示一帧
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # 所有操作结束后不要忘记释放
    vc.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # video = 'video/face.mp4'
    video = 0
    detect_face(video)
    # test_mac_camre()