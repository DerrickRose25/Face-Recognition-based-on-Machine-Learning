import cv2


class FacialRecognition:
    def __init__(self):
        faces = {'luxin'}

    def analysis(self, frame):
        classfier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        # 识别出人脸后画出边框
        color = (0, 255, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前帧转变为灰度图
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        # scaleFactor：图像缩放比例，可以理解为同一个物体与相机距离不同，其大小亦不同，必须将其缩放到一定大小才方便识别，
        # 该参数指定每次缩放的比例。
        # minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏。minSize：特征检测点的最小值
        faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                # cv2.rectangle()完成画框的工作，在这里外扩了10个像素以框出比人脸稍大一点的区域。
                # cv2.rectangle()函数的最后两个参数一个用于指定矩形边框的颜色，一个用于指定矩形边框线条的粗细程度。


if __name__ == '__main__':
    cameraCapture = cv2.VideoCapture(0)
    while True:
        ret, frame = cameraCapture.read()
        ld = FacialRecognition()
        ld.analysis(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cameraCapture.release()
    cv2.destroyAllWindows()
