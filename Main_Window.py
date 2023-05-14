from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import time
import threading
from PyQt5.QtWidgets import QApplication, QProgressDialog
from PyQt5.QtCore import QLibraryInfo

from loguru import logger
import sys
import cv2
import torch
from my_predictor import Predictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_exp import Exp

from visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from detectors.yolox.tracking_utils.timer import Timer


import argparse
import os
import time


os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

def preimage_flow(args,frame_rate=30):
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    return tracker,timer,frame_id,results
    

def imageflow_demo(predictor, tracker,timer,frame_id,results,ret_val,frame, args,exp):
        if frame_id % 1 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        
        if ret_val:
            
            outputs, img_info = predictor.inference(frame, timer)
            tracker.isyolox=(predictor.exp._model == "yolox")
            # print(outputs[0])
            if outputs[0] is not None:
                
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_clss=[]
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_clss.append(t.cls)
                timer.toc()
                # save results
        
                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores,online_clss))

                if predictor.exp._model == "ovd":
                    thing_classes=predictor.model.model.metadata.thing_classes
                else:
                    thing_classes=None
                
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, online_clss,frame_id=frame_id + 1,
                                          fps=1. / timer.average_time,model_type=predictor.exp._model,thing_classes=thing_classes)
                
                
                
                
                
            else:
                timer.toc()
                online_im = img_info['raw_img']

            rgb_frame = cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)  # 转成rgb形式
            h, w, ch = rgb_frame.shape  # 获取scale
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            # 把处理后的图像展示在 detection Label 上
            
            detection_pixmap = QtGui.QPixmap.fromImage(qimg)

        
            

            
            
        frame_id += 1
        return detection_pixmap,tracker,timer,frame_id,results


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--detector",default="yolox", help="choose your detector, eg. yolox-s, yolox-m, yolox-l, yolox-x"
    )

    parser.add_argument(
        "--path", default="/home/workspace/ByteTrack/videos/palace.mp4", help="path to images or video"
    )
    
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument(
        "--vocabulary",
        default='lvis',
        type=str,
        help="vocabulary of the dataset. Now support lvis and coco and custom.",
    )

    parser.add_argument(
        "--thing_classes",
        default=None,
        type=str,
        help="predict only these thing classes. Only valid when vocabulary is custom. Using ',' to split classes.",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.1, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Ui_MainWindow(object):
    # UI界面的代码
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1838, 1016)
        MainWindow.setStyleSheet("background-color: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(116, 193, 252, 255), stop:1 rgba(255, 255, 255, 255));")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(20, 210, 891, 691))
        self.camera.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.camera.setObjectName("camera")
        self.detection = QtWidgets.QLabel(self.centralwidget)
        self.detection.setGeometry(QtCore.QRect(920, 210, 881, 691))
        self.detection.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.detection.setObjectName("detection")
        self.Begin = QtWidgets.QPushButton(self.centralwidget)
        self.Begin.setGeometry(QtCore.QRect(390, 130, 89, 25))
        self.Begin.setMinimumSize(QtCore.QSize(0, 25))
        self.Begin.setStyleSheet("background-color: rgba(153, 193, 241, 0);")
        self.Begin.setObjectName("Begin")
        self.Pause = QtWidgets.QPushButton(self.centralwidget)
        self.Pause.setGeometry(QtCore.QRect(510, 130, 89, 25))
        self.Pause.setStyleSheet("background-color: rgba(153, 193, 241, 0);")
        self.Pause.setObjectName("Pause")
        self.Intro = QtWidgets.QLabel(self.centralwidget)
        self.Intro.setGeometry(QtCore.QRect(1440, 170, 301, 16))
        self.Intro.setStyleSheet("")
        self.Intro.setObjectName("Intro")
        self.Caption = QtWidgets.QLabel(self.centralwidget)
        self.Caption.setGeometry(QtCore.QRect(550, 30, 721, 121))
        self.Caption.setStyleSheet("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 235, 235, 206), stop:0.35 rgba(188, 234, 255, 80), stop:0.4 rgba(137, 214, 255, 80), stop:0.425 rgba(132, 208, 255, 156), stop:0.44 rgba(128, 252, 247, 80), stop:1 rgba(255, 255, 255, 0));")
        self.Caption.setObjectName("Caption")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(20, 130, 86, 25))
        self.comboBox.setStyleSheet("background-color: rgba(153, 193, 241, 0);\n"
"selection-background-color: rgb(153, 193, 241);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_choose_model = QtWidgets.QLabel(self.centralwidget)
        self.label_choose_model.setGeometry(QtCore.QRect(20, 100, 91, 21))
        self.label_choose_model.setStyleSheet("background-color: rgba(255, 255, 255, 0%);")
        self.label_choose_model.setObjectName("label_choose_model")
        self.Input = QtWidgets.QLineEdit(self.centralwidget)
        self.Input.setGeometry(QtCore.QRect(130, 130, 113, 25))
        self.Input.setStyleSheet("background-color: rgba(153, 193, 241, 100);")
        self.Input.setObjectName("Input")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(130, 100, 131, 21))
        self.label.setStyleSheet("background-color: rgba(153, 193, 241,0);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(270, 100, 67, 17))
        self.label_2.setStyleSheet("background-color: rgba(153, 193, 241,0);")
        self.label_2.setObjectName("label_2")
        self.comboBox_dic = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_dic.setGeometry(QtCore.QRect(270, 130, 86, 25))
        self.comboBox_dic.setStyleSheet("background-color: rgb(153, 193, 241);")
        self.comboBox_dic.setObjectName("comboBox_dic")
        self.comboBox_dic.addItem("")
        self.comboBox_dic.addItem("")
        self.comboBox_dic.addItem("")
        self.camera.raise_()
        self.detection.raise_()
        self.Begin.raise_()
        self.Intro.raise_()
        self.Caption.raise_()
        self.comboBox.raise_()
        self.label_choose_model.raise_()
        self.Input.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.comboBox_dic.raise_()
        self.Pause.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1838, 28))
        self.menubar.setObjectName("menubar")
        self.menuPRML_2 = QtWidgets.QMenu(self.menubar)
        self.menuPRML_2.setObjectName("menuPRML_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuPRML_2.addSeparator()
        self.menubar.addAction(self.menuPRML_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)






        # designer基础上，为实现UI界面的功能，添加的代码
        # 连接                                                
        self.Begin.clicked.connect(self.begin_clicked)       
        self.Pause.clicked.connect(self.pause_clicked)
        # 设置图片大小，自适应                                    
        self.camera.setScaledContents(True)                  
        self.detection.setScaledContents(True)               
        
    # UI界面的代码
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.camera.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">camera</p></body></html>"))
        self.detection.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">detection</p></body></html>"))
        self.Begin.setText(_translate("MainWindow", "Begin"))
        self.Pause.setText(_translate("MainWindow", "Pause"))
        self.Intro.setText(_translate("MainWindow", "Work of Yifei Zhang, Jian Yu, Shing-Ho Lin"))
        self.Caption.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; color:#241f31;\">PRML-课程展示-多目标跟踪系统</span></p></body></html>"))
        self.comboBox.setItemText(0, _translate("MainWindow", "yolov8"))
        self.comboBox.setItemText(1, _translate("MainWindow", "yolox"))
        self.comboBox.setItemText(2, _translate("MainWindow", "ovd"))
        self.label_choose_model.setText(_translate("MainWindow", "检测模型选择"))
        self.label.setText(_translate("MainWindow", "跟踪目标输入"))
        self.label_2.setText(_translate("MainWindow", "词典选择"))
        self.comboBox_dic.setItemText(0, _translate("MainWindow", "coco"))
        self.comboBox_dic.setItemText(1, _translate("MainWindow", "lvis"))
        self.comboBox_dic.setItemText(2, _translate("MainWindow", "custom"))
        self.menuPRML_2.setTitle(_translate("MainWindow", "PRML"))




    # designer基础上，为实现UI界面的功能，添加的代码
    def begin_clicked(self):
        selected_detector = self.comboBox.currentText()
        selected_dic = self.comboBox_dic.currentText()
        thing_to_detect = self.Input.text()
        self.pause = False
        self.Camera_thread = CameraThread()  # 创建线程
        self.Detection_thread = DetectionThread(self.Camera_thread, selected_detector, selected_dic, thing_to_detect) # 创建线程，传入摄像头线程（因为不能同时打开摄像头）
        self.Camera_thread.changePixmap.connect(self.set_pixmap)
        self.Detection_thread.changeDetectionPixmap.connect(self.set_detection_pixmap)
        self.Camera_thread.start()
        self.Detection_thread.start()


    def set_pixmap(self, pixmap):
        self.camera.setPixmap(pixmap)
        self.camera.setScaledContents(True)


    def set_detection_pixmap(self, pixmap):
        self.detection.setPixmap(pixmap)
        self.detection.setScaledContents(True)

    def pause_clicked(self):
        self.pause = not self.pause

class CameraThread(QtCore.QThread):
    def __init__(self):
        super(CameraThread, self).__init__()
        self.ret = None
        self.frame = None

    changePixmap = QtCore.pyqtSignal(QtGui.QPixmap)


    def run(self):
        cap = cv2.VideoCapture(0)
        cap.open(0)
        while True:
            ret, frame = cap.read()
            if not ret:  # 如果没有读取到数据则跳过
                continue
            self.ret = ret
            self.frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转成rgb形式
            h, w, ch = rgb_frame.shape  # 获取scale
            bytes_per_line = ch * w

            # 转成QImage格式，在界面上显示
            qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.changePixmap.emit(pixmap)    

            if ui.pause:
                break

            # time.sleep(0.03)

        cap.release()

class DetectionThread(QtCore.QThread):
    def __init__(self, Camera_thread, selected_detector, selected_dic, thing_to_detect):
        super(DetectionThread, self).__init__()
        self.camera_thread = Camera_thread
        self.selected_detector = selected_detector
        self.selected_dic = selected_dic
        self.thing_to_detect = thing_to_detect


    changeDetectionPixmap = QtCore.pyqtSignal(QtGui.QPixmap)
    def run(self):
        dialog = QProgressDialog("正在加载中...", None, 0, 0)
        dialog.show()
        args = make_parser().parse_args()
        args.detector = self.selected_detector
        args.vocabulary = self.selected_dic
        args.thing_classes = self.thing_to_detect
        exp=Exp()
        predictor,current_time,args =main(exp,args)
        tracker,timer,frame_id,results =preimage_flow(args=args)
        
        while True:
            if self.camera_thread.ret is not None:
                detection_pixmap,tracker,timer,frame_id,results=imageflow_demo(predictor,tracker,timer,frame_id,results,self.camera_thread.ret,self.camera_thread.frame,args,exp)
                self.changeDetectionPixmap.emit(detection_pixmap)
                dialog.close()
            if ui.pause:
                break
            time.sleep(0.03)


def main(exp, args):
    torch.cuda.set_device('cuda:0')

    file_name = exp.output_dir
    os.makedirs(file_name, exist_ok=True)

    exp.vocabulary = args.vocabulary
    

    if args.vocabulary=='custom':
        if args.thing_classes is None:
            raise ValueError("Custom vocabulary must be specified by --thing_classes")
        else:
            exp.thing_classes.clear()
            thing_classes = args.thing_classes.split(',')
            for thing_class in thing_classes:
                exp.thing_classes.append(thing_class)


    
    args.device = "cuda"

    logger.info("Args: {}".format(args))

    
    exp.get_model_from_args(args)

    predictor = Predictor(None, exp)
    current_time = time.localtime()
    return predictor,  current_time, args
    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
