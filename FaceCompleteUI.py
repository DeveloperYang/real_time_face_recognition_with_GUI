# -*- coding: utf-8 -*-

# Copyright (c) 2019 XuYang
#
# E-mail : xuyangucas@163.com
#
# Form implementation generated from reading ui file 'FaceUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import os
import cv2
import sys
import threading
import tensorflow as tf
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import create_dataset
import face_recognition
from utils import image_processing, file_processing
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont,QImage,QPixmap,QPalette,QBrush,QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox


resize_height = 160
resize_width = 160
show_image_size = [782, 540]
dataset_path='dataset/emb/faceEmbedding.npy'
font_ttf_path = 'font_ttf/font_1.ttf'
model_path='models/20191209-113700'
filename='dataset/emb/name.txt'

class Ui_MainWindow(object):

    def __init__(self, MainWindow):
        super().__init__()

        self.flag_pause = False
        self.flag_stop = True
        self.flag_face_recognition = True
        self.flag_video_path_change = False
        self.flag_no_camera_tip = True
        self.default_face_show_time_index = 2
        self.video_path = 0
        self.face_recognition_dict_to_show = {}
        self.threshold = 0.7

        self.setupUi(MainWindow)




    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 710)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 782, 540))
        self.label.setObjectName("MainShow")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 570, 782, 116))
        self.label_2.setObjectName("FaceToShowNs")


        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(830, 23, 96, 96))
        self.label_3.setObjectName("Logo")

        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(830, 132, 116, 21))
        self.radioButton.setObjectName("LocalCam")
        self.radioButton.setFont(QFont("Times New Roman", 12))
        self.radioButton.toggled.connect(self.slots_target_radiobtnLocal)
        self.radioButton.setChecked(True)


        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(830, 165, 116, 21))
        self.radioButton_2.setObjectName("WebCam")
        self.radioButton_2.setFont(QFont("Times New Roman", 12))
        self.radioButton_2.toggled.connect(self.slots_target_radiobtnWebCam)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(830, 310, 100, 33))
        self.pushButton.setObjectName("Start")
        self.pushButton.setFont(QFont("Times New Roman", 14))
        self.pushButton.clicked.connect(self.slots_target_btnstate)
        self.pushButton.setCheckable(True)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(830, 363, 100, 33))
        self.pushButton_2.setObjectName("Pause")
        self.pushButton_2.setFont(QFont("Times New Roman", 14))
        self.pushButton_2.clicked.connect(self.slots_target_btn2state)
        self.pushButton_2.setCheckable(False)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(830, 417, 100, 33))
        self.pushButton_3.setObjectName("FaceRecognitionFlag")
        self.pushButton_3.setFont(QFont("Times New Roman", 14))
        self.pushButton_3.clicked.connect(self.slots_target_btn3state)
        self.pushButton_3.setCheckable(False)

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(830, 470, 100, 33))
        self.pushButton_4.setObjectName("UploadFaceDatabase")
        self.pushButton_4.setFont(QFont("Times New Roman", 14))
        self.pushButton_4.clicked.connect(self.slots_target_btn4state)
        self.pushButton_4.setCheckable(True)

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(830, 523, 100, 33))
        self.pushButton_5.setObjectName("updateFaceDatabase")
        self.pushButton_5.setFont(QFont("Times New Roman", 14))
        self.pushButton_5.clicked.connect(self.slots_target_btn5state)
        self.pushButton_5.setCheckable(True)

        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(830, 572, 100, 33))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("ThresholfTradeOff")
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(100)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setTickInterval(5)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider.setValue(50)
        self.horizontalSlider.sliderReleased.connect(self.slots_target_qslider)



        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(825, 605, 96, 20))
        self.label_4.setObjectName("HighPrecision")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(895, 605, 96, 20))
        self.label_5.setObjectName("HighRecall")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(818, 645, 130, 22))
        self.label_6.setObjectName("ShowTimeSetText")
        self.label_6.setFont(QFont("KaiTi", 11, 60))

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(886, 645, 40, 22))
        self.comboBox.setObjectName("SetShowTime")
        self.comboBox.addItems(['1','2','3','4','5','6','7','8','9','10','15','20','30','60'])
        self.comboBox.setCurrentIndex(self.default_face_show_time_index)
        self.comboBox.setFont(QFont('Times New Roman', 10))
        self.comboBox.activated.connect(self.slots_target_combox_set_face_show_time)

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(838, 192, 90, 26))
        self.lineEdit.setObjectName("WebCamUser")
        self.lineEdit.setText('admin')

        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(838, 225, 90, 26))
        self.lineEdit_2.setObjectName("WebCamPassword")
        self.lineEdit_2.setPlaceholderText("password")
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)

        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(838, 258, 90, 26))
        self.lineEdit_3.setObjectName("WebCamIP")
        self.lineEdit_3.setPlaceholderText('ip')

        self.label_show_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_1.setGeometry(QtCore.QRect(35, 580, 96, 96))
        self.label_show_1.setObjectName("label_show_1")

        self.label_show_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_2.setGeometry(QtCore.QRect(150, 580, 96, 96))
        self.label_show_2.setObjectName("label_show_2")

        self.label_show_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_3.setGeometry(QtCore.QRect(265, 580, 96, 96))
        self.label_show_3.setObjectName("label_show_3")

        self.label_show_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_4.setGeometry(QtCore.QRect(380, 580, 96, 96))
        self.label_show_4.setObjectName("label_show_4")

        self.label_show_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_5.setGeometry(QtCore.QRect(495, 580, 96, 96))
        self.label_show_5.setObjectName("label_show_5")

        self.label_show_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_show_6.setGeometry(QtCore.QRect(610, 580, 96, 96))
        self.label_show_6.setObjectName("label_show_6")

        self.list_label_show = [self.label_show_1,self.label_show_2,self.label_show_3,
                                self.label_show_4,self.label_show_5,self.label_show_6]

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 15))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FaceUI"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.radioButton.setText(_translate("MainWindow", "本地摄像头"))
        self.radioButton_2.setText(_translate("MainWindow", "网络摄像头"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_2.setText(_translate("MainWindow", "暂停"))
        self.pushButton_3.setText(_translate("MainWindow", "人脸识别"))
        self.pushButton_4.setText(_translate("MainWindow", "添加人脸库"))
        self.pushButton_5.setText(_translate("MainWindow", "更新人脸库"))
        self.label_4.setText(_translate("MainWindow", "精准率高"))
        self.label_5.setText(_translate("MainWindow", "召回率高"))
        self.label_6.setText(_translate("MainWindow", "结果展示      秒"))

        MainWindow.setWindowIcon(QIcon("UI_images/camera.jpg"))
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("UI_images/background.jpg")))
        MainWindow.setPalette(palette)

        init_image = cv2.imread("UI_images/face.jpg")
        init_image = cv2.cvtColor(init_image, cv2.COLOR_RGB2BGR)
        img = QtGui.QImage(init_image.data, init_image.shape[1], init_image.shape[0], QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(img.scaled(show_image_size[0], show_image_size[1])))

        init_image = cv2.imread("UI_images/muban3.jpg")
        init_image = cv2.cvtColor(init_image, cv2.COLOR_RGB2BGR)
        img = QtGui.QImage(init_image.data, init_image.shape[1], init_image.shape[0], QtGui.QImage.Format_RGB888)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(img.scaled(782, 170)))

        self.label_3.setText("人脸\n识别")
        self.label_3.setFont(QFont("Roman times", 20))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        pe = QtGui.QPalette()
        pe.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        self.label_3.setAutoFillBackground(True)
        pe.setColor(QtGui.QPalette.Window, QtCore.Qt.gray)
        self.label_3.setPalette(pe)

    def slots_target_btnstate(self):
        if self.pushButton.isChecked():
            print("pushButton")
            if self.flag_stop:
                print("Start")
                print(self.video_path)
                try:
                    int(self.video_path)
                    video_capture_tmp = cv2.VideoCapture(self.video_path)
                    if video_capture_tmp.read()[1] is None:
                        self.flag_stop = True
                        QMessageBox.information(None, "提示", "未能成功打开摄像头，请检查是否正常连接！", QMessageBox.Ok, QMessageBox.Ok)
                        self.pushButton.setChecked(False)
                    else:
                        self.flag_stop = False
                        self.pushButton.setCheckable(False)
                        self.pushButton_2.setCheckable(True)
                        self.pushButton_3.setCheckable(True)
                        self.Start()
                    video_capture_tmp.release()
                except:
                    self.flag_stop = False
                    self.pushButton.setCheckable(False)
                    self.pushButton_2.setCheckable(True)
                    self.pushButton_3.setCheckable(True)
                    self.Start()
            else:
                print("Pause")
                self.pushButton.setChecked(False)
                self.pushButton_2.setCheckable(True)
                self.flag_pause = False
    def slots_target_btn2state(self):
        if self.pushButton_2.isChecked():
            print("pushButton_2")
            self.pushButton_2.setCheckable(False)
            self.pushButton.setCheckable(True)
            self.flag_pause = True

    def slots_target_btn3state(self):
        if self.pushButton_3.isChecked():
            self.pushButton_3.setChecked(False)
            self.flag_face_recognition = not self.flag_face_recognition
            print("pushButton_3")
            print("flag_face_recognition", self.flag_face_recognition)

    def slots_target_btn4state(self):
        if self.pushButton_4.isChecked():
            print("pushButton_4")
            self.pushButton_4.setChecked(False)
            QFileDialog.getOpenFileName(None, 'Open file', './dataset/images')

    def slots_target_btn5state(self):
        if self.pushButton_5.isChecked():
            print("pushButton_5")
            self.pushButton_5.setChecked(False)
            select_result = QMessageBox.question(None, "提示", "确认更新人脸数据库？", QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)

            # 判断是否选择的Yes，确认更新
            if select_result == QMessageBox.Yes:
                # 利用结束while循环，结束进程
                self.flag_stop = True

                # 更新过程中，将主界面设为等待界面
                init_image = cv2.imread("UI_images/loading_pic.jpg")
                init_image = cv2.cvtColor(init_image, cv2.COLOR_RGB2BGR)
                img = QtGui.QImage(init_image.data, init_image.shape[1], init_image.shape[0],
                                   QtGui.QImage.Format_RGB888)
                self.label.setPixmap(QtGui.QPixmap.fromImage(img.scaled(show_image_size[0], show_image_size[1])))
                cv2.waitKey(1)

                # 对人脸进行编码
                create_dataset.create_dataset_face()

                # 更新人脸库终止了实时视频人脸识别进程，置位三个按键
                self.pushButton.setCheckable(True)
                self.pushButton_2.setCheckable(False)
                self.pushButton_3.setCheckable(False)

                # 更新完成后，回到主界面
                init_image = cv2.imread("UI_images/face.jpg")
                init_image = cv2.cvtColor(init_image, cv2.COLOR_RGB2BGR)
                img = QtGui.QImage(init_image.data, init_image.shape[1], init_image.shape[0],
                                   QtGui.QImage.Format_RGB888)
                self.label.setPixmap(QtGui.QPixmap.fromImage(img.scaled(show_image_size[0], show_image_size[1])))
                cv2.waitKey(1)

                # 更新完成，弹出提示框
                QMessageBox.information(None, "提示", "更新成功！", QMessageBox.Ok, QMessageBox.Ok)


    def slots_target_radiobtnLocal(self):
        if self.radioButton.isChecked():
            self.flag_video_path_change = True
            print("radiobtnLocal")
            self.video_path = 0

    def slots_target_radiobtnWebCam(self):
        if self.radioButton_2.isChecked():
            print("radioButton_2")
            self.flag_video_path_change = True
            WebCam_path = 'rtsp://' + self.lineEdit.text() + ':' + self.lineEdit_2.text() + '@' + self.lineEdit_3.text()
            if len(WebCam_path) > 30:
                self.video_path = WebCam_path
            else:
                QMessageBox.information(None, "提示", "请正确输入网络摄像头相关信息 ( admin, password, ip ) ！", QMessageBox.Ok,QMessageBox.Ok)
                self.video_path = 0
                self.radioButton.setChecked(True)
                self.radioButton_2.setChecked(False)

    def slots_target_qslider(self):
        self.threshold = 0.5 + 0.004 * self.horizontalSlider.value()

    def slots_target_combox_set_face_show_time(self):
        if self.default_face_show_time_index != self.comboBox.currentIndex():
            select_result = QMessageBox.question(None, "提示", "确认修改展示时间？", QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if select_result == QMessageBox.Yes:
                self.default_face_show_time_index = self.comboBox.currentIndex()
            if select_result == QMessageBox.Cancel:
                self.comboBox.setCurrentIndex(self.default_face_show_time_index)
            print("face_show_time is {} s. ".format(self.comboBox.currentText()))


    def Start(self):
        self.gif = QtGui.QMovie('UI_images/loading.gif')
        self.gif.setScaledSize(QtCore.QSize(show_image_size[0], show_image_size[1]))
        self.label.setMovie(self.gif)
        self.gif.start()
        th = threading.Thread(
            target=self.real_time_display_video_face_recognition,
            args=(dataset_path, filename, model_path, show_image_size, 2, self.label))
        th.start()

    def load_dataset(self, dataset_path, filename):
        '''
        加载人脸数据库
        :param dataset_path: embedding.npy文件（faceEmbedding.npy）
        :param filename: labels文件路径路径（name.txt）
        :return:
        '''
        embeddings = np.load(dataset_path)
        names_list = file_processing.read_data(filename, split=None, convertNum=False)
        return embeddings, names_list

    def cv2_img_add_text(self, img, text, left, top, textColor=(0, 255, 0), textSize=35):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(font_ttf_path, textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


    def face_detect_recognition(self, input_image, face_detect_model, face_net_model, dataset_emb, names_list, resize_height,
                          resize_width, flag_face_recognition=True):
        bboxes, landmarks = face_detect_model.detect_face(input_image)
        bboxes, landmarks = face_detect_model.get_square_bboxes(bboxes, landmarks, fixed="height")
        if bboxes == [] or landmarks == [] or not flag_face_recognition:
            rgb_image = input_image
        else:
            #print("-----image have {} faces".format(len(bboxes)))
            face_images = image_processing.get_bboxes_image(input_image, bboxes, resize_height, resize_width)
            face_images = image_processing.get_prewhiten_images(face_images)
            pred_emb = face_net_model.get_embedding(face_images)
            pred_name, pred_score,face_to_show_name = face_net_model.compare_embadding(pred_emb, dataset_emb, names_list, threshold=self.threshold)
            for name in face_to_show_name:
                self.face_recognition_dict_to_show[name] = time.time()
            # show_info = [n + ':' + str(s)[:5] for n, s in zip(pred_name, pred_score)]
            show_info = pred_name
            bgr_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            for name, box in zip(show_info, bboxes):
                box = [int(b) for b in box]
                cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5, 8, 0)
                bgr_image = self.cv2_img_add_text(bgr_image, name, box[0]-4, box[1]-30, textColor=(0, 0, 255), textSize=22)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def show_face_list_n_s(self, show_list, label_list):
        show_list_count_org = len(show_list)
        show_list_finish = []
        for i in range(show_list_count_org):
            show_img_path = 'dataset/images/' + show_list[i] + '/'
            if os.path.exists(show_img_path):
                show_list_finish.append(show_list[i])
        show_list_count = min(len(show_list_finish), 6)
        for i in range(show_list_count):
            show_img_path = 'dataset/images/' + show_list_finish[i] + '/'
            list_file = os.listdir(show_img_path)
            show_img_path = show_img_path + list_file[0]
            init_image = image_processing.read_image_gbk(show_img_path, 96, 96)
            img = QImage(init_image.data, init_image.shape[1], init_image.shape[0], QImage.Format_RGB888)
            label_list[i].setPixmap(QPixmap.fromImage(img.scaled(96, 96)))
        for i in range(show_list_count,6):
            label_list[i].setText(' ')

    def real_time_display_video_face_recognition(self, dataset_path, filename, model_path, show_image_size, capture_interval=2,
                                       PyQT_label=False):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with sess.as_default():
                video_capture = cv2.VideoCapture(self.video_path)
                video_capture.set(3, 960)
                video_capture.set(4, 720)
                capture_count = 0


                # 初始化mtcnn人脸检测
                face_detect = face_recognition.faceDetection()
                # 加载数据库的数据
                dataset_emb, names_list = self.load_dataset(dataset_path, filename)
                # 初始化facenet
                face_net = face_recognition.facenetEmbedding(model_path)



                # 确保加载过程中点击暂停键，而造成的实时识别无法正常运行
                self.flag_pause = False

                while True:
                    if self.flag_pause:
                        continue
                    if self.flag_stop:
                        break
                    ret, frame = video_capture.read()
                    if self.flag_video_path_change:
                        self.flag_video_path_change = False
                        video_capture = cv2.VideoCapture(self.video_path)
                        video_capture.set(3, 960)
                        video_capture.set(4, 720)
                        ret, frame = video_capture.read()
                    # 确保丢帧时报错
                    if frame is None:
                        video_capture = cv2.VideoCapture(self.video_path)
                        print("-----can't open camera: " + str(self.video_path))

                        ############### 提示操作者，摄像头没能正常启动 ##########################
                        self.label.setText("摄像头数据丢失，请检查摄像头是否正常连接！")
                        self.label.setFont(QFont("Roman times", 20))
                        self.label.setAlignment(QtCore.Qt.AlignCenter)
                        pe = QtGui.QPalette()
                        pe.setColor(QtGui.QPalette.WindowText, QtCore.Qt.red)
                        self.label.setAutoFillBackground(True)
                        pe.setColor(QtGui.QPalette.Window, QtCore.Qt.gray)
                        self.label.setPalette(pe)
                        ###########################################################################

                        continue

                    if (capture_count % capture_interval == 0):
                        rgb_image = self.face_detect_recognition(frame, face_detect, face_net, dataset_emb,
                                                           names_list, resize_height, resize_width,
                                                           self.flag_face_recognition)
                        label_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    capture_count += 1
                    if PyQT_label:
                        img = QImage(label_image.data, label_image.shape[1], label_image.shape[0], QImage.Format_RGB888)
                        PyQT_label.setPixmap(QPixmap.fromImage(img.scaled(show_image_size[0], show_image_size[1])))
                        cv2.waitKey(1)
                    tmp_record_face_list = []
                    for key in self.face_recognition_dict_to_show.keys():
                        if time.time() - self.face_recognition_dict_to_show.get(key) > int(self.comboBox.currentText()):
                            tmp_record_face_list.append(key)
                    for key in tmp_record_face_list:
                        self.face_recognition_dict_to_show.pop(key)
                    show_list = list(self.face_recognition_dict_to_show.keys())
                    self.show_face_list_n_s(show_list, self.list_label_show)
                video_capture.release()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(widget)
    widget.show()
    sys.exit(app.exec_())



