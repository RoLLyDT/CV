import tracking
import time
import os
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Person Tracking")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(230, 80, 341, 31))
        self.comboBox.setObjectName("comboBox")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(230, 300, 341, 101))
        self.pushButton.setObjectName("pushButton")

        self.playButton = QtWidgets.QPushButton(self.centralwidget)
        self.playButton.setGeometry(QtCore.QRect(230, 450, 341, 101))
        self.playButton.setObjectName("playButton")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 40, 341, 31))
        self.label.setObjectName("label")

        self.label_model = QtWidgets.QLabel(self.centralwidget)
        self.label_model.setGeometry(QtCore.QRect(230, 120, 341, 31))
        self.label_model.setObjectName("label_model")

        self.label_video_size = QtWidgets.QLabel(self.centralwidget)
        self.label_video_size.setGeometry(QtCore.QRect(230, 200, 341, 31))
        self.label_video_size.setObjectName("label_video_size")

        self.comboBox_model = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_model.setGeometry(QtCore.QRect(230, 160, 341, 31))
        self.comboBox_model.setObjectName("comboBox_model")

        model_names = [
            "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
            "yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6"
        ]
        for model_name in model_names:
            self.comboBox_model.addItem(model_name)


        self.comboBox_video = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_video.setGeometry(QtCore.QRect(230, 240, 341, 31))
        self.comboBox_video.setObjectName("comboBox_video")

        video_sizes = [
            "1280 x 720", "640 x 480", "320 x 180", "868 x 876",
            "1920 x 1080", "1440 x 1080", "2048 x 2048", "3840 x 2160"
        ]
        for video_size in video_sizes:
            self.comboBox_video.addItem(video_size)

        self.checkBox_apply_mask = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_apply_mask.setGeometry(QtCore.QRect(230, 410, 341, 31))
        self.checkBox_apply_mask.setObjectName("checkBox_apply_mask")

        fontQ = QtGui.QFont()
        fontQ.setFamily("Rockwell")
        fontQ.setPointSize(14)
        self.checkBox_apply_mask.setFont(fontQ)

        self.checkBox_apply_mask.setText("Apply mask to recognised objects")

        self.checkBox_apply_mask.stateChanged.connect(self.applyMaskStateChanged)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.populateComboBox()
        self.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.playButton.clicked.connect(self.on_playButton_clicked)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Person Tracking", "Person Tracking"))
        self.pushButton.setText(_translate("Person Tracking", "Generate Video"))
        self.playButton.setText(_translate("Person Tracking", "Play Video"))
        self.label.setText(_translate("Person Tracking", "Choose a file:"))
        self.label_model.setText(_translate("Person Tracking", "YOLOV5 Model:"))
        self.label_video_size.setText(_translate("Person Tracking", "Video Size for Playing:"))


        # Set font for labels
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(18)
        self.label.setFont(font)
        self.label_model.setFont(font)
        self.label_video_size.setFont(font)
        font.setPointSize(24)
        self.pushButton.setFont(font)
        self.playButton.setFont(font)

    def populateComboBox(self):
        videos_folder = "videos"
        video_files = os.listdir(videos_folder)
        video_files = [file for file in video_files if file.endswith(".mp4")]
        for file in video_files:
            file_name = file.replace("videos/", "")
            self.comboBox.addItem(file_name)

    def applyMaskStateChanged(self, state):
        if state == QtCore.Qt.Checked:
            print("Apply mask checked")
        else:
            print("Apply mask unchecked")
    def on_pushButton_clicked(self):
        print('Button Generate Clicked')
        selected_file = self.comboBox.currentText()  # Get the selected file from combo box
        selected_model = self.comboBox_model.currentText()
        apply_mask = self.checkBox_apply_mask.isChecked()
        path = f'videos/{selected_file}'

        msg_proc = QMessageBox()
        msg_proc.setWindowTitle("Generation Started")
        msg_proc.setText("You will need to wait some time for the video generation to be completed."
                         "\nClose this window to start the generation."
                         "\nAnother message will be shown when the generation is completed.")
        msg_proc.setIcon(QMessageBox.Information)
        msg_proc.exec_()

        tracking.track(path, selected_model, apply_mask)

        msg = QMessageBox()
        msg.setWindowTitle("Generation Completed")
        msg.setText("Video generation completed successfully."
                    "\nIf you want to play the video, click on the Play Video button."
                    "\nGenerated video also saved in Output folder.")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def on_playButton_clicked(self):
        print('Button Play Clicked')
        video_path = 'Output/output_video.mp4'  # Path to the generated video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            msg_err = QMessageBox()
            msg_err.setWindowTitle("ERROR")
            msg_err.setText("Error: Could not find the video.")
            msg_err.setIcon(QMessageBox.Information)
            msg_err.exec_()
            return

        selected_video_size = self.comboBox_video.currentText()
        width, height = map(int, selected_video_size.split(' x '))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f'Press Q to exit', (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            frame = cv2.resize(frame, (width, height))
            cv2.imshow('Final Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
