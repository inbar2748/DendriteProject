#
#  Copyright (c) 2019  INBAR DAHARI.
#  All rights reserved.
#
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import sys
from PyQt5.QtWidgets import QFileDialog

from main_solution_update import Interface




class GUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(GUI, self).__init__()
        self.ui = uic.loadUi("mainwindow.ui", self)
        self.ui.uploadFile.clicked.connect(self.upload_file)
        self.ui.NextButton.clicked.connect(self.Next_Button)
        self.ui.threshold1_slider_2.valueChanged.connect(self.on_slider_value_changed)
        self.ui.min_distance.valueChanged.connect(self.on_min_distance_changed)
        self.ui.min_angel.valueChanged.connect(self.on_min_angel_changed)
        self.ui.btn_ok.clicked.connect(self.on_ok_clicked)
        self.ui.btn_create_preview.clicked.connect(self.on_preview_clicked)

        self.interface = Interface()
        self.ui.tabWidget.setCurrentIndex(0)



    def upload_file(self):
        f_name = QFileDialog.getOpenFileName(self, "Open Png File", "", "Files (*.png)")[0]
        if not f_name:
            return
        self.ui.UploadFileName.setText(f_name)
        # self.previewPhoto.setImage(f_name)
        self.interface.p_file_path = f_name

    def Next_Button(self):
        print("Next clicked")
        self.ui.tabWidget.setCurrentIndex(1)

    def on_slider_value_changed(self, value):
        print("Slider changed", value)
        self.interface.p_threshold1 = value

    def on_min_distance_changed(self, value):
        print("Min distance", value)
        self.interface.min_distance_to_merge = value

    def on_min_angel_changed(self, value):
        print("Min angele", value)
        self.interface.min_angle_to_merge = value

    def on_ok_clicked(self):
        print("Ok clicekd")
        f_name = QFileDialog.getSaveFileName(self, "Save Excel file", "", "Excel Files (*.xlsx)")[0]
        if not f_name:
            f_name = 'data_information.xlsx'
        self.interface.excel_file = f_name
        print("Saving excel file to", f_name)
        self.interface.main()

    def on_preview_clicked(self):
        print("preview clicked")
        self.interface.create_preview()




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    QtCore.QCoreApplication.processEvents()
    mapp = GUI()
    mapp.show()
    sys.exit(app.exec_())
