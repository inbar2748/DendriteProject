

#
#  Copyright (c) 2019  INBAR DAHARI.
#  All rights reserved.
#

import PyQt5
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog


class GUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(GUI,self).__init__()
        uic.loadUi("mainwindow.ui", self)
        self.uploadFile.clicked.connect(self.upload_file)
        self.NextButton.clicked.connect(self.Next_Button)

    def upload_file(self):
        f_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open Png File", "", "Files (*.png)")
        if not f_name:
            return
        self.UploadFileName.setText(f_name)
        # self.previewPhoto.setImage(f_name)

    def Next_Button(self):
        self.tab_2.show()






if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    QtCore.QCoreApplication.processEvents()
    mapp = GUI()
    mapp.show()
    sys.exit(app.exec_())
