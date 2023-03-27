from PyQt5 import QtCore
from PyQt5 import QtWidgets
import sys

from view.SOA import GUI

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    QtCore.QCoreApplication.processEvents()
    mapp = GUI()
    mapp.show()
    sys.exit(app.exec_())
