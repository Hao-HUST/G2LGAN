import sys
import qdarkstyle
from PyQt4.QtGui import QApplication
from ui import MainWindow

if __name__ == "__main__":

    app = QApplication(sys.argv)
    desktopWidget = QApplication.desktop()
    screenRect =desktopWidget.screenGeometry()
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))
    window = MainWindow(screenRect.width(),screenRect.height())
    window.setWindowTitle("model_view")
    window.show()
    sys.exit(app.exec_())
