import numpy as np
import time
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class ConstrainedOpt(QThread):

    def __init__(self, model,index):
        QThread.__init__(self)
        self.model = model
        self.index = index

    def run(self):
#        while True:
            self.update_voxel_model()
#            self.msleep(10)

    def update_voxel_model(self):
        self.emit(SIGNAL('update_voxels'))