import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from gui_viewer import GUIViewer
from opt import ConstrainedOpt

class MainWindow(QMainWindow):
    signal_save_images = pyqtSignal(str)
    signal_setCamera = pyqtSignal(float,float,float)
    def __init__(self,width,height,parent=None):
        QMainWindow.__init__(self, parent)
        self.width = width
        self.height = height
        self.resize(width,height)
        self.mdi_Area = QMdiArea()
        self.mdi_Area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi_Area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.left_widget = QWidget()
        self.left_widget.setFixedSize(100,self.height)
        self.right_widget = QWidget()
        self.widget = QWidget()
        viewerBox = QHBoxLayout()
        viewerBox.addWidget(self.left_widget)
        viewerBox.addWidget(self.mdi_Area)
        self.widget.setLayout(viewerBox)
        self.setCentralWidget(self.widget)
        
        self.camerax = 0
        self.cameray = 0
        self.cameraz = 0

#        labelcamera = QLabel()
#        labelcamera.setText("set camera")
#        
#        labelx = QLabel()
#        labelx.setText("x")
#        self.valuex = QLineEdit()
#        camerax = QHBoxLayout()
#        camerax.addWidget(labelx)
#        camerax.addWidget(self.valuex)
#        
#        labely = QLabel()
#        labely.setText("y")
#        self.valuey = QLineEdit()
#        cameray = QHBoxLayout()
#        cameray.addWidget(labely)
#        cameray.addWidget(self.valuey)
#        
#        labelz = QLabel()
#        labelz.setText("z")
#        self.valuez = QLineEdit()
#        cameraz = QHBoxLayout()
#        cameraz.addWidget(labelz)
#        cameraz.addWidget(self.valuez)
#        
#        lvBox = QVBoxLayout()
#        lvBox.addWidget(labelcamera)
#        lvBox.addLayout(camerax)
#        lvBox.addLayout(cameray)
#        lvBox.addLayout(cameraz)
#        lvBox.addStretch()
#        lvBox.setSpacing(1)
#        self.connect(self.valuex, SIGNAL("returnPressed()"), self.setCamera)
#        
#        self.left_widget.setLayout(lvBox)
        
        self.setCentralWidget(self.mdi_Area)

        self.mdi_Win= list()
        self.frame = list()
        self.viewerWidget = list()
        
        
        # create openAction
        openAction = QAction("&Open", self)
        openAction.setShortcut(QKeySequence.Open)
        openAction.setToolTip("Open a file")
        openAction.setStatusTip("Open a file")
        self.connect(openAction, SIGNAL("triggered()"), self.open_file)
        
        preAction = QAction("&PreGroup", self)
        preAction.setShortcut(QKeySequence.Open)
        preAction.setToolTip("Show pre group model")
        preAction.setStatusTip("Show pre group model")
        self.connect(preAction, SIGNAL("triggered()"), self.pre_group)
        
        nextAction = QAction("&NextGroup", self)
        nextAction.setShortcut(QKeySequence.Open)
        nextAction.setToolTip("Show next group model")
        nextAction.setStatusTip("Show next group model")
        self.connect(nextAction, SIGNAL("triggered()"), self.next_group)
        
        saveAction = QAction("&SaveAll", self)
        saveAction.setShortcut(QKeySequence.Open)
        saveAction.setToolTip("Save all models as Image")
        saveAction.setStatusTip("Save all models as Image")
        self.connect(saveAction, SIGNAL("triggered()"), self.save_group)
        
        saveOneAction = QAction("&SaveCurrent", self)
        saveOneAction.setShortcut(QKeySequence.Open)
        saveOneAction.setToolTip("Save current model")
        saveOneAction.setStatusTip("Save current model")
        self.connect(saveOneAction, SIGNAL("triggered()"), self.save_one)

        # create toolbar
        toolbar = self.addToolBar("tool")
        toolbar.setMovable(False)
        toolbar.setObjectName("ToolBar")
        toolbar.addAction(openAction)
        toolbar.addAction(preAction)
        toolbar.addAction(nextAction)
        toolbar.addAction(saveAction)
        toolbar.addAction(saveOneAction)

    def closeEvent(self, event):
        for i in range(self.mdi_Win):
            self.mdi_Win[i].opt_engine.quit()

    def open_file(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file','.',"all files (*)")  
        print 'open file:',file_path
        model = np.load(unicode(file_path))          
        self.opt_engine = model
        self.models_count = model.shape[0]
        self.current_index = 0
        self.banch = 8
        self.view_model()

    def view_model(self):
        start = self.current_index
        banch = self.banch
        end = start+banch
        if end>self.models_count:
            end = self.models_count
        i=0
        width = (self.width*2/banch)*0.95
        height = (self.height/2)*0.95
        mainWidth = width + 10
        mainHeight = height + 10
        self.setWindowTitle("model_view      models_count:"+str(self.models_count)+"    current:("+str(start)+"~"+str(end-1)+")")    
        for index in range(start,end):
            model = (self.opt_engine[index,:,:,:,:])
             
            self.frame.append(QFrame())
            self.mdi_Win.append(QMdiSubWindow())
            self.mdi_Win[i].opt_engine = ConstrainedOpt(model,index)
            self.mdi_Win[i].setWindowTitle("model_"+str(index))
            self.mdi_Win[i].setGeometry(0, 0, mainWidth, mainHeight)
            
            self.viewerWidget.append(GUIViewer(self.frame[i], self.mdi_Win[i].opt_engine,self.camerax,self.cameray,self.cameraz))
            self.viewerWidget[i].resize(width,height)

            viewerBox = QVBoxLayout()
            viewerBox.addWidget(self.viewerWidget[i])
            self.frame[i].setLayout(viewerBox)
            self.mdi_Win[i].setWidget(self.frame[i])

#            saveAction = QAction("&Save", self)
#            saveAction.setShortcut(QKeySequence.Open)
#            saveAction.setToolTip("Save current image")
#            saveAction.setStatusTip("Save current image")
#            self.mdi_Win[i].connect(saveAction, SIGNAL("triggered()"), self.viewerWidget[i].save_image)
#    
#            # create toolbar
#            toolbar = self.mdi_Win[i].addToolBar("tool")
#            toolbar.setMovable(False)
#            toolbar.setObjectName("ToolBar")
#            toolbar.addAction(saveAction)

            self.viewerWidget[i].interactor.Initialize()
            self.connect(self.mdi_Win[i].opt_engine, SIGNAL('update_voxels'), self.viewerWidget[i].update_actor)
            self.connect(self.mdi_Win[i], SIGNAL('save_image'), self.viewerWidget[i].save_image2)
            self.signal_save_images.connect(self.viewerWidget[i].save_image1)
            self.signal_setCamera.connect(self.viewerWidget[i].set_camera)
            
            self.mdi_Win[i].opt_engine.start()
            self.mdi_Area.addSubWindow(self.mdi_Win[i])
            self.mdi_Win[i].show()
            i = i+1
          
    def pre_group(self):
         self.current_index = self.current_index -self.banch
         if self.current_index<0:
             self.current_index = 0
         for i in range(len(self.mdi_Win)):
             self.mdi_Win[i].close()
         self.mdi_Win[:]=[]
         self.frame[:]=[]
         self.viewerWidget[:]=[]
         self.view_model()
        
    def next_group(self):
         self.current_index = self.current_index +self.banch
         
         for i in range(len(self.mdi_Win)):
             self.mdi_Win[i].close()
         self.mdi_Win[:]=[]
         self.frame[:]=[]
         self.viewerWidget[:]=[]
         self.view_model()
         
    def save_group(self):
#        file_path = QFileDialog.getSaveFileName(self,"save file","C:\Users\Administrator\Desktop" ,"png file(*.png)")                 
        file_path = QFileDialog.getExistingDirectory(self,"Open a folder",".",QFileDialog.ShowDirsOnly)
        self.signal_save_images.emit(file_path)
    def save_one(self):
#        file_path = QFileDialog.getSaveFileName(self,"save file","." ,"png file(*.png)") 
#        file_path = QFileDialog.getExistingDirectory(self,"Open a folder",".",QFileDialog.ShowDirsOnly)
#        print 'sub1',self.mdi_Area.currentSubWindow()==self.mdi_Win[0]
#        self.mdi_Area.currentSubWindow().signal_save_image.emit(file_path)
        self.mdi_Area.currentSubWindow().emit(SIGNAL('save_image'))
    
    def setCamera(self):
        self.camerax = float(unicode(self.valuex.text()))
        self.cameray = float(unicode(self.valuey.text()))
        self.cameraz = float(unicode(self.valuez.text()))
        valuex = self.camerax 
        valuey = self.cameray 
        valuez = self.cameraz 
        self.signal_setCamera.emit(valuex,valuey,valuez)
        