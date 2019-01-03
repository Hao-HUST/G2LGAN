import numpy as np
import vtk
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class GUIViewer(QVTKRenderWindowInteractor):

    def __init__(self, parent, engine,camerax,cameray,cameraz):
        QVTKRenderWindowInteractor.__init__(self, parent)
        self.engine = engine
        self.resetCamera = True
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.GetRenderWindow().GetInteractor()
        self.create_actor()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(255,255,255)
        camera = self.renderer.GetActiveCamera()
#        camera.SetViewUp( 0.2,0.0983, 0.9345)
        camera.SetViewUp( 0.5,0, 0)
        camera.SetPosition(0.1245,0.1139, 0.2932)
        self.renderer.SetActiveCamera(camera)
        
    def create_voxel(self):
        numberOfVertices = 8

        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(1, 0, 0)
        points.InsertNextPoint(0, 1, 0)
        points.InsertNextPoint(1, 1, 0)
        points.InsertNextPoint(0, 0, 1)
        points.InsertNextPoint(1, 0, 1)
        points.InsertNextPoint(0, 1, 1)
        points.InsertNextPoint(1, 1, 1)

        voxel = vtk.vtkVoxel()
        for i in range(0, numberOfVertices):
            voxel.GetPointIds().SetId(i, i)

        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.InsertNextCell(voxel.GetCellType(), voxel.GetPointIds())

        gfilter = vtk.vtkGeometryFilter()
        gfilter.SetInput(ugrid)
        gfilter.Update()
        return gfilter

    def create_actor(self):
        self.points = vtk.vtkPoints()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetName("colors")
        self.colors.SetNumberOfComponents(4)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self.points)
        polydata.GetPointData().SetScalars(self.colors)

        # create cell
        voxel = self.create_voxel()

        self.glyph3D = vtk.vtkGlyph3D()
        self.glyph3D.SetColorModeToColorByScalar()
        self.glyph3D.SetSource(voxel.GetOutput())
        self.glyph3D.SetInput(polydata)
        self.glyph3D.ScalingOff()
        self.glyph3D.Update()

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(self.glyph3D.GetOutput())

        # actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty()
#        .SetAmbient(0.15)

    def update_actor(self):
        data = self.engine.model
        self.points.Reset()
        self.colors.Reset()
        channel = data.shape[3]
        n=0       
        for i in range(len(data[0])):
            for j in range(len(data[1])):
                for k in range(len(data[2])):
                    if channel == 1:
                       dx = (data[i,j,k,:]+0.5).astype(int); 
                       #idx = ((data[i,j,k,:]+1.)*(4.99/2)).astype('int32');
                    else:
                        ceil = data[i,j,k,:]
                        idx = np.argmax(ceil)
                        #idx = (data[i,j,k,4]+0.5).astype(int); 
                    
                    if idx==1:
                        self.points.InsertNextPoint(i, j, k)
                        self.colors.InsertTuple4(n, 255, 0, 0, 192) 
                        n=n+1 
                    
                    if idx==2:
                        self.points.InsertNextPoint(i, j, k)
                        self.colors.InsertTuple4(n, 0, 255, 0, 192) 
                        n=n+1 
                    if idx==3:
                        self.points.InsertNextPoint(i, j, k)
                        self.colors.InsertTuple4(n, 0, 0, 255, 192) 
                        n=n+1 
                    
                    if idx==4:
                        self.points.InsertNextPoint(i, j, k)
                        self.colors.InsertTuple4(n, 255, 255, 0, 192) 
                        n=n+1  
                    
                    
        self.glyph3D.Modified()
        if self.resetCamera:
            self.renderer.ResetCamera()
            
            self.resetCamera = False
        self.update()
        QApplication.processEvents()
        
        
    def save_image2(self):  
        file_path = QFileDialog.getSaveFileName(self,"save file","." ,"png file(image_"+str(self.engine.index)+".png)") 
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.GetRenderWindow())
        window_to_image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(unicode(file_path))
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()
        
    def save_image1(self,file_path):  
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.GetRenderWindow())
        window_to_image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(unicode(file_path)+'/image_'+str(self.engine.index)+'.png')
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()    

    def set_camera(self,x,y,z):
        camera = self.renderer.GetActiveCamera()
        camera.SetViewUp(x,y,z)
        self.renderer.SetActiveCamera(camera)
