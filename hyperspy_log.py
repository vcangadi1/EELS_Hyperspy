im=signals.Image(np.random.random((64,64)))
im=signals.Image(np.random.random((64,64)))
preferences.gui()
s=signals.Spectrum(np.random.random((10,20,100)))
s
s.metadata
im=s.to_image()
im
im.metadata
s.set_signal_type("EELS")
s.metadata
s.set_signal_origin="Simulations"
s.metadata
s.Signals.set_signal_origin="Simulations"
s.metadata
s.Signal.signal_type
s
s.set_signal_origin("simulation")
s.metadata
s.set_signal_origin("simulation")
s.set_signal_type("EELS")
s.set_signal_origin("simulation")
s.metadata.Signal
s.metadata.Signal.signal_origin="simulation"
s.metadata
s=signals.Spectrum(np.arange(10))
s
s.data
s.0
s[0]
s[0].data
s[5::-1]
s[5::-1].data
s[-1]
s[-1].data
s[:-1]
s[:-1].data
s[10::-1].data
s[9::-1].data
s.axes_manager[0]
s.axes_manager[0].scale=0.5
s
s.axes_manager[0].axis
s.data
s = signals.Spectrum(np.arange(2*3*4).reshape((2,3,4)))
s
s.data
s.axes_manager[0]="x"
s.axes_manager[0]='x'
s.axes_manager[0].name='x'
s.axes_manager[1].name='y'
s.axes_manager[2].name='z'
s.axes_manager[2].name='t'
s.axes_manager.signal_axes
s.axes_manager.navigation_axes
s.inav[0,0].data
s.inav[2,1].data
image_stack=signals.Image(np.random.random((2,5,64,64)))
image_stack
for single_image in image_stack:
for single_image in image_stack:
    single_image.save("image %s.png" % str(image_stack.axes_manager.indices))
import hyperspy.Release
print hyperspy.Release.info
image_stack=signals.Image(np.random.random((2,5,64,64)))
image_stack
for single_image in image_stack:
    single_image.save("image %s.png" % str(image_stack.axes_manager.indices))
import scipy.ndimage
image_stack=signals.Image(np.array([scipy.misc.lena]*5))
image_stack=signals.Image(np.array([scipy.misc.lena()]*5))
image_stack.axes_manager[1].name="x"
image_stack.axes_manager[2].name="y"
for image, angle in zip(image_stack, [0,45,90,135,180])
for image, angle in zip(image_stack, (0,45,90,135,180)
for image, angle in zip(image_stack, (0,45,90,135,180))
for image, angle in zip(image_stack, (0,45,90,135,180))
for image, angle in zip(image_stack, (0,45,90,135,180)):
    image.data=scipy.ndimage.rotate(image.data, angle=angle, reshape=False)
    
for image, angle in zip(image_stack, (0,45,90,135,180)):
    image.data=scipy.ndimage.rotate(image.data, angle=angle, reshape=False)
collage=utils.stack([image for image in image_stack], axis=0)
collage.plot()
for image, angle in zip(image_stack, (0,45,90,135,180)):
    image.data[:]=scipy.ndimage.rotate(image.data, angle=angle, reshape=False)
collage=utils.stack([image for image in image_stack], axis=0)
collage.plot()
collage=utils.stack([image for image in image_stack], axis=1)
collage.plot()
collage=utils.stack([image for image in image_stack], axis=2)
collage=utils.stack([image for image in image_stack], axis=0)
collage.plot()
for image, angle in zip(image_stack, (0,45,90,135,180)):
    image.data[:]=scipy.ndimage.rotate(image.data, angle=angle, reshape=True)
for image, angle in zip(image_stack, (0,45,90,135,180)):
    image.data[:]=scipy.ndimage.rotate(image.data, angle=angle, reshape=False)
collage=utils.stack([image for image in image_stack], axis=0)
collage.plot()
import scipy.ndimage
image_stack = signals.Image(np.array([scipy.misc.lena()]*4))
image_stack.axes_manager[1].name = "x"
image_stack.axes_manager[2].name = "y"
angles = signals.Signal(np.array([0, 45, 90, 135]))
angles.axes_manager.set_signal_dimension(0)
modes = signals.Signal(np.array(['constant', 'nearest', 'reflect', 'wrap']))
modes.axes_manager.set_signal_dimension(0)
image_stack.map(scipy.ndimage.rotate,
                           angle=angles,
                           reshape=False,
                           mode=modes)
import scipy.ndimage
image_stack = signals.Image(np.array([scipy.misc.lena()]*4))
image_stack.axes_manager[1].name = "x"
image_stack.axes_manager[2].name = "y"
angles = signals.Signal(np.array([0, 45, 90, 135]))
angles.axes_manager.set_signal_dimension(0)
modes = signals.Signal(np.array(['constant', 'nearest', 'reflect', 'wrap']))
modes.axes_manager.set_signal_dimension(0)
image_stack.map(scipy.ndimage.rotate,
                           angle=angles,
                           reshape=False,
                           mode=modes)
import scipy.ndimage
image_stack = signals.Image(np.array([scipy.misc.lena()]*4))
image_stack.axes_manager[1].name = "x"
image_stack.axes_manager[2].name = "y"
angles = signals.Signal(np.array([0, 45, 90, 135]))
angles.axes_manager.set_signal_dimension(0)
modes = signals.Signal(np.array(['constant', 'nearest', 'reflect', 'wrap']))
modes.axes_manager.set_signal_dimension(0)
image_stack.map(scipy.ndimage.rotate,
                           angle=angles,
                           reshape=False,
                           mode=modes)
import hyperspy.Release
print hyperspy.Release.info
import scipy.ndimage
image_stack = signals.Image(np.array([scipy.misc.lena()]*4))
image_stack.axes_manager[1].name = "x"
image_stack.axes_manager[2].name = "y"
angles = signals.Signal(np.array([0, 45, 90, 135]))
angles.axes_manager.set_signal_dimension(0)
modes = signals.Signal(np.array(['constant', 'nearest', 'reflect', 'wrap']))
modes.axes_manager.set_signal_dimension(0)
image_stack.map(scipy.ndimage.rotate,
                           angle=angles,
                           reshape=False,
                           mode=modes)
image=signals.Image(scipy.misc.lena())
image=utils.stack([utils.stack([image]*3),axis=0],axis=1)
image=utils.stack([utils.stack([image]*3),axis=0]*3,axis=1)
image=signals.Image(np.array(scipy.misc.lena())
image=signals.Image(np.array([scipy.misc.lena()])
image=signals.Image(np.array([scipy.misc.lena()]))
image=utils.stack([utils.stack([image]*3),axis=0]*3,axis=1)
image=signals.Image(np.array([scipy.misc.lena()]))
import scipy.ndimage
image = utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image=utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image.plot()
image = utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image.plot()
image=signals.Image([scipy.misc.lena()])
import scipy.ndimage
image = utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image = utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image.plot()
image=signals.Image(scipy.misc.lena())
import scipy.ndimage
image = utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image = utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image.plot()
image=signals.Image(scipy.misc.lena())
import scipy.ndimage
image = utils.stack([utils.stack([image]*3,axis=0)]*3,axis=1)
image.plot()
image.split()[0].split()[0]
image.plot()
image.split()[0].split()[0]
image.plot()
s=load('EELS Spectrum Image 16x16 0.2s 0.3eV 0offset.dm3')
s.change_dtype(float64)
s
s=load('EELS Spectrum Image 16x16 0.2s 0.3eV 0offset.dm3')
s
s.change_dtype('float64')
s
print(s)
s=signals.EELSSpectrum(np.random.normal(10,100))
s
s.print_summary_statistics
s.print_summary_statistics()
s = signals.EELSSpectrum(np.random.normal(size=(10,100)))
s.print_summary_statistics()
s.get_current_signal().print_summary_statistics()
img = signals.Image([np.random.chisquare(i+1,[100,100]) for i in range(5)])
utils.plot.plot_histograms(img,legend='auto')
s.metadata.Signal.set_item("Noise_properties.variance", 10)
s = signals.SpectrumSimulation(np.ones(100))
s.estimate_poissonian_noise_variance()
s.metadata
from urllib import urlretrieve
url = 'http://cook.msm.cam.ac.uk//~hyperspy//EDS_tutorial//'
urlretrieve(url + 'TiFeNi_010.rpl', 'Ni_superalloy_010.rpl')
urlretrieve(url + 'TiFeNi_010.raw', 'TiFeNi_010.raw')
urlretrieve(url + 'TiFeNi_012.rpl', 'TiFeNi_012.rpl')
urlretrieve(url + 'TiFeNi_011.raw', 'TiFeNi_011.raw')
urlretrieve(url + 'image010.tif', 'image010.tif')
urlretrieve(url + 'image011.tif', 'image011.tif')
img = load('image*.tif', stack=True)
img.plot(navigator='slider')
s = load('TiFeNi_0*.rpl', stack=True).as_spectrum(0)
s.plot()
img = load('image*.tif', stack=True)
img.plot(navigator='slider')
from urllib import urlretrieve
url = 'http://cook.msm.cam.ac.uk//~hyperspy//EDS_tutorial//'
urlretrieve(url + 'TiFeNi_010.rpl', 'Ni_superalloy_010.rpl')
urlretrieve(url + 'TiFeNi_010.raw', 'TiFeNi_010.raw')
urlretrieve(url + 'TiFeNi_012.rpl', 'TiFeNi_012.rpl')
urlretrieve(url + 'TiFeNi_011.raw', 'TiFeNi_011.raw')
urlretrieve(url + 'image010.tif', 'image010.tif')
urlretrieve(url + 'image011.tif', 'image011.tif')
img = load('image*.tif', stack=True)
img.plot(navigator='slider')
img = load('image*.tif', stack=True)
img.plot(navigator='slider')
s = load('TiFeNi_0*.rpl', stack=True).as_spectrum(0)
s.plot()
im = load('image*.tif', stack=True)
s = load('TiFeNi_0*.rpl', stack=True).as_spectrum(0)
dim = s.axes_manager.navigation_shape
#Rebin the image
im = im.rebin([dim[2], dim[0], dim[1]])
s.plot(navigator=im)
s=load('C:\Users\elp13va.VIE\Downloads\Dspec.530614.13.msa')
s.metadata
s.plot()
s=load('C:\Users\elp13va.VIE\Downloads\Dspec.56364.13.msa')
s.metadata
s.metadata
s.plot()
s.plot()
s=load('C:\Users\elp13va.VIE\Dropbox\MATLAB\EELS Spectrum Image 16x16 1s 0.5eV 78offset.dm3')
s.metadata
s.plot()
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.hspy as hp
s=load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Si B P 10082015\EELS Spectrum Image 25nm')
s=hp.load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Si B P 10082015\EELS Spectrum Image 25nm')
s=hp.load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Si B P 10082015\EELS Spectrum Image 25nm.dm3')
s.plot()
s.plot()
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.hspy as hp
s = load('EELS Spectrum Image 25nm')
s = load('EELS Spectrum Image 25nm.dm3')
s
s.plot()
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.hspy as hp
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.hspy as hp
s = hp.load('\EELS Spectrum Image disp0.2offset0time0.1s')
s = load('\EELS Spectrum Image disp0.2offset0time0.1s')
s = load('EELS Spectrum Image disp0.2offset0time0.1s')
s = load('EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s
s.plot()
s.plot()
s = load('EELS Spectrum Image 25nm.dm3')
s
s.plot()
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.hspy as hp
s = load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Si B P 19082015\EELS Spectrum Image 1nm probe 2.3nm step 80kX L.dm3')
s
s.plot()
s.plot()
s = load('EELS Spectrum Image 16x16 0.2s 0.3eV 0offset.dm3')
s
s.plot()
s = load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Ge-basedSolarCell_24082015\EELS Spectrum Image disp0.5offset250time0.5s.dm3')
s
s.plot()
import hyperspy.Release
print hyperspy.Release.info
import hyperspy.hspy as hp
s = load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Ge-basedSolarCell_24082015\EELS Spectrum Image disp0.5offset250time0.5s.dm3')
s
s.plot()
m = s.create_model()
s = hs.load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Ge-basedSolarCell_24082015\EELS Spectrum Image disp0.5offset250time0.5s.dm3')
s = hp.load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Ge-basedSolarCell_24082015\EELS Spectrum Image disp0.5offset250time0.5s.dm3')
s
m = s.create_model()
m = s.create_model()
import hyperspy.Release
print hyperspy.Release.info
import hp as hyperspy.hspy
get_ipython().magic(u'clear ')
s = load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Ge-basedSolarCell_24082015\EELS Spectrum Image disp1offset950time2s.dm3');
s
s.plot()
s.set_microscope_parameters(beam_energy=197, convergence_angle=20, collection_angle=13.5)
s
s.add_elements(('Cu', 'Ga', 'Ge', 'As', 'Al'))
ll = load('C:\Users\elp13va.VIE\Dropbox\MATLAB\Ge-basedSolarCell_24082015\EELS Spectrum Image disp0.2offset0time0.1s.dm3')
m = s.create_model(ll=ll)
m = s.create_model()
m = s.create_model()
s1=s(:,:,1)
s1 = s[1,1].data
s1
s1.plot()
s1.plot()
splot(s1)
plot(s1)
plot(s1)
s.plot()
s1 = s[6,12].data
plot(s1)
plot(s1)
s1.set_microscope_parameters(beam_energy=197, convergence_angle=20, collection_angle=13.5)
close all
close
exit()
s = hp.load('C:\Users\elp13va.VIE\Dropbox\MATLAB\AlNTb-P14-800degree')
s = hp.load("C:\Users\elp13va.VIE\Dropbox\MATLAB\AlNTb-P14-800degree")
s = hp.load('C:\\Users\\elp13va.VIE\\Dropbox\\MATLAB\\AlNTb-P14-800degree')
s = hp.load("C:\\Users\\elp13va.VIE\\Dropbox\\MATLAB\\AlNTb-P14-800degree")
s = hp.load("C:\\Users\\elp13va.VIE\\Dropbox\\MATLAB\\AlNTb-P14-800degree\\EELS Spectrum Image1.dm3")
s.plot()
get_ipython().magic('matplotlib inline')
s.plot()
q()
exit()
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic('pinfo hp')
s = hp.load("F:\\GaN_EPistar-nanowiresVEELS_28032017\\Data_16_06_2017\\sum25_GaNseries1")
s = hp.load("F:\GaN_EPistar-nanowiresVEELS_28032017\Data_16_06_2017\sum25_GaNseries1")
s = hp.load("F://GaN_EPistar-nanowiresVEELS_28032017//Data_16_06_2017//sum25_GaNseries1")
s = hp.load("sum25_GaNseries1")
s = hp.load("EELS Spectrum Image disp1offset950time2s.dm3")
s = hp.load("sum25_GaNseries1.dm3")
s.metadata
s.set_signal_type("EELS")
s.plot()
s
z = hp.load("singleframeZL_1D_0.05.dm3")
z = hp.load("singleframeZL_1D_0.05.dm3")
z.set_signal_type("EELS")
z.plot()
ds = s.richardson_lucy_deconvolution(z, iterations=15, mask=None, show_progressbar=None, parallel=None)
ds = s.richardson_lucy_deconvolution(z, iterations=15, mask=None)
ds.plot()
ds = s.richardson_lucy_deconvolution(z, iterations=100, mask=None)
ds.plot()
get_ipython().magic('matplotlib qt')
ds.plot()
get_ipython().magic('matplotlib gtk')
ds.plot()
ds = s.richardson_lucy_deconvolution(z, iterations=15, mask=None)
ds.plot()
ds = s.richardson_lucy_deconvolution(z, iterations=200, mask=None)
ds.plot()
s.plot()
plt.hold(True)
ds.plot()
plt.plot(s)
s.metadata
ds.plot()
ds.plot()
s.plot()
print hp.__version__
print hyperspy.__version__
print hyperspy.api.__version__
print hyperspy.__version__
print hp.__version__
pip install --upgrade hyperspy==1.3
import hyperspy.api as hp
conda install traitsui
conda instal traitsui
s = hp.load('sum25_GaNseries1.dm3')
z = hp.load('singleframeZL_1D_0.05.dm3')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.metadata
get_ipython().magic('matplotlib qt5')
s.plot()
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As'))
m = s.create_model()
m
m.multifit(kind='smart')
m.quantify()
m.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
import hyperspy.api as hp
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.metadata
get_ipython().magic('matplotlib qt5')
s.plot()
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu','Al'))
m = s.create_model()
m
m.multifit(kind='smart')
m.quantify()
m.plot()
m1 = s.create_model()
m1.smart_fit()
m1.plot()
m1
m1.quantify()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.metadata
get_ipython().magic('matplotlib qt5')
s.plot()
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu','Al'))
m = s.create_model()
m
m.multifit(kind='smart')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu','Al'))
m = s.create_model()
m
m.multifit(kind='smart')
m.quantify()
Ga = m.components.Ga_L3.intensity.as_signal();
Ga.plot()
As = m.components.As_L3.intensity.as_signal()
As.plot()
Al = m.components.Al_K.intensity.as_signal()
Al.plot()
Cu = m.components.Cu_L3.intensity.as_signal()
Cu.plot()
hs.preferences.gui(toolkit="ipywidgets")
hp.preferences.gui(toolkit="ipywidgets")
import traits.api as t
import traitlets
from link_traits import link
class A(t.HasTraits):
    a = t.Int()
class B(traitlets.HasTraits):
    b = t.Int()
a = A()
b = B()
l = link((a,"a"),(b,"b"))
a.a = 3
b.b
get_ipython().magic('matplotlib qt4')
import hyperspy.api as h
get_ipython().magic('matplotlib gtk')
Cu.plot()
m.components
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.set_signal_type('EELS')
m = s.create_model()
m.components
s
s.axes_manager[0] = 'x'
s.axes_manager[1] = 'y'
s.axes_manager[2] = 'eV'
s.axes_manager[0] = 'x'
s.axes_manager[1] = 'y'
s.axes_manager[2] = 't'
s
s.plot()
s.plot()
Sp = s.inav[10,80].data
Sp.plot()
Sp
Sp.signal
Sp.set_signal_type('EELS')
import hyperspy.api as hp
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
Sp = s.inav[10,80].data
Sp
s
m
get_ipython().magic('matplotlib nbagg')
import hyperspy.api as hp
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.plot()
import hyperspy.api as hp
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu','Al'))
m = s.create_model()
m
m.multifit(kind='smart')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.plot()
s.remove_background()
roi = s.roi.SpanROI(left=923, right=953)
s.plot()
roi.add_widget(s, axes=["Energy loss"])
roi = hp.roi.SpanROI(left=923, right=953)
s.plot()
roi.add_widget(s, axes=["Energy loss"])
roi = hp.roi.SpanROI(left=923, right=1023)
s.plot()
roi.add_widget(s, axes=["Energy loss"])
s_Cu = s.isig[roi].integrate1D(axis="Energy loss")
s_Cu.plot()
s_Ga = s.remove_background(signal_range=(950.,1110.)).isig[1115.:1315.].integrate1D(axis="Energy loss")
s_As = s.remove_background(signal_range=(1200.,1323.)).isig[1323.:1523.].integrate1D(axis="Energy loss")
s_Ga.plot
s_Ga.plot
s_Ga.plot()
s_Ga.plot()
s_Ga.plot()
s_Ga = s.remove_background(signal_range=(950.,1110.)).isig[1115.:1315.].integrate1D(axis="Energy loss")
s_Ga.plot()
s.plot()
s_Ga = s.remove_background(signal_range=(950.,1110.)).isig[1115.:1315.].integrate1D(axis="Energy loss")
s_Ga.plot()
s_Ga = s.remove_background()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_Ga = s.remove_background()
roi_Ga = hp.roi.SpanROI(left=1115, right=1315)
s.plot()
roi_Ga.add_widget(s, axes=["Energy loss"])
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_As = s.remove_background(signal_range=(1200.,1323.)).isig[1323.:1523.].integrate1D(axis="Energy loss")
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_As = s.remove_background(signal_range=(1200.,1323.)).isig[1323.:1523.].integrate1D(axis="Energy loss")
s.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_As = s.remove_background()
roi_As = hp.roi.SpanROI(left=1323, right=1523)
s.plot()
roi_As.add_widget(s, axes=["Energy loss"])
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_As = s.remove_background()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_As = s.remove_background()
roi_As = hp.roi.SpanROI(left=1323, right=1523)
s.plot()
roi_As.add_widget(s, axes=["Energy loss"])
s.spikes_removal_tool()
s.spikes_removal_tool()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_As = s.remove_background()
roi_As = hp.roi.SpanROI(left=1323, right=1600)
s.plot()
roi_As.add_widget(s, axes=["Energy loss"])
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu','Al'))
m = s.create_model()
m
m.multifit(kind='smart')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu','Al'))
m = s.create_model()
m
m.multifit(kind='smart')
m.quantify()
Ga = m.components.Ga_L3.intensity.as_signal();
Ga.plot()
As = m.components.As_L3.intensity.as_signal()
As.plot()
Al = m.components.Al_K.intensity.as_signal()
Al.plot()
Cu = m.components.Cu_L3.intensity.as_signal()
Cu.plot()
m.components
m.plot()
Al
for PIL import Image
import PIL
from PIL import Image
im = Image.fromarray(Cu)
im.save("~/EELS_hyperspy/Ge_based_GaAs_solarcell/Cu.png")
im = Image.fromarray(Cu)
im.save("~/EELS_hyperspy/Ge_based_GaAs_solarcell/Cu.jpeg")
im = Image.fromarray(Cu)
im.save("Cu.jpeg")
Cu
Cu.size
im = Cu.load()
m.components.Cu_L3.intensity.as_signal()
scipy.io.savemat("~/Ge_based_GaAs_solarcell/Cu",{"Cu":Cu.data})
import scipy.io as sio
scipy.io.savemat("~/Ge_based_GaAs_solarcell/Cu",{"Cu":Cu.data})
import scipy.io as sio
sio.savemat("~/Ge_based_GaAs_solarcell/Cu",{"Cu":Cu.data})
import scipy.io as sio
sio.savemat("Cu",{"Cu":Cu.data})
sio.savemat("Cu",{"Cu":Cu.data})
sio.savemat("Ga",{"Ga":Ga.data})
sio.savemat("As",{"As":As.data})
sio.savemat("Al",{"Al":Al.data})
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
m = s.create_model()
m
m.multifit(kind='smart')
m.quantify()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
ll = ll[90,43].data;
m = s.create_model(ll=ll)
ll = ll.isig[90,43].data;
m = s.create_model(ll=ll)
ll = ll.isig[90,43].data;
ll.isig[90,43].data;
ll.isig[90,43].data
ll = ll.inav[0,0].data
m = s.create_model(ll=ll)
ll = ll.inav[90,43].data
ll = ll.inav[43,90].data
ll.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
ll.plot()
m = s.create_model(ll=ll)
ll = ll.inav[43,90].data
ll = ll.inav[1,1].data
ll = ll.inav[40,90].data
ll = ll.inav[40,40].data
ll.plot()
ll = ll.inav[43,43].data
ll = ll.inav[1,2].data
import hyperspy.api as hp
import numpy as np
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
ll.plot()
ll = ll.inav[1,2].data
ll = ll.inav[43,90].data
ll = ll.inav[1,2].data
import hyperspy.api as hp
import numpy as np
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
ll.plot()
ll = ll.inav[40,40].data
m = s.create_model(ll=ll)
ll = ll.inav[40,40].data
ll.plot()
import hyperspy.api as hp
import numpy as np
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
ll.plot()
ll.inav[40,40].data
ll.inav[40,40].data.plot()
ll.inav[40,40].data
ll.inav[43,90].data
ll.inav[43,89].data
ll = ll.inav[43,89].data
m = s.create_model(ll=ll)
m = s.create_model()
m
m.quantify()
m.multifit(kind='smart')
m.quantify()
m.quantify()
Ga = m.components.Ga_L3.intensity.as_signal();
Ga.plot()
As = m.components.As_L3.intensity.as_signal()
As.plot()
Al = m.components.Al_K.intensity.as_signal()
Cu = m.components.Cu_L3.intensity.as_signal()
Cu.plot()
m.components
m.plot()
Ga
import scipy.io as sio
sio.savemat("Cu",{"Cu":Cu.data})
sio.savemat("Ga",{"Ga":Ga.data})
sio.savemat("As",{"As":As.data})
m.components.Cu_L3.intensity.as_signal()
m.plot()
m.plot()
