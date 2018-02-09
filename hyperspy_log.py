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
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s
s.save('low_loss_0eVoffset.hdf5')
s1 = hp.load('/Users/veersaysit/Dropbox/EELS_Hyperspy/low_loss_0eVoffset.hdf5')
s1
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
ll.plot()
m = s.create_model()
m
m.multifit(kind='smart')
m.quantify()
Ga = m.components.Ga_L3.intensity.as_signal();
Ga.plot()
As = m.components.As_L3.intensity.as_signal()
As.plot()
Cu = m.components.Cu_L3.intensity.as_signal()
Cu.plot()
m.components
m.plot()
import scipy.io as sio
sio.savemat("Cu",{"Cu":Cu.data})
sio.savemat("Ga",{"Ga":Ga.data})
sio.savemat("As",{"As":As.data})
m.components.Cu_L3.intensity.as_signal()
ll1 = ll.inav[1:90,1:43].data
ll1
ll1.set_signal_type("EELS")
ll2 = ll1.rebin((90,43,205))
ll2 = ll1.rebin((90,43,205)).data
ll1 = ll.inav[1:90,1:43].isig[]
ll1 = ll.inav[1:90,1:43].isig[1:1024]
ll1
ll
ll1 = ll.inav[0:42,0:89]
ll1
ll1 = ll.inav[0:44,0:90]
ll1
ll1 = ll.inav[0:42,0:90]
ll2
ll1
ll1 = ll.inav[0:43,0:90]
ll1
ll2 = ll1.rebin((43,90,205))
ll2
ll2.plot()
ll1 = ll.inav[:43,:90]
ll1
ll2 = ll1.rebin((43,90,205))
ll2 = ll1.rebin((1,1,5))
ll2
ll2 = ll1.rebin(scale=(1,1,5))
ll2
ll2.plot()
ll = ll.inav[:43,:90]
ll = ll.inav[:43,:90]
ll = ll.rebin(scale=(1,1,5))
m = s.create_model(ll=ll)
s = s.inav[:43,:90]
s = ll.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin(scale=(1,1,5))
ll
import hyperspy.api as hp
import numpy as np
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin(scale=(1,1,5))
ll
m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
m.quantify()
m.quantify()
Ga = m.components.Ga_L3.intensity.as_signal();
Ga.plot()
As = m.components.As_L3.intensity.as_signal()
As.plot()
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
m.components.Cu_L3.intensity.as_signal()
s = hs.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hs.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
s = hs.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")
ll = hs.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")
hs.datasets.eelsdb(formula="B2O3")
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
s.add_elements(('B', 'N'))
m = s.create_model(ll=ll)
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
m = s.create_model()
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll.data;
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll.data
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll.data = ll.data/sum(ll.data)
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll.data = ll.data/sum(ll.data)
ll
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll
ll.data = ll.data/sum(ll.data)
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll
#ll.data = ll.data/sum(ll.data)
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll.plot()
ll.data = ll.data/sum(ll.data)
ll.data = ll.data/sum(ll.data)
ll.plot()
s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
s.add_elements(('B', 'N'))
m = s.create_model()
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
m.quantify()
m = s.create_model(ll=ll)
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
m = s.create_model()
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
B = m.components.B_K.intensity.as_signal()
N = m.components.N_K.intensity.as_signal()
B/(B+N)
display(B/(B+N))
print(B/(B+N))
print(B/(B+N))
B
B
print B
print 'B'
print(B)
B.plot()
B.plot()
B.plot()
size(B)
print "%d" % (B)
print "%f" % (B)
B
m.components.B_K.intensity.as_signal()
B = m.components.B_K.intensity.as_signal()
N = m.components.N_K.intensity.as_signal()
B.data
B.data*100/(B.data+N.data)
N.data*100/(B.data+N.data)
import hyperspy.api as hp
import numpy as np
get_ipython().magic('matplotlib qt5')
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll.plot()
#ll.data = ll.data/sum(ll.data)
ll.plot()
s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
s.add_elements(('B', 'N'))
m = s.create_model()
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
B = m.components.B_K.intensity.as_signal()
m = s.create_model(ll=ll)
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
B = m.components.B_K.intensity.as_signal()
N = m.components.N_K.intensity.as_signal()
B.data
B.data*100/(B.data+N.data)
N.data*100/(B.data+N.data)
ll.data = ll.data/sum(ll.data)
ll.plot()
s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
s.add_elements(('B', 'N'))
m = s.create_model(ll=ll)
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
B = m.components.B_K.intensity.as_signal()
N = m.components.N_K.intensity.as_signal()
B.data
B.data*100/(B.data+N.data)
N.data*100/(B.data+N.data)
s = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="coreloss")[0]
ll = hp.datasets.eelsdb(title="Hexagonal Boron Nitride", spectrum_type="lowloss")[0]
ll.plot()
ll.data = ll.data/sum(ll.data)
ll.plot()
s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
s.add_elements(('B', 'N'))
m = s.create_model()
m.components
m.enable_fine_structure()
m.components
m.smart_fit()
m.quantify()
B = m.components.B_K.intensity.as_signal()
N = m.components.N_K.intensity.as_signal()
B.data
B.data*100/(B.data+N.data)
N.data*100/(B.data+N.data)
ll.data = ll.data/sum(ll.data)
ll.data = ll.data/sum(ll.data)
ll
ll.data = ll.data/sum(ll.data)
ll.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')[0]
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')[0]
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin(scale=(1,1,5))
ll
ll.data = ll.data/sum(ll.data)
ll.plot()
s = s.inav[:43,:90]
s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
ll.data = ll.data/sum(ll.data)
ll.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#ll.data = ll.data/sum(ll.data)
ll.plot()
#ll.data = ll.data/sum(ll.data)
ll.plot()
sum(ll.data).shape
#ll.data = ll.data/sum(ll.data)
ll.plot()
sum(sum(ll.data)).shape
#ll.data = ll.data/sum(ll.data)
ll.plot()
ll.sum(-1).data.shape
ll.data = ll.data/ll.sum(-1).data
ll.plot()
ll.data.shape
ll.data = ll.data/ll.sum(-1).data
ll.plot()
ll.sum(-1).data
ll.sum(-1).data.plot()
s1 = ll.sum(-1).data
s1.plot()
s1 = ll.sum(-1)
s1.plot()
ll.data = ll.data/ll.sum(-1)
ll.plot()
ll.data = ll.data./ll.sum(-1)
ll.plot()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s1 = ll.sum(-1)
s2 = np.matlib.repmat(s1, 1, 1, 205)
s2
s1 = ll.sum(-1)
s2 = np.matlib.repmat(s1, 1, 205)
s2
s1 = ll.sum(-1)
s2 = np.matlib.repmat(s1, 1, 205)
s2.shape
s1 = ll.sum(-1)
s2 = numpy.tile(s1, [1, 1, 5])
s2.shape
s1 = ll.sum(-1)
s2 = numpy.tile(s1, [1, 1, 5])
s1.shape
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s1 = ll.sum(-1)
s2 = numpy.tile(s1, [1, 1, 5])
s1.shape
s1 = ll.sum(-1)
#s2 = numpy.tile(s1, [1, 1, 5])
s1.shape
s1 = ll.sum(-1).data
#s2 = numpy.tile(s1, [1, 1, 5])
s1.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [1, 1, 5])
s1.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [1, 1, 5])
s2.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [1, 1, 1])
s2.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [1, 1, 2])
s2.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [1, 1, 1])
s2.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [5, 1, 1])
s2.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [1024, 1, 1])
s2.shape
s1 = ll.sum(-1).data
s2 = numpy.tile(s1, [205, 1, 1])
s2.shape
s1 = ll
s1.data = ll.sum(-1).data
s2.data = numpy.tile(s1.data, [205, 1, 1])
s2.shape
s1 = ll
s1.data = ll.sum(-1).data
s2.data = numpy.tile(s1.data, [205, 1, 1])
s2
s2 = ll
s1 = ll.sum(-1).data
s2.data = numpy.tile(s1.data, [205, 1, 1])
s2
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s2 = ll
s1 = ll.sum(-1).data
s2.data = numpy.tile(s1.data, [205, 1, 1])
s2
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
s2
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
s2.plot()
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = ll.data/s2.data
s2.plot()
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = ll.data/s2.data
ll.plot()
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = ll.data/s2.data
ll.plot()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = ll.data/s2.data
ll.plot()
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = np.divide(ll.data,s2.data)
ll.plot()
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = np.divide(ll.data,s2.data)
ll.plot()
import numpy as np
import matplotlib.pyplot as plt
s = hp.load('/Users/veersaysit/Desktop/EELS data/InGaN/100kV/EELS Spectrum Image6-b.dm3')
ax = s.plot_explained_variance_ratio(n=20)
s.decomposition()
ax = s.plot_explained_variance_ratio(n=20)
get_ipython().magic('matplotlib inline')
ax = s.plot_explained_variance_ratio(n=20)
ax = s.plot_explained_variance_ratio(n=20,threshold=4,xaxis_type='number')
sc = s.get_decomposition_model(components)
sc = s.get_decomposition_model(4)
(s-sc).plot()
get_ipython().magic('matplotlib qt5')
sc.plot()
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/InGaN/100kV/EELS Spectrum Image6-b.dm3')
s.decomposition()
ax = s.plot_explained_variance_ratio(n=20)
ax = s.plot_explained_variance_ratio(n=20,threshold=4,xaxis_type='number')
sc = s.get_decomposition_model(4)
(s-sc).plot()
sc.plot()
s.blind_source_separation(3)
plot_bss_results()
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=2, loadings_dim=2)
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=2, loadings_dim=1)
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=1)
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=1, loadings_dim=1)
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=2, loadings_dim=1)
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=2, loadings_dim=1(2))
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=1(2), loadings_dim=1(2))
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=1, loadings_dim=1(2))
plot_bss_results(factors_navigator='smart_auto', loadings_navigator='smart_auto', factors_dim=2, loadings_dim=1(2))
plot_decomposition_factors()
B = s.blind_source_separation(3)
B
B
ax
B
get_ipython().magic('matplotlib qt5')
S = hp.load('/Users/veersaysit/Desktop/EELS data/Dspec9TnZnl.msa')
S = hp.datasets.eelsdb(formula="GaAs")
S.plot()
S
S[:,1].plot()
S[:,1]
S(1)
S[1]
S[1].plot()
S[2].plot()
S[2].plot()
S[2]
S[1]
S[1,1]
S[1].data
S
S[-1]
S[0]
S[0].plot()
S[0].plot()
S[1].plot()
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/GaN_EPistar-nanowiresVEELS_28032017/profile_series1_gatan.dm3')
s
s.plot()
m = s.create_model(ll=s, auto_background=False)
m = s.create_model(auto_background=False)
s.estimate_thickness()
s.align_zero_loss_peak()
s.remove_background()
s.remove_background()
m = s.create_model()
s.set_signal_type("EELS")
m = s.create_model()
m = s.create_model()
s.metadata
m = s.create_model(ll=ll)
m = s.create_model()
s.set_microscope_parameters(beam_energy=197, convergence_angle=10, collection_angle=20)
m = s.create_model()
m.components
m = s.create_model(auto_background=False)
m.components
L1 = hs.model.components1D.Lorentzian()
L1 = hp.model.components1D.Lorentzian()
m.append(L1)
m.components
m.fit_component(g1)
m.fit_component(L1)
m.plot()
f = L1.function
f
f.plot()
L1.plot()
L1.plot()
m.plot()
L1.plot()
PL = L1.as_signal()
PL = L1.function(L1.A,L1.centre)
PL = L1.function()
PL = L1.function(A=L1.A, gamma=L1.gamma, centre=L1.centre)
PL = L1.function(gamma=L1.gamma, centre=L1.centre)
L1.A
L1.A.data
L1.A.as_signal
print L1.A
list L1.A
L1
L1.parameters.as_signal
L1.parameters.as_signal()
PL = m.components.Lorentzian.as_signal()
m
m.components
PL = m.as_signal()
PL.plot()
s1p = s-PL
s1p.plot()
import scipy.io as sio
sio.loadmat('S1p.m')
mat_contents = sio.loadmat('S1p.mat')
mat_contents
S1p = mat_contents['S1p']
S1p.plot()
S1p
S1p.set_signal_type('EELS')
S1p.set_signal_type("EELS")
plot(S1p)
S1p.data()
S1p.metadata
S1p = hp.signals.Signal1D(S1p)
S1p
S1p.plot()
S1p.plot()
S1p.set_signal_type('EELS')
S1p.plot()
S1p.shape
S1p = mat_contents['S1p']
S1p = mat_contents['S1p']
S1p.shape
S1p = hp.signals.Signal1D(S1p)
S1p = hp.signals.Signal1D(S1p)
S1p.set_signal_type('EELS')
S1p.shape
S1p.metadata
S1p.plot()
mat_contents = sio.loadmat('S1p.mat')
mat_contents
S1p = mat_contents['S1p']
S1p.shape
S1p = hp.signals.Signal1D(S1p)
S1p.set_signal_type('EELS')
S1p.metadata
S1p.plot()
rS1p = S1p.fourier_log_deconvolution()
mat_contents = sio.loadmat('eS1p.mat')
mat_contents = sio.loadmat('eS1p.mat')
eS1p = mat_contents['eS1p']
eS1p = hp.signals.Signal1D(eS1p)
eS1p.set_signal_type('EELS')
rS1p = S1p.fourier_log_deconvolution(zlp=eS1p)
rS1p.plot()
flS1p = S1p.fourier_log_deconvolution(zlp=eS1p)
rS1p.plot()
flS1p.plot()
sio.savemat("flS1p",{"flS1p":flS1p.data})
flS1p.data
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/GaN_EPistar-nanowiresVEELS_28032017/Data_27_06_2017/Profile Of sum24_series3.dm3')
s
s.plot()
s.plot()
s.align_zero_loss_peak()
s.set_signal_type("EELS")
s.metadata
s.set_microscope_parameters(beam_energy=197, convergence_angle=10, collection_angle=20)
m = s.create_model(auto_background=False)
m.components
L3 = hp.model.components1D.Lorentzian()
m.append(L3)
m.components
m.fit_component(L3)
m.plot()
L3
L3.A.as_signal
m.components
PL = m.as_signal()
PL.plot()
s3p = s-PL
s3p.plot()
import scipy.io as sio
mat_contents = sio.loadmat('S3_combined.mat')
#Note that while saving .mat file in matlab always make sure it is in (1,1024) dimention and not in (1024,1).
mat_contents
S3p = mat_contents['S3p']
S3p.shape
S3p = hp.signals.Signal1D(S3p)
S3p.set_signal_type('EELS')
S3p.metadata
S3p.plot()
eS3p = mat_contents['eS3p']
eS3p = hp.signals.Signal1D(eS3p)
eS3p.set_signal_type('EELS')
flS3p = S1p.fourier_log_deconvolution(zlp=eS3p)
flS3p = S3p.fourier_log_deconvolution(zlp=eS3p)
flS3p.plot()
sio.savemat("flS3p",{"flS3p":flS3p.data})
flS3p.data
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/GaAs100_Q4_EELStest_130705/Pos1_20muCA/Profile Of Pos 1 EELS_1s.dm3')
s
s.plot()
s.align_zero_loss_peak()
s.set_signal_type("EELS")
s.metadata
s.set_microscope_parameters(beam_energy=197, convergence_angle=10, collection_angle=20)
m = s.create_model(auto_background=False)
m.components
Lg = hp.model.components1D.Lorentzian()
m.append(Lg)
m.components
m.fit_component(Lg)
m.plot()
m.plot()
Lg
Lg.A.as_signal
m.components
PL = m.as_signal()
PL.plot()
sgp = s-PL
sgp.plot()
import scipy.io as sio
#mat_contents = sio.loadmat('S3_combined.mat')
#Note that while saving .mat file in matlab always make sure it is in (1,1024) dimention and not in (1024,1).
#mat_contents
#S3p = mat_contents['S3p']
#S3p.shape
#S3p = hp.signals.Signal1D(S3p)
#S3p.set_signal_type('EELS')
#S3p.metadata
sgp.plot()
import scipy.io as sio
mat_contents = sio.loadmat('Sg_combined.mat')
#Note that while saving .mat file in matlab always make sure it is in (1,1024) dimention and not in (1024,1).
mat_contents
S3p = mat_contents['S3p']
S3p.shape
Sg1p = mat_contents['Sg1p']
Sg1p.shape
Sg1p = hp.signals.Signal1D(Sg1p)
Sg1p.set_signal_type('EELS')
Sg1p.metadata
Sg1p.plot()
eSg1p = mat_contents['eSg1p']
eSg1p = hp.signals.Signal1D(eSg1p)
eSg1p.set_signal_type('EELS')
flSg1p = Sg1p.fourier_log_deconvolution(zlp=eSg1p)
flSg1p.plot()
sio.savemat("flSg1p",{"flSg1p":flS3p.data})
sio.savemat("flSg1p",{"flSg1p":flSg1p.data})
flSg1p.data
flSg1p = Sg1p.fourier_log_deconvolution()
flSg1p = Sg1p.fourier_log_deconvolution(zlp=eSg1p)
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.plot()
import hyperspy.api as hp
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.plot()
s.remove_background()
s.remove_background()
s.plot()
s.remove_background()
s.remove_background()
roi = hp.roi.SpanROI(left=923, right=1023)
s.plot()
roi.add_widget(s, axes=["Energy loss"])

s_Cu = s.isig[roi].integrate1D(axis="Energy loss")
s_Cu.plot()
roi = hp.roi.SpanROI(left=931, right=1031)
s.plot()
roi.add_widget(s, axes=["Energy loss"])

s_Cu = s.isig[roi].integrate1D(axis="Energy loss")
s_Cu.plot()

s_Cu = s.isig[roi].integrate1D(axis="Energy loss")
s_Cu.plot()
s_Cu.data
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_Ga = s.remove_background()
s.plot()
roi = hp.roi.SpanROI(left=1023, right=1113)
s.remove_background(signal_range=roi, background_type="Exponential")
s.remove_background(signal_range=roi, background_type="Power Law")
s.remove_background(signal_range=roi)
roi = hp.roi.SpanROI(left=931, right=1031)
s.plot()
roi.add_widget(s, axes=["Energy loss"])
s.remove_background()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s.remove_background(signal_range=(891.,929.)).isig[931.:1031.].integrate1D(axis="Energy loss")
roi = hp.roi.SpanROI(left=931, right=10)
s.plot()
roi.add_widget(s, axes=["Energy loss"])

#s_Cu = s.isig[roi].integrate1D(axis="Energy loss")
s_Cu.plot()
s_Cu = s.remove_background(signal_range=(891.,929.)).isig[931.:1031.].integrate1D(axis="Energy loss")
#roi = hp.roi.SpanROI(left=931, right=10)
#s.plot()
#roi.add_widget(s, axes=["Energy loss"])

#s_Cu = s.isig[roi].integrate1D(axis="Energy loss")
s_Cu.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_Ga = s.remove_background(signal_range=(1023.,1113.)).isig[1115.:1217.].integrate1D(axis="Energy loss")
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_As = s.remove_background()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
s_Ge = s.remove_background(signal_range=(1166.,1215.)).isig[1217.:1323.].integrate1D(axis="Energy loss")
s_As = s.remove_background(signal_range=(1270.,1321.)).isig[1323.:1560.].integrate1D(axis="Energy loss")
s_AlK = s.remove_background(signal_range=(1441.,1558.)).isig[1560.:1660.].integrate1D(axis="Energy loss")
import scipy.io as sio
sio.savemat("Cu_hspy",{"Cu_hspy":s_Cu.data})
sio.savemat("Ga_hspy",{"Ga_hspy":s_Ga.data})
sio.savemat("Ge_hspy",{"Ge_hspy":s_Ge.data})
sio.savemat("As_hspy",{"As_hspy":s_As.data})
sio.savemat("AlK_hspy",{"AlK_hspy":s_AlK.data})
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
s_C = s.remove_background(signal_range=(242.,283.)).isig[284.:334.].integrate1D(axis="Energy loss")
s_In = s.remove_background(signal_range=(363.,442.)).isig[443.:532.].integrate1D(axis="Energy loss")
s_O = s.remove_background(signal_range=(487.,531.)).isig[532.:582.].integrate1D(axis="Energy loss")
sio.savemat("C_hspy",{"C_hspy":s_C.data})
sio.savemat("In_hspy",{"In_hspy":s_In.data})
sio.savemat("O_hspy",{"O_hspy":s_O.data})
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image small disp0.1offset80time0.5s.dm3')
s_AlL3 = s.remove_background(signal_range=(72.,72.8)).isig[73.:100.].integrate1D(axis="Energy loss")
s_Si = s.remove_background(signal_range=(86.,99.8)).isig[100.:115.].integrate1D(axis="Energy loss")
s_AlL1 = s.remove_background(signal_range=(109.,117.8)).isig[118.:135.].integrate1D(axis="Energy loss")
s_P = s.remove_background(signal_range=(126.,134.8)).isig[135.:172.4].integrate1D(axis="Energy loss")
sio.savemat("AlL3_hspy",{"AlL3_hspy":s_AlL3.data})
sio.savemat("Si_hspy",{"Si_hspy":s_Si.data})
sio.savemat("AlL1_hspy",{"AlL1_hspy":s_AlL1.data})
sio.savemat("P_hspy",{"P_hspy":s_P.data})
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s2.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = np.divide(ll.data,s2.data)
ll.plot()
s.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = np.divide(ll.data,s2.data)
ll.plot()
s.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = np.divide(ll.data,s.data)
ll.plot()
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = np.divide(ll.data,s.data)
ll.plot()
s1.plot()
#s.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s1.plot()
ll.data = ll.data/ll.sum(-1)
ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
m = s.create_model(ll=ll)
m
m.components
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#s.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s1.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
m.components
s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
m
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
m.quantify()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
m = s.create_model(ll=ll)
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
#s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
m
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
m.quantify()
s.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
ll.data = np.divide(ll.data,s.data)
#ll.plot()
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s1.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
m.quantify()
Ga = m.components.Ga_L3.intensity.as_signal();
Ga.plot()
As = m.components.As_L3.intensity.as_signal()
As.plot()
Cu = m.components.Cu_L3.intensity.as_signal()
Cu.plot()
m.components
m.plot()
m.plot()
import scipy.io as sio
sio.savemat("Cu_hspy_mlls",{"Cu_hspy_mlls":Cu.data})
sio.savemat("Ga_hspy_mlls",{"Ga_hspy_mlls":Ga.data})
sio.savemat("As_hspy_mlls",{"As_hspy_mlls":As.data})
m.components.Cu_L3.intensity.as_signal()
#m.plot()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s1.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
#s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s1.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
s.add_elements(('C','In','O'))
m = s.create_model(ll=ll)
s.plot()
s.plot()
ll.plot()
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
s.add_elements(('C','In','O'))
m = s.create_model(ll=ll)
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
#s.add_elements(('C','In','O'))
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
#s.add_elements(('C','In','O'))
#s = s.inav[:44,:90]
###########s = s.rebin(scale=(1,1,1))
#s = s.rebin((44,90,1024))
#s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
#s.add_elements(('C','In','O'))
#s = s.inav[:44,:90]
###########s = s.rebin(scale=(1,1,1))
#s = s.rebin((44,90,1024))
#s
ll = ll.inav[:44,:90]
ll = ll.rebin((44,90,205))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
s.add_elements(('C','In','O'))
m = s.create_model(ll=ll)
s.add_elements(('C','In','O'))
m = s.create_model(ll=ll)
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
#s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,410))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
s.add_elements(('C','In','O'))
m = s.create_model(ll=ll)
s.add_elements(('C','In','O'))
m = s.create_model()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('C','In','O'))
#s.add_elements(('C','In','O'))
m = s.create_model()
s.plot()
#ll.plot()
#s.add_elements(('C','In','O'))
m = s.create_model()
s.add_elements(('C','In','O'))
hp.preferences.gui()
s.add_elements(('C','In','O'))
#s.add_elements(('C','In','O'))
m = s.create_model()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s1.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
m
m.components
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
hp.preferences.gui()
s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,409))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
#ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('C','In','O'))
m = s.create_model()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
s.add_elements(('C','In','O'))
#s.add_elements(('C','In','O'))

m = s.create_model()
hp.preferences.gui()
s.add_elements(('C','In','O'))
#s.add_elements(('C','In','O'))

m = s.create_model()
s.add_elements(('C','O'))
#s.add_elements(('C','In','O'))

m = s.create_model()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
s.add_elements(('C','O'))
#s.add_elements(('C','In','O'))

m = s.create_model()
s.add_elements(('C','In','O'))
#s.add_elements(('C','In','O'))

m = s.create_model()
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
s.add_elements(('C','In','O'))
#s.add_elements(('C','In','O'))

m = s.create_model()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
hp.preferences.gui()
hp.preferences.gui()
import hyperspy.api as hs
import numpy as np
x = np.arange(100, 1000)
s = hs.signals.EELSSpectrum(x)
s.set_microscope_parameters(100, 10, 10)
s.add_elements(("C", "O"))
m = s.create_model()
m.components
import hyperspy.api as hs
import numpy as np
x = np.arange(100, 1000)
s = hs.signals.EELSSpectrum(x)
s.set_microscope_parameters(100, 10, 10)
s.add_elements(("C", "O", "In"))
m = s.create_model()
m.components
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
hp.preferences.gui()
s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,409))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
#ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('C','In','O'))

m = s.create_model()
import hyperspy.api as hs
import numpy as np
x = np.arange(100, 1000)
s = hs.signals.EELSSpectrum(x)
s.set_microscope_parameters(100, 10, 10)
s.add_elements(("C", "O", "In"))
m = s.create_model()
m.components
import hyperspy.api as hs
import numpy as np
x = np.arange(100, 1000)
s = hs.signals.EELSSpectrum(x)
s.set_microscope_parameters(100, 10, 10)
s.add_elements(("C", "O", "In"))
m = s.create_model()
m.components
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
hp.preferences.gui()
s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,409))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
#ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('C','In','O'))

m = s.create_model()
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.5offset250time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,409))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
#ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('C','In','O'))

m = s.create_model()
m
m.components
m.multifit(kind='smart')
hp.preferences.gui()
s.add_elements(('C','In','O'))
s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,409))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
#ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('C','In','O'))

m = s.create_model()
m
m.components
m.multifit(kind='smart')
m.quantify()
Ga = m.components.Ga_L3.intensity.as_signal();
C = m.components.C_K.intensity.as_signal();
C.plot()
In_M5 = m.components.In_M5.intensity.as_signal()
In_M4 = m.components.In_M4.intensity.as_signal()
In_M3 = m.components.In_M3.intensity.as_signal()
In_M2 = m.components.In_M2.intensity.as_signal()
In.plot()
In_M5.plot()
In_M4.plot()
In_M3.plot()
In_M2.plot()
In_M5.plot()
In_M5.plot()
In = In_M5+In_M4+In_M3+In_M2
In.plot()
#In_M5.plot()
In = In_M5+In_M4+In_M3+In_M2
In.plot()
(In_M5+In_M4+In_M3+In_M2).plot()
In.plot()
In.plot()
O = m.components.O_K.intensity.as_signal()
O.plot()
m.components
m.plot()
#s.add_elements(('C','In','O'))

m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
m.quantify()
C = m.components.C_K.intensity.as_signal();
C.plot()
In_M5 = m.components.In_M5.intensity.as_signal()
In_M4 = m.components.In_M4.intensity.as_signal()
In_M3 = m.components.In_M3.intensity.as_signal()
In_M2 = m.components.In_M2.intensity.as_signal()
In.plot()
In_M5.plot()
O
import scipy.io as sio
sio.savemat("C_hspy_mlls",{"C_hspy_mlls":Cu.data})
sio.savemat("In_hspy_mlls",{"In_hspy_mlls":Ga.data})
sio.savemat("O_hspy_mlls",{"O_hspy_mlls":As.data})
sio.savemat("C_hspy_mlls",{"C_hspy_mlls":C.data})
sio.savemat("In_hspy_mlls",{"In_hspy_mlls":In.data})
sio.savemat("O_hspy_mlls",{"O_hspy_mlls":O.data})
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image small disp0.1offset80time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
s.add_elements(('Al','Si','P'))
#s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
#s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,2048))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
s.plot()
#ll.plot()
#s.plot()
ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('C','In','O'))

m = s.create_model(ll=ll)
s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image small disp0.1offset80time0.5s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
hp.preferences.gui()
s.add_elements(('Al','Si','P'))
#s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
#s = s.rebin((43,90,1024))
#s
#s = s.inav[:43,:90]
###########s = s.rebin(scale=(1,1,1))
#s = s.rebin((43,90,1024))
s
ll = ll.inav[:22,:45]
ll = ll.rebin((22,45,2048))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s.plot()
ll.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
#s.add_elements(('C','In','O'))

m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
m.quantify()
C = m.components.C_K.intensity.as_signal();
Al_L3 = m.components.Al_L3.intensity.as_signal();
In_M5.plot()
In = (In_M5+In_M4+In_M3+In_M2)
O = m.components.O_K.intensity.as_signal()
m.components
import scipy.io as sio
sio.savemat("C_hspy_mlls",{"C_hspy_mlls":C.data})
sio.savemat("In_hspy_mlls",{"In_hspy_mlls":In.data})
sio.savemat("O_hspy_mlls",{"O_hspy_mlls":O.data})
Al.plot()
Al_L3.plot()
Al_L1.plot()
Al_L3 = m.components.Al_L3.intensity.as_signal();
Al_L1.plot()
Al_L3 = m.components.Al_L3.intensity.as_signal();
Al_L3.plot()
Al_L1 = m.components.Al_L1.intensity.as_signal();
Al_L1.plot()
Al = (Al_L3+Al_L1)
Al = (Al_L3+Al_L1)
Al.plot()
Si_L3 = m.components.Si_L3.intensity.as_signal()
Si_L2 = m.components.Si_L2.intensity.as_signal()
Si_L1 = m.components.Si_L1.intensity.as_signal()
Si.plot()
Si_L3 = m.components.Si_L3.intensity.as_signal()
Si_L2 = m.components.Si_L2.intensity.as_signal()
Si_L1 = m.components.Si_L1.intensity.as_signal()
Si = (Si_L3+Si_L2+Si_L1)
Si.plot()
m.components
m.plot()
P = m.components.P_L3.intensity.as_signal()
P = m.components.P_L3.intensity.as_signal()
P.plot()
m.components
m.plot()
P
import scipy.io as sio
sio.savemat("Al_hspy_mlls",{"Al_hspy_mlls":Al.data})
sio.savemat("Si_hspy_mlls",{"Si_hspy_mlls":Si.data})
sio.savemat("P_hspy_mlls",{"P_hspy_mlls":P.data})
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
#s1.data = numpy.tile(ll.sum(-1).data, [205, 1, 1])
#ll.data = np.divide(ll.data,s.data)
#ll.plot()
#s1.plot()
#ll.data = ll.data/ll.sum(-1)
#ll.plot()
s.add_elements(('Ga','As','Cu'))
m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
import hyperspy.api as hp
import numpy as np
import numpy.matlib
get_ipython().magic('matplotlib qt5')
s = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp1offset950time2s.dm3')
ll = hp.load('/Users/veersaysit/Desktop/EELS data/Ge-basedSolarCell_24082015/EELS Spectrum Image disp0.2offset0time0.1s.dm3')
s.set_signal_type("EELS")
ll.set_signal_type("EELS")
s.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
ll.set_microscope_parameters(beam_energy=197,collection_angle=15,convergence_angle=16.6)
s.add_elements(('Ga','As','Cu'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s.add_elements(('Ga','As','Cu','In'))
s = s.inav[:43,:90]
#s = s.rebin(scale=(1,1,1))
s = s.rebin((43,90,1024))
s
ll = ll.inav[:43,:90]
ll = ll.rebin((43,90,205))
ll
s.add_elements(('Ga','As','Cu','In'))

m = s.create_model(ll=ll)
m
m.components
m.multifit(kind='smart')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib qt5')
get_ipython().magic('matplotlib inline')
s = hp.load('/Users/veersaysit/Desktop/EELS data/InGaN/100kV/EELS Spectrum Image6-b.dm3')
s.decomposition()
ax = s.plot_explained_variance_ratio(n=20)
ax = s.plot_explained_variance_ratio(n=20)
ax = s.plot_explained_variance_ratio(n=20,threshold=4,xaxis_type='number')
sc = s.get_decomposition_model(4)
(s-sc).plot()
sc.plot()
B = s.blind_source_separation(3)
B
m.quantify()
