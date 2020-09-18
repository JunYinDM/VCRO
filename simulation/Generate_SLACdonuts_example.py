# Python script to make the donuts.

import sys

import numpy as np

# Import relevant donut libraries.
from donutlib.makedonut import makedonut
from donutlib.donutfit import donutfit

# Import libraries to do image processing.
# We will open the donut files created and the save the images as png and hd5 files.
# Several libraries must be imported to make this happen.
from astropy.io import fits as pyfits
import imageio
import h5py

# Define a function to make the donuts.
def makeTheDonuts(donut1Name,donut2Name,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,iNr):
    
    # Define Zernike coefficients of the donuts.
    # The first 3 terms are set to 0.
    z1 = 0.0
    z2 = 0.0
    z3 = 0.0
    
    # In order to create defocussed images, we add 40 micrometer to z4 for donut image 1 and 
    # subtract 40 micrometer for donut image 2.
    #z4 = z4 + 40.

    # Define the background.
    bkgr = 0.0

    # Define the x,y position of the camera.
    xCam = 0.00
    yCam = 0.00

    # Set the random number used in the simulation
    # rndSeed = 2314809
    rndSeed = np.random.randint(1,np.iinfo(np.int32).max)

    # Create the input dictionary for donut production (donut 1
    inputDict = {'writeToFits':True,'outputPrefix':donut1Name,'iTelescope':3,'nZernikeTerms':37,\
                 'nbin':1792,'nPixels':224,'pixelOverSample':8,'scaleFactor':1.,'rzero':0.125,\
                 'nEle':1.0e8, 'background':bkgr, 'randomFlag':False, 'randomSeed':rndSeed,\
                 'ZernikeArray':[z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22],\
                 'xDECam':xCam,'yDECam':yCam,'printLevel':5}
    #,'xDECam':xCam,'yDECam':yCam}
            
    # Create the image of donut 1
    m = makedonut(**inputDict)
    donut1 = m.make()
            
    # Update the disctionary for donut 2 and create the image of donut 2.
    # Note: we also change the value of z4 by -80 micrometer to reflect that we want to look at the second 
    # out-of-focus sensor.  The out-of-focus sensors are assumed to be defocussed by +40 micrometer and 
    # -40 micrometer.  We added 40 micrometer to z4 when we generated donut 1; we now have to subtract
    # 80 micrometer in order to create donut 2.
    #z4 = z4 - 80.
    newDict = {'outputPrefix':donut2Name,\
               'ZernikeArray':[z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22]}
    donut2 = m.make(**newDict)

    # Print current file names.
    print donut1Name
    print donut2Name

    # Done making donuts.
    
    # We now create the images we need for learning:
    # donut 1
    # donut 2
    # donut difference = donut 1 - donut 2
    # These images are written in png and hdf5 format.
    # Open the data files.
    #hdu1 = pyfits.open(donut1Name+'.stamp.fits')
    #hdu2 = pyfits.open(donut2Name+'.stamp.fits')
                       
    # Get the image data.
    #donut1Data = hdu1[0].data
    #donut2Data = hdu2[0].data

    # Calculate the difference between donut 1 and donut 2
    #donutDiffData = donut1Data - donut2Data
    
    # Calculate the name of the image file for the donut difference.
    #diffName = donut1Name[0:22]+'12'+donut1Name[24:]

    # Write images in png format.
    #imageio.imwrite(donut1Name+'.png', donut1Data)
    #imageio.imwrite(donut2Name+'.png', donut2Data)
    #imageio.imwrite(diffName+'.png', donutDiffData)

    # Write image in hdf5 format.
    #h5f = h5py.File(donut1Name+'.h5', 'w')
    #h5f.create_dataset('Donut 1', data=donut1Data)
    #h5f.close()
    
    #h5f = h5py.File(donut2Name+'.h5', 'w')
    #h5f.create_dataset('Donut 2', data=donut2Data)
    #h5f.close()
    
    #h5f = h5py.File(diffName+'.h5', 'w')
    #h5f.create_dataset('Donut 1 - Donut 2', data=donutDiffData)
    #h5f.close()

    return


if __name__ == "__main__":
    print sys.argv[1]
    print sys.argv[2]
    print sys.argv[3]
    print sys.argv[4]
    print sys.argv[5]
    print sys.argv[6]
    print sys.argv[7]
    print sys.argv[8]
    print sys.argv[9]
    print sys.argv[10]
    print sys.argv[11]
    print sys.argv[12]
    print sys.argv[13]
    print sys.argv[14]
    print sys.argv[15]
    print sys.argv[16]
    print sys.argv[17]
    print sys.argv[18]
    print sys.argv[19]
    print sys.argv[20]
    print sys.argv[21]
    print sys.argv[22]

    makeTheDonuts(sys.argv[1],sys.argv[2],\
                  float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),\
                  float(sys.argv[8]),float(sys.argv[9]),float(sys.argv[10]),float(sys.argv[11]),float(sys.argv[12]),\
                  float(sys.argv[13]),float(sys.argv[14]),float(sys.argv[15]),float(sys.argv[16]),float(sys.argv[17]),\
                  float(sys.argv[18]),float(sys.argv[19]),float(sys.argv[20]),float(sys.argv[21]),\
                  int(sys.argv[22]))



