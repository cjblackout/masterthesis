package thud.meanshift;

import java.awt.image.BufferedImage;

import org.opencv.core.Mat;

import com.greatmindsworking.EDISON.segm.ImageType;
import com.greatmindsworking.EDISON.segm.MSImageProcessor;
import com.greatmindsworking.EDISON.segm.SpeedUpLevel;
import thud.utils.ImageProcessingUtils;


public class Pymeanshift
{
	// COLOR RADIUS FOR MEAN SHIFT ANALYSIS IMAGE SEGMENTATION
	private float colorRadius = 7f; 	
	// SPATIAL RADIUS FOR MEAN SHIFT ANALYSIS IMAGE SEGMENTATION
	private int spatialRadius = 6; 	
	// MINIMUM NUMBER OF PIXEL THAT CONSTITUTE A REGION FOR MEAN SHIFT ANALYSIS IMAGE SEGMENTATION
	private int minRegion = 40; 				

    ImageProcessingUtils utils = new ImageProcessingUtils();

    public Pymeanshift(int spatialRadius, float colorRadius, int minRegion) 
    {
        this.spatialRadius = spatialRadius;
        this.colorRadius = colorRadius;
        this.minRegion = minRegion;
    }
    
/*
 * 
 * ============================================================================== 
 * ReadMe file for Java port of the EDISON mean shift image segmentation code. 
 * ============================================================================== 
 * Copyright (c) 2002-2014, Brian E. Pangburn & Jonathan P. Ayo 
 * All rights reserved. Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met: Redistributions of source code must retain 
 * the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary 
 * form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the 
 * documentation and/or other materials provided with the distribution. The names of its contributors may not be 
 * used to endorse or promote products derived from this software without specific prior written permission. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 * IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
 * OF SUCH DAMAGE. 
 * 
 * This software is a partial port of the EDISON system developed by Chris M. Christoudias and Bogdan Georgescu at 
 * the Robust Image Understanding Laboratory at Rutgers University (http://www.caip.rutgers.edu/riul/). 
 * EDISON is available from: http://www.caip.rutgers.edu/riul/research/code/EDISON/index.html 
 * It is based on the following references: 
 * [1] D. Comanicu, P. Meer: "Mean shift: A robust approach toward feature space analysis". 
 * IEEE Trans. Pattern Anal. Machine Intell., May 2002. 
 * [2] P. Meer, B. Georgescu: "Edge detection with embedded confidence". 
 * IEEE Trans. Pattern Anal. Machine Intell., 28, 2001. 
 * [3] C. Christoudias, B. Georgescu, P. Meer: "Synergism in low level vision". 
 * 16th International Conference of Pattern Recognition, Track 1 - Computer Vision and Robotics, Quebec City, Canada, August 2001. 
 * The above cited papers are available from: http://www.caip.rutgers.edu/riul/research/robust.html 
 * ============================================================================== 
 * This program is a Java port of the mean shift image segmentation portion of the EDISON system developed by the 
 * Robust Image Understanding Laboratory at Rutgers University. It is more of a hack than an attempt at software engineering. 
 * The port involved the following general steps: 1. consolidate header files (.h) and class files (.cpp) into Java classes 
 * (.java) 2. consolidate existing documentation following JavaDoc conventions 3. eliminate pointers 4. tinker with any 
 * other data structures and constructs not compatible with Java until the code compiled 5. move the code into the Java 
 * package com.greatmindsworking.EDISON.segm We've added an executable class called SegTest that can be used to segment an 
 * image from the command line. The port was done so that the mean shift image segmentation algorithms in EDISON could be 
 * incorporated into a separate Java software system called Experience-Based Language Acquisition (EBLA). EBLA allows a computer 
 * to acquire a simple language of nouns and verbs based on a series of visually perceived "events". The segmentation algorithms 
 * form the backbone for EBLA's vision system. For more information on EBLA, visit http://www.greatmindsworking.com This 
 * release of jEDISON allow for segmentation based on either the 04-25-2002 or the 04-14-2003 release of the C++ EDISON code.
 * 
 */
    public Mat pms_segment(BufferedImage tmpImage, float colorRadius, int spatialRadius, int minRegion)
    {
		int edisonPortVersion = 1;			// INDICATES WHICH PORT OF EDISON TO USE (0=04-25-2003; 1=04-14-2003)
		boolean displayText = false;			// INDICATES WHETHER OR NOT TO DISPLAY DETAILED MESSAGES
//		float colorRadius = (float)6; 	// COLOR RADIUS FOR MEAN SHIFT ANALYSIS IMAGE SEGMENTATION
//		int spatialRadius = 7; 				// SPATIAL RADIUS FOR MEAN SHIFT ANALYSIS IMAGE SEGMENTATION
//		int minRegion = 40; 				// MINIMUM NUMBER OF PIXEL THAT CONSTITUTE A REGION FOR MEAN SHIFT ANALYSIS IMAGE SEGMENTATION
		int speedUp = 1; 					// SPEED-UP LEVEL FOR MEAN SHIFT ANALYSIS IMAGE SEGMENTATION 0=NO SPEEDUP, 1=MEDIUM SPEEDUP, 2=HIGH SPEEDUP
		float highSpeedUpFactor = (float)1.0; // WHEN speedUp = 2, values range between 0.0 (high quality) and 1.0 (high speed)

		// LOAD SOURCE IMAGE USING LOADER CLASS
//			BufferedImage tmpImage = ImageIO.read(new File(inputFile));

		// DETERMINE WIDTH AND HEIGHT
			int width = tmpImage.getWidth();
			int height = tmpImage.getHeight();

		// CROP IMAGE
		//	tmpImage = tmpImage.getSubimage(5, 5, width-5, height-5);

		// RECALCULATE WIDTH AND HEIGHT
			width = tmpImage.getWidth();
			height = tmpImage.getHeight();

		// DETERMINE NUMBER OF PIXELS
			int pixelCount = width * height;

		// INITIALIZE ARRAYS FOR RGB PIXEL VALUES
			int rgbPixels[] = new int[pixelCount];
			tmpImage.getRGB(0, 0, width, height, rgbPixels, 0, width);

		// CREATE MSImageProcessor OBJECT
			MSImageProcessor mySegm = new MSImageProcessor();

		// SET IMAGE
			mySegm.DefineBgImage(rgbPixels, ImageType.COLOR, height, width);

		// SET SetSpeedThreshold FOR HIGH SPEEDUP OPTION
			if (speedUp == 2) 
			{
				mySegm.SetSpeedThreshold(highSpeedUpFactor);
			}

		// SEGMENT IMAGE
			if (speedUp == 0) 
			{
				mySegm.Segment(edisonPortVersion, displayText, spatialRadius,
					colorRadius, minRegion, SpeedUpLevel.NO_SPEEDUP);
			} 
			else if (speedUp == 1) 
			{
				mySegm.Segment(edisonPortVersion, displayText, spatialRadius,
					colorRadius, minRegion, SpeedUpLevel.MED_SPEEDUP);
			} 
			else 
			{
				mySegm.Segment(edisonPortVersion, displayText, spatialRadius,
					colorRadius, minRegion, SpeedUpLevel.HIGH_SPEEDUP);
			}

		// GET RESULTING SEGMENTED IMAGE (RGB) PIXELS
			int segpixels[] = new int[pixelCount];
			mySegm.GetResults(segpixels);

		// BUILD BUFFERED IMAGE FROM RGB PIXEL DATA
			BufferedImage segImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			segImage.setRGB(0, 0, width, height, segpixels, 0, width);

		// SAVE BUFFERED IMAGE(S) AS PNG
//			ImageIO.write(segImage, "png", new File(outputFile));

			return ImageProcessingUtils.bufferdImg2Mat(segImage);
//			return segImage;
    }
    




public float getColorRadius()
{
	return colorRadius;
}

public void setColorRadius(float colorRadius)
{
	this.colorRadius = colorRadius;
}

public int getSpatialRadius()
{
	return spatialRadius;
}

public void setSpatialRadius(int spatialRadius)
{
	this.spatialRadius = spatialRadius;
}

public int getMinRegion()
{
	return minRegion;
}

public void setMinRegion(int minRegion)
{
	this.minRegion = minRegion;
}


}
