package thud.skypixel;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import thud.meanshift.Pymeanshift;
import thud.utils.CommonImageUtils;
import thud.utils.ImageProcessingUtils;


public class SkyPixelDetection
{
	public int validationColor1=0;
	public int validationColor2=0;
	public int validationColor3=255;
	CommonImageUtils commonUtil = new CommonImageUtils();
	ImageProcessingUtils utils = new ImageProcessingUtils();
	
	public static void main(String[] args)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		SkyPixelDetection g = new SkyPixelDetection();
		g.process();
	}
	
	public static String SOBEL_95 = "Sobel_95";
	public static String SOBEL_90 = "Sobel_90";
	public static String SOBEL_80 = "Sobel_80";
	public static String SOBEL_70 = "Sobel_70";
	public static String SOBEL_60 = "Sobel_60";
	public static String SOBEL_50 = "Sobel_50";
	public static String MEAN_7_8_300 = "Mean_7_8_300";
	public static String MEAN_3_6_100 = "Mean_3_6_100";
	public static String MEAN_5_7_210 = "Mean_5_7_210";
	public static String MEAN_7_6_100 = "Mean_7_6_100";
	public static String KMEANS_12 = "Kmeans_12";
	public static String KMEANS_6 = "Kmeans_6";
	public static String KMEANS_14 = "Kmeans_14";
	public static String SOBEL_CONTAINS = "Sobel"; 
	public static String MEAN_CONTAINS = "Mean_"; 
			
	public void process()
	{
		ArrayList<String> testClasses = new ArrayList<String>();
		testClasses.add(SOBEL_95);
		testClasses.add(SOBEL_90);
		testClasses.add(SOBEL_80);
		testClasses.add(SOBEL_70);
		testClasses.add(SOBEL_60);
		testClasses.add(SOBEL_50);	
		testClasses.add(MEAN_7_8_300);		
		testClasses.add(MEAN_3_6_100);
		testClasses.add(MEAN_5_7_210);
		testClasses.add(MEAN_7_6_100);
		testClasses.add(KMEANS_12);
		testClasses.add(KMEANS_6);		
		testClasses.add(KMEANS_14);
		
		ArrayList<String> testFiles = new ArrayList<String>();
		testFiles.add("examples/65/20130101_143305.jpg");
		testFiles.add("examples/65/20130101_200307.jpg");
		testFiles.add("examples/75/20130101_211257.jpg");
		testFiles.add("examples/75/20130101_234303.jpg");
		testFiles.add("examples/GSVCubic/0010_-37.8091287866_144.973076705_panorama.png");
		
		ArrayList<String> validationImages = new ArrayList<String>();
		validationImages.add("examples/65.png");
		validationImages.add("examples/65.png");
		validationImages.add("examples/75.png");
		validationImages.add("examples/75.png");
		validationImages.add("examples/GSVMarkedPanorama/0010_-37.8091287866_144.973076705_panorama.png");
			
		for (int i=0;i<testFiles.size();i++)
		{
			String file = testFiles.get(i);
			String validationFile = validationImages.get(i);
			for (String testClass : testClasses)
			{
				processFile(file, testClass, validationFile);
			}
		}
	}

	
	public void processFile(String file, String testClass, String validationImage
			)
	{
		int K=0;
	    double skyreq=0;
	    double H_uplimit=0;
	    double H_downlimit=0;
	    double L_lightness=0;
	    double L_gray=0;
	    double S_gray=0;
		
		if (commonUtil.verifyFileExists(validationImage))
		{
			
		}
		else
		{
			System.out.println("no verification image for " + validationImage);
			return;
		}
		
		String outputFile = file.replace(".png", "_p.png");
		String fileToProcess =  file;
		Mat image;
		if (file.contains("panorama"))
		{
			image = loadCubicAndConvert(file, outputFile);
			fileToProcess = outputFile;
		}
		else
		{
			image = Imgcodecs.imread(file);
		}
			if (testClass.contains(MEAN_CONTAINS))
			{	
			    System.out.println ("class: " + testClass + ": "+ fileToProcess);	
			    String completedImage = fileToProcess.replace(".png", "_ms_sky_mark.png")
			    								.replace(".jpg", "_ms_sky_mark.jpg");
				String markedImageName = completedImage.replace("examples/", "examples/output/" + testClass + "/");
				String croppedFilename = markedImageName.replace("_ms_sky_mark.png", "_cropped.png")
												.replace("_ms_sky_mark.jpg", "_cropped.jpg");
				String segFilename = markedImageName.replace("_ms_sky_mark.png", "_seg.png")
												.replace("_ms_sky_mark.jpg", "_seg.jpg");
				
				int newDirectoryPoint = markedImageName.lastIndexOf("/");
				String newDirectory = markedImageName.substring(0,newDirectoryPoint);
				commonUtil.createDirectory(newDirectory);
				
	            if (commonUtil.verifyFileExists(markedImageName))
	            {
	            	System.out.println ("skip "+ markedImageName);
	            	return;
	            }
				
				int spatial_radius = 7;
				float range_radius = 8f;
				int min_density = 300;
				if (testClass.equals(MEAN_7_8_300))
				{
					
				}
				else if (testClass.equals(MEAN_3_6_100))
				{
					spatial_radius = 3;
					range_radius = 6.0f;
					min_density = 100;
				}
				else if (testClass.equals(MEAN_5_7_210))
				{
					spatial_radius = 5;
					range_radius = 7.0f;
					min_density = 210;
				}
				else if (testClass.equals(MEAN_7_6_100))
				{
					spatial_radius = 7;
					range_radius = 6.0f;
					min_density = 100;
				}
		    	pymeanshiftcomb(image ,validationImage, markedImageName, croppedFilename, segFilename, testClass,
		    			skyreq, H_uplimit, H_downlimit, L_lightness, L_gray, S_gray, 
		    			spatial_radius, range_radius, min_density, file);
				return;   
			}	
			else if (testClass.contains(SOBEL_CONTAINS))
			{
				String testClassRemoveThreshold = testClass.substring(0, testClass.length()-3);		
	
			    String markedOutputFile = fileToProcess.replace(".png", "_" + testClass + "_marked.png")
						.replace(".jpg", "_" + testClass + "_marked.jpg")
						.replace("examples/", "examples/output/" + testClassRemoveThreshold + "/");
			    String sobelFile = markedOutputFile.replace(testClass + "_marked", "_sobel");
			    String probImageOutput = markedOutputFile.replace(testClass + "_marked", "_prob");
			    
				int newDirectoryPoint = markedOutputFile.lastIndexOf("/");
				String newDirectory = markedOutputFile.substring(0,newDirectoryPoint);
				commonUtil.createDirectory(newDirectory);
            
	            if (commonUtil.verifyFileExists(markedOutputFile))
	            {
	            	System.out.println ("skip "+ markedOutputFile);
	            	return;
	            }
	            
	            double sobelProbThreshold = 0.95;
	            if (testClass.equals(SOBEL_95))
	            {
	            	sobelProbThreshold = 0.95;
	            }
	            if (testClass.equals(SOBEL_90))
	            {
	            	sobelProbThreshold = 0.90;
	            }
	            if (testClass.equals(SOBEL_80))
	            {
	            	sobelProbThreshold = 0.80;
	            }
	            if (testClass.equals(SOBEL_70))
	            {
	            	sobelProbThreshold = 0.70;
	            }
	            if (testClass.equals(SOBEL_60))
	            {
	            	sobelProbThreshold = 0.60;
	            }
	            if (testClass.equals(SOBEL_50))
	            {
	            	sobelProbThreshold = 0.50;
	            }
	            
				Mat markedSobelImage = commonUtil.sobel(image, sobelFile, sobelProbThreshold, markedOutputFile, probImageOutput);
				

				output_evaluation(validationImage,markedSobelImage,
						file, testClass,
						skyreq, H_uplimit, H_downlimit, L_lightness, L_gray, S_gray
						);
				return;
			}
			else if (testClass.equals(KMEANS_12))
			{
					System.out.println ("class: "
							+ testClass
							+ ": "+ file);
				    K=12;
				    skyreq=0.7;
				    H_uplimit=0.75;
				    H_downlimit=0.3;
				    L_lightness=0.95;
				    L_gray=0.75;
				    S_gray=0.2;
			}
			else if (testClass.equals(KMEANS_14))
			{
					System.out.println ("class: "
							+ testClass
							+ ": "+ file);
				    K=14;
				    skyreq=0.6;
				    H_uplimit=0.75;
				    H_downlimit=0.3;
				    L_lightness=0.95;
				    L_gray=0.75;
				    S_gray=0.2;
			}
			else if (testClass.equals(KMEANS_6))
			{
					System.out.println ("class: "
							+ testClass
							+ ": "+ file);
				    K=6;
				    skyreq=0.4;
				    H_uplimit=0.75;
				    H_downlimit=0.3;
				    L_lightness=0.95;
				    L_gray=0.65;
				    S_gray=0.2;
	
			}
		    String completedImage = fileToProcess.replace(".png", "_sky_mark"+".png")
					.replace(".jpg", "_sky_mark"+".jpg")
					.replace("examples/", "examples/output/" + testClass + "/");
			int newDirectoryPoint = completedImage.lastIndexOf("/");
			String newDirectory = completedImage.substring(0,newDirectoryPoint);
			commonUtil.createDirectory(newDirectory);
			
	        if (commonUtil.verifyFileExists(completedImage))
	        {
	        	System.out.println ("skip "+ completedImage);
	        	return;
	        }
	
	
	        String kmeansFile = completedImage.replace("_sky_mark", "_kmeans");
	        String clusteredFilename = completedImage.replace("_sky_mark", "_clustered");
	        String hlsFilename = completedImage.replace("_sky_mark", "_hls");
	        
	        Mat HLS;
	        Mat kmeanoutval;
	        if (commonUtil.verifyFileExists(clusteredFilename))
	        {
	        	HLS = Imgcodecs.imread(hlsFilename); 
	        	kmeanoutval = Imgcodecs.imread(clusteredFilename);
	        }
	        else
	        {
				HLS=commonUtil.hls(image);						
	
				List<Mat> label2List=CommonImageUtils.K_means(image,K);
				kmeanoutval = label2List.get(0);
				for (Mat layer : label2List)
				{
					Core.add(kmeanoutval, layer, kmeanoutval);							
				}
				Imgcodecs.imwrite(clusteredFilename,kmeanoutval);										
				Imgcodecs.imwrite(hlsFilename,HLS);
	        }
	        						
			Mat data=commonUtil.mark_sky(kmeanoutval,image
					,skyreq,H_uplimit,H_downlimit,L_lightness,
					L_gray,S_gray,HLS,
					completedImage
					);
	
			output_evaluation(validationImage,data, 
					file, 
					testClass,
					skyreq, H_uplimit, H_downlimit, L_lightness, L_gray, S_gray
					);
					

	}
	
	

	
	public Mat loadCubicAndConvert(String file, String outputFile)
	{
		Mat cubic_image = Imgcodecs.imread(file, 1);  
		Mat result_data = convertToPanorama(cubic_image,640*2, 960);
		
		Imgcodecs.imwrite(outputFile, result_data);
		return result_data;
	}
	

	
	public Mat convertToPanorama(Mat source_texture, int output_width, int output_length)
	{
		int xPixel=0,xOffset=0,yPixel=0,yOffset=0;
	    int cube_width = 640;
	    int cube_height = 640;

	    Mat pano_to_cubic_mapping = new Mat(output_length, output_width, CvType.CV_8UC3);
	    for (int j=0;j<output_length;j++)
	    {
	        double v = 1 - 1.0*j/output_length;
	        double theta = v * Math.PI;

	        for (int i=0;i<output_width;i++)
	        {
	            double u = 1.0*i/output_width;
	            double phi = u*2*Math.PI;

	            double x = Math.sin(phi) * Math.sin(theta) * -1;
	            double y = Math.cos(theta);
	            double z = Math.cos(phi) * Math.sin(theta) * -1;

	            double a =  Math.max(
	            		Math.max(Math.abs(x), Math.abs(y)), 
	            		Math.abs(z));

	            double xa = x / a;
	            double ya = y / a;
	            double za = z / a;
	            

	            if (xa == 1)
	            {
	                xPixel = (int)((((za + 1)/2) - 1)*cube_width);
	                xOffset = 2 * cube_width;
	                yPixel = (int)(((ya+1)/2)*cube_height);
	                yOffset = cube_height;
	            }
	            else if (xa == -1)
	            {
	                xPixel = (int)(((za + 1) / 2) * cube_width);
	                xOffset = 0;
	                yPixel = (int)(((ya + 1) / 2) * cube_height);
	                yOffset = cube_height;
	            }
	            else if (ya == 1)
	            {
	                xPixel = (int)(((xa + 1) / 2) * cube_width);
	                xOffset = cube_width;
	                yPixel = (int)((((za + 1) / 2) - 1) * cube_height);
	                yOffset = cube_height * 2;
	            }
	            else if (ya == -1)
	            {
	                xPixel = (int)(((((xa + 1) / 2)) * cube_width));
	                xOffset = cube_width;
	                yPixel = (int)(((((za + 1) / 2)) * cube_height));
	                yOffset = 0;
	            }
	            else if (za == 1)
	            {
	                xPixel = (int)(((((xa + 1) / 2)) * cube_width));
	                xOffset = cube_width;
	                yPixel = (int)(((((ya + 1) / 2)) * cube_height));
	                yOffset = cube_height;
	            }
	            else if (za == -1)
	            {
	                xPixel = (int)(((((xa + 1) / 2) - 1) * cube_width));
	                xOffset = 3 * cube_width;
	                yPixel = (int)(((ya + 1) / 2) * cube_height);
	                yOffset = cube_height;
	            }
	            else
	            {
	                System.out.println("Something is wrong");
	            }

	            xPixel = Math.abs(xPixel);
	            yPixel = Math.abs(yPixel);

	            xPixel += xOffset;
	            yPixel += yOffset;
	            if(xPixel == 2560 || yPixel == 1920) //# the dimension of 6 google street images(640*640) stiched together into a cubic map
	            {
	                xPixel = xPixel -1;
	                yPixel = yPixel -1;
	            }
	            double[] pixelColor = source_texture.get(yPixel, xPixel);
	            pano_to_cubic_mapping.put(j,i, pixelColor);	

	        }
	    }

	    return pano_to_cubic_mapping;
	}

	public void output_evaluation(String validationImage, Mat data,
			String image,
			String testClass,
			double skyreq, double H_uplimit, double H_downlimit, double L_lightness, double L_gray, double S_gray
			)
	{
	//    """
	//    calculate recall and precision if manual marking is provided 
	//    manual marking is named as file = file.replace('.png',' - Copy.png')
	//    for example 1.png's manual marking is called 1 - Copy.png
	//    """
       int validationImageWidth=0 ;
       int validationImageHeight=0;
       int validationImageArea =0;
       Mat marked_pixel_values = null;
       double recall;
       double precision;
		
		if (commonUtil.verifyFileExists(validationImage))
		{
			marked_pixel_values = Imgcodecs.imread(validationImage);
			validationImageWidth = marked_pixel_values.width();
			validationImageHeight = marked_pixel_values.height();
			validationImageArea = validationImageWidth*validationImageHeight;
		}

	    int true_sky_counter=0;
	    int true_positive_counter=0;
	    int false_positive_counter=0;

	    //count the manual marked sky pixels
	    for (int j=0;j<validationImageWidth;j++)
	    {
	    	for (int i=0;i<validationImageHeight;i++)
	    	{
	    		double[] pixelColor = marked_pixel_values.get(i, j);
	    		double[] pixelColorOfData = data.get(i, j);

	    		if (pixelColor[0] == validationColor1 && pixelColor[1] == validationColor2 && pixelColor[2] == validationColor3)
	    		{
	                true_sky_counter+=1;
	                if (pixelColorOfData[0] == 255 && pixelColorOfData[1] == 0 && pixelColorOfData[2] == 0)
	                {
	                    true_positive_counter+=1; //true positive counter
	                }
	    		}
	            else
	            {
	            	if (pixelColorOfData[0] == 255 && pixelColorOfData[1] == 0 && pixelColorOfData[2] == 0)
	            	{
	                    false_positive_counter+=1; //false positive counter
	            	}
	            }
	    	}
	    }

	    System.out.println ("true_sky_counter "+true_sky_counter);
	    System.out.println ("true_positive_counter "+true_positive_counter);
	    System.out.println ("false_positive_counter "+false_positive_counter);
	    if (true_sky_counter==0)
	    {
	    	recall=0;
	        System.out.println ("recall rate: 0");
	    }
	    else
	    {
	        recall = 1.0*true_positive_counter/true_sky_counter;
	        System.out.println ("Recall rate: "+ recall);
	    }
	    if (true_positive_counter==0)
	    {
	    	precision=0;
	        System.out.println ("precision rate: 0");
	    }
	    else
	    {
	        precision = 1.0*true_positive_counter/(true_positive_counter+false_positive_counter);
	        System.out.println ("precision rate: "+ precision);
	    }
	    double hsl_acc = (1.0*true_positive_counter+false_positive_counter)/validationImageArea;
	    System.out.println ("hsl_acc "+hsl_acc);
	    double manual_acc = 1.0*true_sky_counter/validationImageArea;
	    System.out.println ("manual_acc " +manual_acc);
	    System.out.println ("program marked proportion: "+ hsl_acc);
	    System.out.println ("manually marked proportion: "+ manual_acc);
	    
	    

	}


	public void pymeanshiftcomb(Mat origin, 
			String validationImage, String markedImageName,
			String croppedFilename, String segFilename,
			String testClass,
			double skyreq, double H_uplimit, double H_downlimit, double L_lightness, double L_gray, double S_gray,
			int spatial_radius, float range_radius, int min_density, String filename)
	{

		
		HashSet<String> exclusionColors = new HashSet<String>();		
		double[] BLUE = new double[]{255.0,0,0};
	
	    Mat image = origin;
	    
	    Mat original_image;
	    if (commonUtil.verifyFileExists(croppedFilename))
	    {
	    	original_image = Imgcodecs.imread(croppedFilename);
	    }
	    else
	    {
	        Imgcodecs.imwrite(croppedFilename, image);
			original_image = Imgcodecs.imread(croppedFilename);
	    }
	    
	    Mat segmented_image;
	    if (commonUtil.verifyFileExists(segFilename))
	    {
	    	segmented_image = Imgcodecs.imread(segFilename);
	    }
	    else
	    {
			Pymeanshift py = new Pymeanshift(spatial_radius, range_radius, min_density);
			segmented_image = py.pms_segment(ImageProcessingUtils.Mat2BufferedImage(original_image),
					range_radius, spatial_radius, min_density);
			Imgcodecs.imwrite(segFilename, segmented_image);
	    }
	    
		for (int j=0;j<segmented_image.width();j++)
		{
	    	double[] pixelColor = segmented_image.get(segmented_image.height()-1, j);
	    	String pixelStr = pixelColor[0]+"_"+pixelColor[1]+"_"+pixelColor[2];
	    	exclusionColors.add(pixelStr);
		}
				
		TreeMap<String,Integer> counterclass_all=new TreeMap<String,Integer>();
		for (int j=0;j<segmented_image.width();j++)
		{
		    for (int i=0;i<segmented_image.height()/2+1;i++)
		    {
		    	double[] pixelColor = segmented_image.get(i, j);
		    	String pixelStr = pixelColor[0]+"_"+pixelColor[1]+"_"+pixelColor[2];
		    	Integer count = counterclass_all.get(pixelStr);
		    	if (count == null)
		    	{
		    		count = 1;
		    	}
		    	else
		    	{
		    		count ++;
		    	}
		    	counterclass_all.put(pixelStr, count);
		    }
		}
		int max=0;
		String most_common_colourStr="";
	
		Set<String> keys = counterclass_all.keySet();
		for (String key : keys)
		{
			Integer count = counterclass_all.get(key);
			if (count>max)
			{
				
				max=count;
				most_common_colourStr=key;
			}
		}
		String[] most_common_colourStrSplit = most_common_colourStr.split("_");
		double[] most_common_colour = new double[]{new Double(most_common_colourStrSplit[0]).doubleValue(),
													new Double(most_common_colourStrSplit[1]).doubleValue(),
													new Double(most_common_colourStrSplit[2]).doubleValue()};

	

		Mat data = origin.clone();

		for (int j=0;j<segmented_image.width();j++)
		{
			 for (int i=0;i<segmented_image.height();i++)
			{
				double[] pixelColor = segmented_image.get(i, j);
			    if (pixelColor[0]==most_common_colour[0] && 
			    	pixelColor[1]==most_common_colour[1] &&
			    	pixelColor[2]==most_common_colour[2])
			    {
			    	data.put(i, j, BLUE);		
			    }
			    else
			    {
			    	continue;
			    }
			}
		}

		Imgcodecs.imwrite(markedImageName, data);

		String tempfile = validationImage;
        int validationImageWidth=0 ;
        int validationImageHeight=0;
        int validationImageArea =0;
        Mat marked_pixel_values = null;
		if (commonUtil.verifyFileExists(tempfile))
		{
			marked_pixel_values = Imgcodecs.imread(tempfile);
            validationImageWidth = marked_pixel_values.width();
            validationImageHeight = marked_pixel_values.height();
            validationImageArea = validationImageWidth*validationImageHeight;
		}
	
			int true_sky_counter=0;
			int true_positive_counter=0;
			int false_positive_counter=0;
	
			for (int j=0;j<validationImageWidth;j++)
			{
				for (int i=0;i<validationImageHeight;i++)
			    {
					double[] pixelColor = marked_pixel_values.get(i, j);
					double[] pixelColorOfData = data.get(i, j);
					if (pixelColor[0] == validationColor1 && pixelColor[1] == validationColor2 && pixelColor[2] == validationColor3)
					{
					    true_sky_counter+=1;
					    if (pixelColorOfData[0] == 255 && pixelColorOfData[1] == 0 && pixelColorOfData[2] == 0)
					    {
					        true_positive_counter+=1;
					    }
					}
					else
					{
					    if (pixelColorOfData[0] == 255 && pixelColorOfData[1] == 0 && pixelColorOfData[2] == 0)
					    {
					        false_positive_counter+=1;
					    }
					}
			    }
			}
			System.out.println ("true_sky_counter " +true_sky_counter);
			System.out.println ("true_positive_counter "+true_positive_counter);
			System.out.println ("false_positive_counter "+false_positive_counter);
            double recall = 1.0*true_positive_counter/true_sky_counter;
			System.out.println ("recall "+recall);
			double avg_preValue = 1.0*true_positive_counter/(true_positive_counter+false_positive_counter);
			System.out.println ("avg_preValue "+avg_preValue);
	        double hst_accValue = (1.0*true_positive_counter+false_positive_counter)/validationImageArea;
			System.out.println ("hst_accValue "+hst_accValue);
			double manual_accValue = 1.0*true_sky_counter/validationImageArea;

			System.out.println ("meanshift program marked proportion: "+ hst_accValue);
			System.out.println ("manually marked proportion: "+manual_accValue);
	
	}

}
