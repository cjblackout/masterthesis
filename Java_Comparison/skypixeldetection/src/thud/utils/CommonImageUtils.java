package thud.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;



public class CommonImageUtils
{

	public boolean verifyFileExists(String filePathString)
	{
		boolean fileExists = false;
		File f = new File(filePathString);
		if (f.exists() && !f.isDirectory())
		{
			fileExists = true;
		}

		return fileExists;

	}

	public void writeFile(String text, String filename)
	{
		FileOutputStream out; // declare a file output object
		PrintStream p; // declare a print stream object

		try
		{
			out = new FileOutputStream(filename);
			p = new PrintStream(out);
			p.println(text);
			p.close();
		}
		catch (Exception e)
		{
			System.err.println("Error writing to file");
		}

	}

	public boolean createDirectory(String directory)
	{
		// Create multiple directories
		boolean success = (new File(directory)).mkdirs();
		if (success)
		{
			System.out.println("Directories: " + directory + " created");
		}

		return success;

	}

	public Mat hls(Mat image)
	{

		// Get a 2D array of an image so that one can access HLS[x][y]
		Mat HLS = image.clone();
		Imgproc.cvtColor(image, HLS, Imgproc.COLOR_RGB2HLS);
		return HLS;
	}
	
	@SuppressWarnings("unchecked")
	public String[] getDirectoryList(String directory, boolean sorted)
	{
		FilenameFilter filter = new FilenameFilter()
		{
			public boolean accept(File dir, String name)
			{
				boolean accept = true;
				if (name.contains("##check"))
				{
					accept = false;
				}
				else if (name.startsWith("."))
				{
					accept = false;
				}
				else if (name.contains("##Check"))
				{
					accept = false;
				}
				return accept;
			}
		};

		File dir = new File(directory);

		File files[] = dir.listFiles(filter);
		if (files == null || files.length < 1)
		{
			return new String[]
			{};
		}

		if (sorted)
		{
			Arrays.sort(files, new Comparator<File>()
			{

				@Override
				public int compare(File o1, File o2)
				{
					return new Long((o1).lastModified()).compareTo(new Long((o2).lastModified()));
				}
			});
		}

		String[] fileNames = new String[files.length];
		int count = 0;
		for (File file : files)
		{
			fileNames[count] = file.getName();

			count++;
		}

		return fileNames;

	}
	
	final static Charset ENCODING = StandardCharsets.UTF_8;

	public void appendFile(String text, String filename)
	{
		BufferedWriter bw = null;

		try
		{
			// APPEND MODE SET HERE
			bw = new BufferedWriter(new FileWriter(filename, true));
			bw.write(text);
			bw.newLine();
			bw.flush();
		}
		catch (IOException ioe)
		{
			ioe.printStackTrace();
		}
		finally
		{ // always close the file
			if (bw != null)
				try
				{
					bw.close();
				}
				catch (IOException ioe2)
				{
					// just ignore it
				}
		} // end try/catch/finally

	}
	
	public HashSet<String> readTextFileToArray(String filename)
	{
		HashSet<String> imagesToProcess = new HashSet<String>();

		Path path = Paths.get(filename);
		try (BufferedReader reader = Files.newBufferedReader(path, ENCODING))
		{
			String line = null;
			while ((line = reader.readLine()) != null)
			{
				imagesToProcess.add(line);
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}

		return imagesToProcess;
	}
	
	public Mat sobel(Mat image, String sobelOutputFile, double sobelProbThreshold, String markedOutputFile,
			String probImageOutput)
	{
		ImageProcessingUtils utils = new ImageProcessingUtils();
		double tMin = 5.0;
		double tMax = 200.0;
		int iterations = 3;

		Mat sobelSave;
		Mat markedSave;
		if (verifyFileExists(sobelOutputFile))
		{
			sobelSave = Imgcodecs.imread(sobelOutputFile, 1);
		}
		else
		{
			Mat sobel = utils.getSobelImage(image);
			sobelSave = saveAndReturn(sobel, sobelOutputFile);
		}

		if (verifyFileExists(markedOutputFile))
		{
			markedSave = Imgcodecs.imread(markedOutputFile, 1);
		}
		else
		{
			Mat marked = printBorder(sobelSave, image, tMin, tMax, iterations, sobelProbThreshold, probImageOutput);
			markedSave = saveAndReturn(marked, markedOutputFile);
		}

		return markedSave;

	}
	
	public Mat saveAndReturn(Mat image, String imgName)
	{

		Imgcodecs.imwrite(imgName, image);
		Mat converted = Imgcodecs.imread(imgName, 1);
		return converted;
	}

	public Mat printBorder(Mat grad, Mat orig, double tMin, double tMax, int iterations, double sobelProbThreshold,
			String probImageOutput)
	{
		Mat maskedOrig = orig.clone();
		double[] blueData = new double[3];
		blueData[0] = 255;
		blueData[1] = 0;
		blueData[2] = 0;

		int width = grad.width();
		int height = grad.height();

		if (verifyFileExists(probImageOutput))
		{
			System.out.println("reuse " + probImageOutput);
			Mat probImage = Imgcodecs.imread(probImageOutput, 1);
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					double[] pixel = probImage.get(j, i);
					double R = pixel[2];
					double G = pixel[1];
					double B = pixel[0];

					double pSky = R;

					if (pSky > sobelProbThreshold * 100.0)
					{
						maskedOrig.put(j, i, blueData);
					}
				}
			}
			return maskedOrig;
		}

		SummaryStatistics ssRSky = new SummaryStatistics();
		SummaryStatistics ssGSky = new SummaryStatistics();
		SummaryStatistics ssBSky = new SummaryStatistics();

		int step = 3;

		int[] border = findOptBoundary(grad, tMin, tMax, iterations, width, height);

		double[] redData = new double[3];
		redData[0] = 0;
		redData[1] = 0;
		redData[2] = 255;

		double[] whiteData = new double[3];
		whiteData[0] = 0;
		whiteData[1] = 0;
		whiteData[2] = 0;

		for (int i = 0; i < border.length; i++)
		{
			int y = border[i];
			int x = i;

			for (int j = 0; j < y; j++)
			{
				double[] pixel = orig.get(j, x);
				ssRSky.addValue(pixel[2]);
				ssGSky.addValue(pixel[1]);
				ssBSky.addValue(pixel[0]);

			}

		}

		double rMean = ssRSky.getMean();
		double rSd = ssRSky.getStandardDeviation();
		double gMean = ssGSky.getMean();
		double gSd = ssGSky.getStandardDeviation();
		double bMean = ssBSky.getMean();
		double bSd = ssBSky.getStandardDeviation();
		System.out.println("red " + rMean + " " + rSd);
		System.out.println("green " + gMean + " " + gSd);
		System.out.println("blue " + bMean + " " + bSd);
		double beta = 5.0;

		Mat probImage = maskedOrig.clone();

		for (int i = 0; i < border.length; i++)
		{

			for (int j = 0; j < height; j++)
			{
				double[] pixel = orig.get(j, i);
				double R = pixel[2];
				double G = pixel[1];
				double B = pixel[0];

				double rTerm = Math.pow((R - rMean) / (beta * rSd), 2);
				double gTerm = Math.pow((G - gMean) / (beta * gSd), 2);
				double bTerm = Math.pow((B - bMean) / (beta * bSd), 2);

				double pColor = Math.exp(-1 * (rTerm + gTerm + bTerm));

				double pPosition = Math.exp(-1 * Math.pow(1.0 * j / height, 2));

				double pSky = pColor * pPosition;

				double[] probColor = new double[]
				{ pSky * 100.0, pSky * 100.0, pSky * 100.0 };
				probImage.put(j, i, probColor);

				if (pSky > sobelProbThreshold)
				{
					maskedOrig.put(j, i, blueData);
				}

			}

		}
		System.out.println("generate " + probImageOutput);
		Imgcodecs.imwrite(probImageOutput, probImage);

		return maskedOrig;
	}
		
		
	// this is equation 2 of Wang 2015 (An Efficient Sky Detection Algorithm
	// Based on Hybrid Probability Model)
	public double findOptCriterion(Mat m1, int width, int height, int step)
	{
		Rect roi = new Rect(0, 0, width, height);
		Mat skyPixels2 = new Mat(m1, roi);

		Rect roi2 = new Rect(0, height, width, height);
		Mat groundPixels2 = new Mat(m1, roi2);

		Mat skyPixelsReshape = skyPixels2.reshape(1, skyPixels2.rows() * skyPixels2.cols());
		Mat groundPixelsReshape = groundPixels2.reshape(1, groundPixels2.rows() * groundPixels2.cols());

		int ctype = CvType.CV_8UC3;
		Mat covarGround = Mat.zeros(3, 3, CvType.CV_8UC3);
		Mat meanGround = Mat.zeros(3, 3, CvType.CV_8UC3);
		Mat covarSky = Mat.zeros(3, 3, CvType.CV_8UC3);
		Mat meanSky = Mat.zeros(3, 3, CvType.CV_8UC3);

		int flags = org.opencv.core.Core.COVAR_NORMAL | org.opencv.core.Core.COVAR_ROWS;

		Core.calcCovarMatrix(groundPixelsReshape, covarGround, meanGround, flags, ctype);
		Core.calcCovarMatrix(skyPixelsReshape, covarSky, meanSky, flags, ctype);

		double gamma = 2.0;
		double formula = gamma * Core.determinant(covarSky) + Core.determinant(covarGround);
		double smallValue = 0.0000001;
		double j = 1 / Math.max(formula, smallValue);

		return j;
	}

	public int[] findOptBoundary(Mat grad, double tMin, double tMax, int iterations, int width, int height)
	{
		int step = 1;
		int[] border = new int[width];
		int[] borderTmp = new int[width];
		double jMax = 0;
		int[] borderOld = new int[width];

		for (int i = 0; i < iterations; i++)
		{
			int n = i;
			double[] tReturn = getThreshold(tMin, tMax, iterations, n + 1);
			double t = tReturn[0];
			tMin = tReturn[1];
			tMax = tReturn[2];
			borderTmp = splitSkyGround(grad, t, borderOld, width, height);
			borderOld = borderTmp;

			double j = findJMax(grad, width, height, step);
			if (j > jMax)
			{
				jMax = j;
				border = borderTmp;
			}
		}
		return border;

	}

	public double findJMax(Mat m1, int width, int height, int step)
	{
		double jMax = 0.0;
		// assume that the horizon is half way down and only search that range
		for (int i = 1; i < height / 1; i++)
		{
			double j = findJn(m1, width, height / 1, step, i);

			if (j > jMax)
			{
				jMax = j;
			}
		}
		return jMax;

	}

	public double findJn(Mat m1, int width, int height, int step, int borderTmp)
	{
		int roi1Height = height - borderTmp;
		int roi2Height = height - roi1Height;

		Rect roi = new Rect(0, 0, width, roi1Height);
		Mat skyPixels2 = new Mat(m1, roi);
		Rect roi2 = new Rect(0, roi1Height, width, roi2Height);
		Mat groundPixels2 = new Mat(m1, roi2);
		Mat skyPixelsReshape = skyPixels2.reshape(1, skyPixels2.rows() * skyPixels2.cols());
		Mat groundPixelsReshape = groundPixels2.reshape(1, groundPixels2.rows() * groundPixels2.cols());
		int ctype = CvType.CV_8UC3;
		Mat covarGround = Mat.zeros(3, 3, CvType.CV_8UC3);
		Mat meanGround = Mat.zeros(3, 3, CvType.CV_8UC3);
		Mat covarSky = Mat.zeros(3, 3, CvType.CV_8UC3);
		Mat meanSky = Mat.zeros(3, 3, CvType.CV_8UC3);
		int flags = org.opencv.core.Core.COVAR_NORMAL | org.opencv.core.Core.COVAR_ROWS;
		Core.calcCovarMatrix(groundPixelsReshape, covarGround, meanGround, flags, ctype);
		Core.calcCovarMatrix(skyPixelsReshape, covarSky, meanSky, flags, ctype);
		double gamma = 2.0;
		double formula = gamma * Core.determinant(covarSky) + Core.determinant(covarGround);
		double smallValue = 0.001;
		double j = 1 / Math.max(formula, smallValue);
		return j;
	}

	public int[] splitSkyGround(Mat grad, double t, int[] borderOld, int width, int height)
	{
		int[] border = borderOld;
		for (int x = 0; x < width; x++)
		{
			for (int y = 1; y < height / 1; y++)
			{
				double tempT = grad.get(y, x)[0];
				if (tempT > t)
				{
					border[x] = y;

					break;
				}
			}
		}
		for (int i = 0; i < border.length; i++)
		{
			if (border[i] == 1 && i > 0)
			{
				border[i] = border[i - 1];
			}
		}

		return border;
	}

	public double[] getThreshold(double tMin, double tMax, int iterations, int n)
	{
		double[] returnValue = new double[3];
		double t = tMin + ((tMax - tMin) / (iterations - 1)) * (n - 1);
		if (t < tMin)
		{
			tMin = t;
		}
		if (t > tMax)
		{
			tMax = t;
		}
		returnValue[0] = t;
		returnValue[1] = tMin;
		returnValue[2] = tMax;
		return returnValue;
	}
		
	public double calcAverage(ArrayList<Double> array)
	{
		double sum = 0.0;
		double total = 0.0;

		for (int i = 0; i < array.size(); i++)
		{
			sum += array.get(i);
		}

		return total / array.size();
	}

	String tmpDir = "/tmp/";

	public Mat convertBySaving(Mat image)
	{
		String fileExtension = ".png";
		String filename = "convertImage";
		Imgcodecs.imwrite(tmpDir + filename + "2" + fileExtension, image);
		Mat converted = Imgcodecs.imread(tmpDir + filename + "2" + fileExtension, 1);
		return converted;
	}

	public Mat convertBySaving(Mat image, String tmpDir)
	{
		String fileExtension = ".png";
		String filename = "convertImage";
		Imgcodecs.imwrite(tmpDir + filename + "2" + fileExtension, image);
		Mat converted = Imgcodecs.imread(tmpDir + filename + "2" + fileExtension, 1);
		return converted;
	}

	public static List<Mat> K_means(Mat cutout, int k)
	{
		Mat samples = cutout.reshape(1, cutout.cols() * cutout.rows());
		Mat samples32f = new Mat();
		samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);
		Mat labels = new Mat();
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
		Mat centers = new Mat();
		Core.kmeans(samples32f, k, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centers);
		List<Mat> clusterList = showClusters(cutout, labels, centers);
		return clusterList;
	}

	private static List<Mat> showClusters(Mat cutout, Mat labels, Mat centers)
	{
		centers.convertTo(centers, CvType.CV_8UC1, 255.0);
		centers.reshape(3);
		List<Mat> clusters = new ArrayList<Mat>();
		for (int i = 0; i < centers.rows(); i++)
		{
			clusters.add(Mat.zeros(cutout.size(), cutout.type()));
		}
		Map<Integer, Integer> counts = new HashMap<Integer, Integer>();
		for (int i = 0; i < centers.rows(); i++)
			counts.put(i, 0);
		int rows = 0;
		for (int y = 0; y < cutout.rows(); y++)
		{
			for (int x = 0; x < cutout.cols(); x++)
			{
				int label = (int) labels.get(rows, 0)[0];
				int r = (int) centers.get(label, 2)[0];
				int g = (int) centers.get(label, 1)[0];
				int b = (int) centers.get(label, 0)[0];
				clusters.get(label).put(y, x, b, g, r);
				rows++;
			}
		}
		return clusters;
	}
				
	// K-means clustering to segment the image into K color class
	// filename_clustered(number of clusters).png is used as output
	public Mat K_means2(String file, int K)
	{

		String nametemp = file;

		file = file + "_cropped.png";
		Mat img = Imgcodecs.imread(file);

		Mat mHSV = img.clone();
		Imgproc.cvtColor(img, mHSV, Imgproc.COLOR_RGBA2RGB, 3);
		Imgproc.cvtColor(img, mHSV, Imgproc.COLOR_RGB2HSV, 3);
		List<Mat> hsv_planes = new ArrayList<Mat>(3);
		Core.split(mHSV, hsv_planes);

		Mat channel = hsv_planes.get(2);
		channel = Mat.zeros(mHSV.rows(), mHSV.cols(), CvType.CV_8UC1);
		hsv_planes.set(2, channel);
		Core.merge(hsv_planes, mHSV);

		Mat clusteredHSV = img.clone();
		mHSV.convertTo(mHSV, CvType.CV_32FC3);
		TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 10, 1.0);
		Core.kmeans(mHSV, K, clusteredHSV, criteria, 50, Core.KMEANS_PP_CENTERS);

		String clusteredFilename = nametemp + "_clustered" + K + ".png";
		Imgcodecs.imwrite(clusteredFilename, clusteredHSV);
		return clusteredHSV;

	}


	// mark out sky by using HSL color filtering
	public Mat mark_sky(Mat segmentedImage, Mat originalImage, 
//			String file, 
			double skyreq, double H_uplimit,
			double H_downlimit, double L_lightness, double L_gray, double S_gray, Mat HLS
//			,String fileWithoutExtension,
//			String newDir
			, String markedImageFile
			)
	{
		HashSet<String> exclusionColors = new HashSet<String>();
		double[] BLUE_COLOR = new double[]
		{ 255.0, 0, 0 };
		// dict for counting sky pixels with regard to each label(key:label
		// value: count)
		TreeMap<String, Integer> counterclass_skypre = new TreeMap<String, Integer>();
		// dict for counting all pixels with regard to each label(key:label
		// value: count)
		TreeMap<String, Integer> counterclass_all = new TreeMap<String, Integer>();
		int width = segmentedImage.width();
		int height = segmentedImage.height();
		int[][] Skypre = new int[height][width];

		// if the color is in the bottom row of the image, it probably isn't a
		// sky color
		for (int j = 0; j < segmentedImage.width(); j++)
		{
			double[] pixelColor = segmentedImage.get(segmentedImage.height() - 1, j);
			String pixelStr = pixelColor[0] + "_" + pixelColor[1] + "_" + pixelColor[2];

			exclusionColors.add(pixelStr);
		}

		for (int j = 0; j < segmentedImage.width(); j++)
		{
			for (int i = 0; i < segmentedImage.height(); i++)
			{
				double[] pixel = segmentedImage.get(i, j);
				String pixelStr = pixel[0] + "_" + pixel[1] + "_" + pixel[2];
				Integer count = counterclass_all.get(pixelStr);
				if (count == null)
				{
					count = 1;
				}
				else
				{
					count++;
				}
				counterclass_all.put(pixelStr, count);

				double[] hslPixel = HLS.get(i, j);

				int r = 2;
				int g = 1;
				int b = 0;

				// HSL colour filtering condition
				if (H_uplimit * 180 >= hslPixel[r] && hslPixel[r] >= H_downlimit * 180
						|| hslPixel[g] >= L_lightness * 255
						|| (hslPixel[g] > L_gray * 255 && hslPixel[b] <= S_gray * 255))

				{
					Skypre[i][j] = 1;
					Integer count2 = counterclass_skypre.get(pixelStr);
					if (count2 == null)
					{
						count2 = 1;
					}
					else
					{
						count2++;
					}
					counterclass_skypre.put(pixelStr, count2);
				}
			}
		}
		// list of sky labels
		HashSet<String> skylabel_list = new HashSet<String>();
		Set<String> keys = counterclass_skypre.keySet();
		for (String key : keys)
		{
			Integer countSkypre = counterclass_skypre.get(key);
			Integer countAll = counterclass_all.get(key);
			// if there is enough sky pixels in one cluster
			if (1.0 * countSkypre / countAll > skyreq)
			{
				skylabel_list.add(key);
			}
		}

		Mat result = originalImage.clone();
		for (int j = 0; j < segmentedImage.width(); j++)
		{
			for (int i = 0; i < segmentedImage.height(); i++)
			{
				double[] pixel = segmentedImage.get(i, j);
				String pixelStr = pixel[0] + "_" + pixel[1] + "_" + pixel[2];
				if (exclusionColors.contains(pixelStr))
				{

				}
				else if (skylabel_list.contains(pixelStr))
				{
					result.put(i, j, BLUE_COLOR);
				}
				else
				{
					continue;
				}
			}
		}
		Imgcodecs.imwrite(markedImageFile, result);

		return result;
	}	
								
				
				
				
				
				
}
