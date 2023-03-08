package thud.utils;

import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImageProcessingUtils
{
	public final static int OPENCV_RED = 2;
	public final static int OPENCV_GREEN = 1;
	public final static int OPENCV_BLUE = 0;

	String tmpDir = "/tmp/";

	@Deprecated
	public Mat pymeanshift(String origFile)
	{
		ProcessBuilder pb = new ProcessBuilder("/bin/sh",
				"/home/kerryn/git/2018-03-MasterITProject/ProcessGSVImages/pymeanshift.sh", origFile);
		Process p = null;
		try
		{
			p = pb.start();
			try
			{
				p.waitFor();
			}
			catch (InterruptedException e)
			{

				e.printStackTrace();
			}

		}
		catch (IOException e1)
		{

			e1.printStackTrace();
		}
		Mat shifted = Imgcodecs.imread("/tmp/testpymeanshift.png", 1);
		return shifted;
	}

	private static BufferedReader getOutput(Process p)
	{
		return new BufferedReader(new InputStreamReader(p.getInputStream()));
	}

	@Deprecated
	public Mat pymeanshift(String origFile, String filename, float colorRadius, int spatialRadius, int minRegion)
	{
		ProcessBuilder pb = new ProcessBuilder("/bin/sh",
				"/home/kerryn/git/2018-03-MasterITProject/ProcessGSVImages/pymeanshift2.sh", origFile, filename,
				"" + colorRadius, "" + spatialRadius, "" + minRegion);
		Process p = null;
		try
		{
			p = pb.start();
			BufferedReader output = getOutput(p);
			String ligne = "";
			while ((ligne = output.readLine()) != null)
			{
				System.out.println(ligne);
			}
			InputStream error = p.getErrorStream();
			for (int i = 0; i < error.available(); i++)
			{
				System.out.println("" + error.read());
			}
			try
			{
				p.waitFor();
			}
			catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}
		catch (IOException e1)
		{
			e1.printStackTrace();
		}
		Mat shifted = Imgcodecs.imread(filename, 1);
		return shifted;
	}

	public Mat[] getCubicImages(Mat orig)
	{
		int rowStart = 640;
		int rowEnd = 1280;
		int colStart = 0;
		int colEnd = 640;
		Mat sub1 = orig.submat(rowStart, rowEnd, colStart, colEnd);

		rowStart = 640;
		rowEnd = 1280;
		colStart = 640;
		colEnd = 1280;
		Mat sub2 = orig.submat(rowStart, rowEnd, colStart, colEnd);

		rowStart = 640;
		rowEnd = 1280;
		colStart = 1280;
		colEnd = 1920;
		Mat sub3 = orig.submat(rowStart, rowEnd, colStart, colEnd);

		rowStart = 640;
		rowEnd = 1280;
		colStart = 1920;
		colEnd = 2560;
		Mat sub4 = orig.submat(rowStart, rowEnd, colStart, colEnd);

		rowStart = 0;
		rowEnd = 640;
		colStart = 640;
		colEnd = 1280;
		Mat sub5 = orig.submat(rowStart, rowEnd, colStart, colEnd);

		rowStart = 1280;
		rowEnd = 1920;
		colStart = 640;
		colEnd = 1280;
		Mat sub6 = orig.submat(rowStart, rowEnd, colStart, colEnd);

		Mat[] returnValues = new Mat[6];
		returnValues[0] = sub1;
		returnValues[1] = sub2;
		returnValues[2] = sub3;
		returnValues[3] = sub4;
		returnValues[4] = sub5;
		returnValues[5] = sub6;

		return returnValues;
	}

	public Mat[] getCubicImages(String origFile)
	{
		Mat orig = Imgcodecs.imread(origFile, 1);
		return getCubicImages(orig);
	}

	public Mat getSobelImage(Mat image)
	{
		Mat src = image.clone();
		Mat src_gray = image.clone();
		Mat grad = image.clone();

		Mat grad_x = image.clone();
		Mat grad_y = image.clone();
		Mat abs_grad_x = image.clone();
		Mat abs_grad_y = image.clone();

		int scale = 1;
		int delta = 0;
		int ddepth = CvType.CV_16S;
		org.opencv.core.Size size = new org.opencv.core.Size(3, 3);

		Imgproc.GaussianBlur(src, src, size, 0, 0, Core.BORDER_DEFAULT);
		Imgproc.cvtColor(src, src_gray, Imgproc.COLOR_RGB2GRAY);

		// Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT
		// );
		Imgproc.Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, Core.BORDER_DEFAULT);
		Core.convertScaleAbs(grad_x, abs_grad_x);

		// Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT
		// );
		Imgproc.Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, Core.BORDER_DEFAULT);
		Core.convertScaleAbs(grad_y, abs_grad_y);

		Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		return grad;
	}

	// this is equation 2 of Wang 2015 (An Efficient Sky Detection Algorithm
	// Based on Hybrid Probability Model)
	public double findOptCriterion(Mat m1, int width, int height, int step)
	{
		Rect roi = new Rect(0, 0, width, height / 2);
		Mat skyPixels2 = new Mat(m1, roi);

		Rect roi2 = new Rect(0, height / 2, width, height / 2);
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

	public int[] splitSkyGrounds(Mat grad, double t, int[] borderOld, int width, int height)
	{
		int[] border = borderOld;
		for (int x = 0; x < width; x++)
		{
			for (int y = 1; y < height; y++)
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

	public double findJMax(String file, int width, int height, int step)
	{
		Mat m1 = Imgcodecs.imread(file);
		return findJMax(m1, width, height, step);
	}

	public static Mat bufferdImg2Mat(BufferedImage in)
	{
		Mat out;
		byte[] data;
		int r, g, b;
		int height = in.getHeight();
		int width = in.getWidth();
		if (in.getType() == BufferedImage.TYPE_INT_RGB || in.getType() == BufferedImage.TYPE_INT_ARGB)
		{
			out = new Mat(height, width, CvType.CV_8UC3);
			data = new byte[height * width * (int) out.elemSize()];
			int[] dataBuff = in.getRGB(0, 0, width, height, null, 0, width);
			for (int i = 0; i < dataBuff.length; i++)
			{
				data[i * 3 + 2] = (byte) ((dataBuff[i] >> 16) & 0xFF);
				data[i * 3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
				data[i * 3] = (byte) ((dataBuff[i] >> 0) & 0xFF);
			}
		}
		else
		{
			out = new Mat(height, width, CvType.CV_8UC1);
			data = new byte[height * width * (int) out.elemSize()];
			int[] dataBuff = in.getRGB(0, 0, width, height, null, 0, width);
			for (int i = 0; i < dataBuff.length; i++)
			{
				r = (byte) ((dataBuff[i] >> 16) & 0xFF);
				g = (byte) ((dataBuff[i] >> 8) & 0xFF);
				b = (byte) ((dataBuff[i] >> 0) & 0xFF);
				data[i] = (byte) ((0.21 * r) + (0.71 * g) + (0.07 * b)); // luminosity
			}
		}
		out.put(0, 0, data);
		return out;
	}

	public static BufferedImage Mat2BufferedImage(Mat matrix)
	{
		MatOfByte mob = new MatOfByte();
		Imgcodecs.imencode(".png", matrix, mob);
		byte ba[] = mob.toArray();

		BufferedImage bi = null;
		try
		{
			bi = ImageIO.read(new ByteArrayInputStream(ba));
		}
		catch (IOException e)
		{

			e.printStackTrace();
		}
		return bi;
	}

	public static Mat bufferedImageToMat(BufferedImage bi)
	{
		String tmpDir = "/tmp/";
		String filename = "temp.png";

		saveImage(tmpDir + filename, bi);
		Mat matImage = Imgcodecs.imread(tmpDir + filename, 1);
		return matImage;
	}

	/**
	 * @param filename
	 * @param image
	 */
	public static void saveImage(String filename, BufferedImage image)
	{
		File file = new File(filename);
		try
		{
			ImageIO.write(image, "png", file);
		}
		catch (Exception e)
		{
			System.out.println(e.toString() + " Image '" + filename + "' saving failed.");
		}
	}

	public Mat negativeToZero(Mat orig)
	{
		Mat returnArray = orig.clone();
		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = pixel[0];
				if (r < 0)
				{
					r = 0.0;
				}
				returnArray.put(i, j, new double[]
				{ r });
			}
		}
		return returnArray;
	}

	public Mat divide(Mat orig, double divisor)
	{
		Mat returnValue = Mat.zeros(orig.height(), orig.width(), CvType.CV_32FC3);

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = (pixel[0] / divisor);
				double g = (pixel[1] / divisor);
				double b = (pixel[2] / divisor);
				returnValue.put(i, j, new double[]
				{ r, g, b });
			}
		}

		return returnValue;
	}

	public Mat multiply(Mat orig, double divisor)
	{
		Mat returnValue = Mat.zeros(orig.height(), orig.width(), CvType.CV_32FC3);

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = (pixel[0] * divisor);
				double g = (pixel[1] * divisor);
				double b = (pixel[2] * divisor);
				returnValue.put(i, j, new double[]
				{ r, g, b });
			}
		}

		return returnValue;
	}

	public Mat multiply1Pixel(Mat orig, double divisor)
	{
		Mat returnValue = orig.clone();

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = (pixel[0] * divisor);

				returnValue.put(i, j, new double[]
				{ r });
			}
		}

		return returnValue;
	}

	public Mat subtract1Pixel(Mat orig, Mat array2)
	{
		Mat returnValue = orig.clone();
		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double[] pixel2 = array2.get(i, j);
				double r = (pixel[0] - pixel2[0]);
				returnValue.put(i, j, new double[]
				{ r });
			}
		}
		return returnValue;
	}

	public Mat multiply1Pixel(Mat orig, Mat array2)
	{
		Mat returnValue = orig.clone();
		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double[] pixel2 = array2.get(i, j);
				double r = (pixel[0] * pixel2[0]);
				returnValue.put(i, j, new double[]
				{ r });
			}
		}
		return returnValue;
	}

	public Mat divide1Pixel(Mat orig, Mat array2)
	{
		Mat returnValue = orig.clone();
		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double[] pixel2 = array2.get(i, j);
				double r = (pixel[0] / pixel2[0]);
				returnValue.put(i, j, new double[]
				{ r });
			}
		}
		return returnValue;
	}

	public Mat subtract(double divisor, Mat orig)
	{
		Mat returnValue = Mat.zeros(orig.height(), orig.width(), CvType.CV_32FC3);

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);

				double r = divisor - pixel[0];
				double g = divisor - pixel[1];
				double b = divisor - pixel[2];
				returnValue.put(i, j, new double[]
				{ r, g, b });
			}
		}

		return returnValue;
	}

	public ArrayList<Integer> findNanNegInf(Mat orig)
	{
		ArrayList<Integer> returnValue = new ArrayList<Integer>();

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);

				double r = pixel[0];
				if (Double.isNaN(r) || Double.isInfinite(r))
				{
					returnValue.add(i);
				}

			}
		}

		return returnValue;
	}

	public double findMax(Mat orig)
	{
		double returnValue = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = (pixel[0]);
				if (r > returnValue)
				{
					returnValue = r;
				}
				if (Double.isNaN(r))
				{
					return Double.NaN;
				}

			}
		}
		return returnValue;
	}

	public Mat subtract1Pixel(double divisor, Mat orig)
	{
		Mat returnValue = orig.clone();

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);

				double r = divisor - pixel[0];

				returnValue.put(i, j, new double[]
				{ r });
			}
		}

		return returnValue;
	}

	public Mat divide(Mat orig, double divisor, boolean singlePixel)
	{
		Mat returnValue = orig.clone();
		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = (pixel[0] / divisor);
				returnValue.put(i, j, new double[]
				{ r });
			}
		}
		return returnValue;
	}

	public Mat multiplySequencePlus1(Mat orig)
	{
		int count = 1;
		Mat returnValue = orig.clone();

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = (pixel[0]) * count;
				returnValue.put(i, j, new double[]
				{ r });
				count++;
			}
		}
		return returnValue;
	}

	public Mat cumsum(Mat orig)
	{
		double sum = 0.0;
		Mat returnValue = orig.clone();

		for (int i = 0; i < orig.height(); i++)
		{
			for (int j = 0; j < orig.width(); j++)
			{
				double[] pixel = orig.get(i, j);
				double r = (pixel[0]);
				sum += r;

				returnValue.put(i, j, new double[]
				{ sum });
			}
		}

		return returnValue;
	}

	public static BufferedImage bufferedImage(Mat m)
	{
		int type = BufferedImage.TYPE_BYTE_GRAY;
		if (m.channels() > 1)
		{
			type = BufferedImage.TYPE_3BYTE_BGR;
		}
		BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
		// get all the pixels
		m.get(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData()); 
		return image;
	}

	public static void showResult2(Mat img)
	{

		BufferedImage bufImage = bufferedImage(img);
		try
		{
			JFrame frame = new JFrame();
			frame.setLayout(new FlowLayout());
			JLabel lbl = new JLabel();
			lbl.setIcon(new ImageIcon(bufImage));

			frame.add(lbl);
			frame.setVisible(true);

			frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

}
