/* $Id: ImageType.java,v 1.7 2014/12/19 23:23:32 yoda2 Exp $
 *
 * Tab Spacing = 4
 *
 * Copyright (c) 2002-2004, Brian E. Pangburn & Jonathan P. Ayo
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.  Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials
 * provided with the distribution.  The names of its contributors may not be
 * used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * This software is a partial port of the EDISON system developed by
 * Chris M. Christoudias and Bogdan Georgescu at the Robust Image
 * Understanding Laboratory at Rutgers University
 * (http://www.caip.rutgers.edu/riul/).
 *
 * EDISON is available from:
 * http://www.caip.rutgers.edu/riul/research/code/EDISON/index.html
 *
 * It is based on the following references:
 *
 * [1] D. Comanicu, P. Meer: "Mean shift: A robust approach toward feature
 *     space analysis". IEEE Trans. Pattern Anal. Machine Intell., May 2002.
 *
 * [2] P. Meer, B. Georgescu: "Edge detection with embedded confidence".
 *     IEEE Trans. Pattern Anal. Machine Intell., 28, 2001.
 *
 * [3] C. Christoudias, B. Georgescu, P. Meer: "Synergism in low level vision".
 *     16th International Conference of Pattern Recognition, Track 1 - Computer
 *     Vision and Robotics, Quebec City, Canada, August 2001.
 *
 * The above cited papers are available from:
 * http://www.caip.rutgers.edu/riul/research/robust.html
 *
 */



package com.greatmindsworking.EDISON.segm;



/**
 * ImageType.java
 *
 * Defines enumerations for EDISON image types (color/grayscale).
 * <pre>
 * To-do:
 * 1. Add JavaDoc for public methods and data members.
 *
 * @author	$Author: yoda2 $
 * @version	$Revision: 1.7 $
 */
public class ImageType {

	private final String name;

	private ImageType(String name) { this.name = name; }

	@Override
	public String toString() { return name; }

	public static final ImageType GRAYSCALE = new ImageType("grayscale");
	public static final ImageType COLOR = new ImageType("color");

} // end ImageType class



/*
 * $Log: ImageType.java,v $
 * Revision 1.7  2014/12/19 23:23:32  yoda2
 * Cleanup of misc compiler warnings. Made EDISON GFunction an abstract class.
 *
 * Revision 1.6  2004/02/25 21:59:22  yoda2
 * Updated copyright notice.
 *
 * Revision 1.5  2003/11/24 16:20:10  yoda2
 * Updated copyright to 2002-2003.
 *
 * Revision 1.4  2002/12/11 23:01:58  yoda2
 * Initial migration to SourceForge.
 *
 * Revision 1.3  2002/09/20 19:49:00  bpangburn
 * Fixed various JavaDoc error messages.
 *
 * Revision 1.2  2002/09/20 19:15:15  bpangburn
 * Added BSD-style license, cleaned up JavaDoc, and moved CVS log to end of each source file.
 *
 * Revision 1.1  2002/06/26 13:36:00  bpangburn
 * Initial CVS commit after porting EDISON segmentation code from C++ to Java.
 *
 */