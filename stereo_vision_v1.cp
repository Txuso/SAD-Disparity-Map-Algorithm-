/*
 *  stereo_vision.cpp
 *  calibration
 *
 *  Created by Carlos and Josu 10/06/16.
 *
 */

//#include "ppl_c.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>


using namespace cv;
//using namespace concurrency;

int main(int argc, char** argv)
{
    
int max_shift;
clock_t tStart = clock();
// set size of the SAD window
int w_l, w_c, d, sad;
    
char* left_window = "Left Image";
char* right_window = "Right Image";
char* disparity_maps = "Disparity Maps";
Mat img_left, img_right, dst_left, dst_right, equ_map;
    
/// Load image
img_left = imread( argv[1], 1);
img_right = imread( argv[2], 1);
    
// load parameter
w_l = atoi(argv[3]);
w_c = atoi(argv[4]);
max_shift = atoi(argv[5]);
    

    if( !img_left.data || !img_right.data  )
    { std::cout<<"Usage: ./SteroVision_Demo <path_to_image>"<<std::endl;
    return -1;}
    
    /// Convert to grayscale
    cvtColor( img_left, img_left, CV_BGR2GRAY );
    cvtColor( img_right, img_right, CV_BGR2GRAY );

    Mat img_disp = img_left.clone();
    Mat img_rest = img_left.clone();
    
    int rows = img_left.rows; // number of rows
    int cols = img_left.cols; // number of colums

    int distance[rows][cols];
    int minSAD[rows][cols];
    

    for(int j = 0; j < rows; j++)
    {   for(int w = 0; w < cols; w++)
        {
            minSAD [j][w] = 999999;
            //img_disp.at<uchar>(j,w) = 0;
        }
    }
    

    for(int i = 0; i < max_shift; i++)
    {
        //std::cout << "New Shift: " << i << std::endl;

        for(int j = 0; j < rows; j++)
        {
           
            for(int w = 0; w < cols; w++)
            {

                    if (int(max_shift/2 <= w))
                        {d = w - max_shift/2;}
                    else
                    {
                        if (int(max_shift/4 <= w))
                        {d = w - max_shift/4;}

                        else
                        {
                            if (int(max_shift/10 <= w))
                            {d = w - max_shift/10;}
                            else
                            {d = w;}
                        }
                    }
                
                    
                    if (d + i > cols)
                    {
                        d=abs(j-max_shift);
                    }
                        img_rest.at<uchar>(j,w) = abs(img_left.at<uchar>(j,w) - img_right.at<uchar>(j,d+i));
                        distance [j][w] = abs(w - d+i);
                
            }
        }
        
        
        for(int j = 0; j < rows; j++)
        {
            for(int w = 0; w < cols; w++)
            {
                sad = 0;
                for (int y = 0; y < w_l; y++)
                {
                    for (int z = 0; z < w_c; z++)
                    {
                        if (j+y <= rows && w+z <= cols)
                        {
                            sad = sad + img_rest.at<uchar>(j+y,w+z);
                        }
                    }
                }

                //save disparity
                if (sad < minSAD[j][w])
                {
                    minSAD[j][w] = sad;
                    img_disp.at<uchar>(j,w) = distance [j][w];
                    
                }
            }
        }
    }
    
    //std::cout<<img_disp;
    
    /// Apply Histogram Equalization
    equalizeHist( img_left, dst_left);
    //equalizeHist( img_right, dst_right);
    equalizeHist( img_disp, img_disp);
    
    /// Display results
    namedWindow( left_window, CV_WINDOW_AUTOSIZE );
    //namedWindow( right_window, CV_WINDOW_AUTOSIZE );
    namedWindow( disparity_maps, CV_WINDOW_AUTOSIZE );
    
    imshow( left_window, dst_left );
    //imshow( right_window, dst_right);
    imshow( disparity_maps, img_disp);
    
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"****************   STEREO VISION PROJECT   **************************"<<std::endl;
    std::cout <<"*************   GENERATION OF DISPARITY MAP    **********************"<<std::endl;
    std::cout <<"**************   Created by Carlos & Josu   *************************"<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    // Elapsed Time:
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"*******************STADISTICAL TABLE*********************************"<<std::endl;
    std::cout <<"CLOCKS        : "<< double(CLOCKS_PER_SEC)<<std::endl;
    std::cout <<"CLOCKS PER SEC: "<< float(clock())<<std::endl;
    std::cout <<"ELAPSED TIME  : "<< float(float(clock() - tStart)/float(CLOCKS_PER_SEC))<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    
    /// Wait until user exits the program
    waitKey(0);
    
    return 0;
}
