/*
 *  stereo_vision.cpp
 *
 *
 *  Created by Carlos and Josu 10/06/16.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"

#define _USE_MATH_DEFINES
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range2d.h"
#include <stddef.h>


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>


using namespace cv;
using namespace tbb;

//******************** inicializar en paralelo **************************************
class ini_en_paralelo {
    int **D;
    int **pruemat;


public:
    ini_en_paralelo(int **D_, int **pruemat_) : D(D_),pruemat(pruemat_) {}
    void operator() (const blocked_range2d <int,int>& r) const
    {   new int (**D);
        new int (**pruemat);
        for (int i = r.rows().begin(); i<r.rows().end();++i)
        {
            for (int j = r.cols().begin(); j<r.cols().end();++j)
            {
                D[i][j] = 999999;
            }
        }
    }

};


void ini_parallel(int **D, int **pruemat, int M, int N){
    parallel_for( blocked_range2d<int,int>(0, M, 20, 0, N, 20),
                ini_en_paralelo(D,pruemat) );
}

//********************* absolute differences in parallel *********************************************

class rest_en_paralelo {
    int **S;    int **D;    Mat L;    Mat R;    int ct;    int N;    int mx;
    
public:
    rest_en_paralelo(int **S_, Mat L_, Mat R_,int ct_,int mx_, int N_,int **D_) :
    S(S_),L(L_),R(R_),ct(ct_),mx(mx_),N(N_),D(D_) {}
    void operator() (const blocked_range2d <int,int>& r) const
    {
        for (int i = r.rows().begin(); i<r.rows().end();++i)
        {
            for (int j = r.cols().begin(); j<r.cols().end();++j)
            {
                
                int d=0, temp=0;
                
                if (int(mx/2 <= j))
                    {d = j - mx/2;}
                else
                {
                    if (int(mx/4 <= j))
                    {d = j - mx/4;}
                    else
                    {
                        if (int(mx/10 <= j))
                        {d = j - mx/10;}
                        else
                        {d = j;}
                    }
                }
               temp = d+ct;
                if (temp > N)
                {
                    d=abs(j-mx);
                }
                S[i][j] = abs(L.at<uchar>(i,j) - R.at<uchar>(i,temp));
                D[i][j] = abs(j - temp);
                
            }
        }
    }
    
};

void rest_parallel(int **S, Mat L,Mat R, int M, int N, int ct,int mx,int **D){
    parallel_for( blocked_range2d<int,int>(0, M, 20, 0, N, 20),
                 rest_en_paralelo(S,L,R,ct,mx,N,D) );
}

    
//************************ sums in parallel *************************************

class sad_en_paralelo {
    int **mSAD;
    int **disp;
    int **dt;
    int wl;
    int wc;
    int M;
    int N;
    int **rest;
    
public:
    sad_en_paralelo(int **mSAD_,int **disp_,int **dt_, int wl_,int wc_, int M_,int N_,int **rest_) :
    mSAD(mSAD_),disp(disp_),dt(dt_),wl(wl_),wc(wc_),M(M_),N(N_),rest(rest_) {}
    void operator() (const blocked_range2d <int,int>& r) const
    {
        for (int i = r.rows().begin(); i<r.rows().end();++i)
        {
            for (int j = r.cols().begin(); j<r.cols().end();++j)
            {
                int sad = 0;
                for (int y = 0; y < wl; y++)
                {
                    for (int z = 0; z < wc; z++)
                    {
                        if (i+y <= M && j+z <= N)
                            
                        {
                            sad = sad + rest[i+y][j+z];
                        }
                    }
                }
                
                //save disparity
                if (sad < mSAD[i][j])
                {
                    mSAD[i][j] = sad;
                    disp[i][j] = dt[i][j];
                    
                }
            }
        }
    }
};

void sad_parallel(int **mSAD,int **disp,int **dt, int wl,int wc, int M,int N,int **rest){
    parallel_for( blocked_range2d<int,int>(0, M, 20, 0, N, 20),
                sad_en_paralelo(mSAD,disp,dt,wl,wc,M,N,rest) );
        }

//********************************* Main **********************************************

int main(int argc, char** argv)
{

int max_shift;
clock_t tStart = clock();
// set size of the SAD window
int w_l, w_c, d, sad;
int yy = 100;
float xx[yy];


char* left_window = "Left Image";
char* right_window = "Right Image";
char* disparity_maps = "Disparity Maps";
Mat img_left,img_right, dst_left, dst_right, equ_map;
    
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
    //Mat img_rest = img_left.clone();
    
    int rows = img_left.rows; // number of rows
    int cols = img_left.cols; // number of colums
    int **minSAD, **img_rest;
    int **distance;
    int **dispa;
    
    minSAD=(int **)calloc(rows,sizeof(int *)); // se reserva memoria  para la matriz de n filas que contiene direcciones de memoria a las segundas dimensiones.
    img_rest=(int **)calloc(rows,sizeof(int *));
    distance=(int **)calloc(rows,sizeof(int *));
    dispa=(int **)calloc(rows,sizeof(int *));
    
    for (int i=0;i<cols;i++)
    {
        minSAD[i]=(int *)calloc(cols,sizeof(int *)); // se reserva memoria para las segundas dimensiones(columnas)
        img_rest[i] =(int *)calloc(cols,sizeof(int *));
        distance[i]=(int *)calloc(cols,sizeof(int *));
        dispa[i]=(int *)calloc(cols,sizeof(int *));
    }

    ini_parallel(minSAD,img_rest,rows,cols);
    
    for(int i = 0; i < max_shift; i++)
    {
        rest_parallel(img_rest, img_left,img_right, rows, cols,i,max_shift,distance);
        sad_parallel(minSAD,dispa,distance, w_l,w_c, rows,cols,img_rest);

    }
    
    
    for(int j = 0; j < rows; j++)
    {
        for(int w = 0; w < cols; w++)
        {
            img_disp.at<uchar>(j,w) = dispa [j][w];
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
    std::cout <<"*****************  STADISTICAL TABLE   ******************************"<<std::endl;
    std::cout <<"CLOCKS        : "<< double(CLOCKS_PER_SEC)<<std::endl;
    std::cout <<"CLOCKS PER SEC: "<< float(clock())<<std::endl;
    std::cout <<"ELAPSED TIME  : "<< float(float(clock() - tStart)/float(CLOCKS_PER_SEC))<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    std::cout <<"*********************************************************************"<<std::endl;
    /// Wait until user exits the program
    waitKey(0);
    
    return 0;
}


