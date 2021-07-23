
//
// Created by will on 19-10-17.
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "Timer.h"
#include "tic_toc.h"
#include "CUDATSDFIntegrator.h"
#include "Reader.h"
#include "parameters.h"
#include "DataManager.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
using namespace std;

double gaussrand()
{
    static double V1, V2, S;

    static int phase = 0;
    double X;

    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

int
main()
{

    vector<float> pt_x;
    srandom((int)time(0));  // 产生随机种子  把0换成NULL也行
    for (int i = 0; i < 10000; i++)
    {
        pt_x.push_back((float)(gaussrand()));
        cout<<pt_x.back()<<endl;
    }


    int sum_test = 0;
    float mean_x = 0;
    float stdv_x = 0;
    vectorSumMean(pt_x, mean_x, stdv_x);
    float min = (float) mean_x - 3.0f*(stdv_x);
    float max = (float) mean_x + 3.0f*(stdv_x);
    cout << min << " " << max << " std:" << stdv_x << endl;

    for (int i = 0; i < pt_x.size(); ++i)
    {
        if (pt_x[i] > min && pt_x[i] < max)
        {
            sum_test++;
        }
    }
    std::cout << "  x percent : " << (double) sum_test / (double) pt_x.size() << std::endl;

}

#pragma clang diagnostic pop