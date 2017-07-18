#ifndef KMEANS_HPP_
#define KMEANS_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <limits>
#include <string.h>


using namespace std;
template<typename Dtype>
void kmeans_cluster(/*vector<int> &*/int *cLabel, /*vector<Dtype> &*/Dtype *cCentro, Dtype *cWeights, int nWeights, int *mask/*vector<int> &mask*/, /*Dtype maxWeight, Dtype minWeight,*/  int nCluster,  int max_iter /* = 1000 */)
{
    //find min max
    Dtype maxWeight=numeric_limits<Dtype>::min(), minWeight=numeric_limits<Dtype>::max();
    for(int k = 0; k < nWeights; ++k)
    {
        if(mask[k])
        {
            if(cWeights[k] > maxWeight)
                maxWeight = cWeights[k];
            if(cWeights[k] < minWeight)
                minWeight = cWeights[k];
        }
    }
	// generate initial centroids linearly
	for (int k = 0; k < nCluster; k++)
		cCentro[k] = minWeight + (maxWeight - minWeight)*k / (nCluster - 1);

	//initialize all label to -1
	for (int k = 0; k < nWeights; ++k)
		cLabel[k] = -1;

	const Dtype float_max = numeric_limits<Dtype>::max();
	// initialize
	Dtype *cDistance = new Dtype[nWeights];
	int *cClusterSize = new int[nCluster];

	Dtype *pCentroPos = new Dtype[nCluster];
	int *pClusterSize = new int[nCluster];
	memset(pClusterSize, 0, sizeof(int)*nCluster);
	memset(pCentroPos, 0, sizeof(Dtype)*nCluster);
	Dtype *ptrC = new Dtype[nCluster];
	int *ptrS = new int[nCluster];

	int iter = 0;
	//Dtype tk1 = 0.f, tk2 = 0.f, tk3 = 0.f;
	double mCurDistance = 0.0;
	double mPreDistance = numeric_limits<double>::max();

	// clustering
	while (iter < max_iter)
	{
		// check convergence
		if (fabs(mPreDistance - mCurDistance) / mPreDistance < 0.01) break;
		mPreDistance = mCurDistance;
		mCurDistance = 0.0;

		// select nearest cluster

		for (int n = 0; n < nWeights; n++)
		{
			if (!mask[n])
				continue;
			Dtype distance;
			Dtype mindistance = float_max;
			int clostCluster = -1;
			for (int k = 0; k < nCluster; k++)
			{
				distance = fabs(cWeights[n] - cCentro[k]);
				if (distance < mindistance)
				{
					mindistance = distance;
					clostCluster = k;
				}
			}
			cDistance[n] = mindistance;
			cLabel[n] = clostCluster;
		}


		// calc new distance/inertia

		for (int n = 0; n < nWeights; n++)
		{
			if (mask[n])
				mCurDistance = mCurDistance + cDistance[n];
		}


	// generate new centroids
	// accumulation(private)

		for (int k = 0; k < nCluster; k++)
		{
			ptrC[k] = 0.f;
			ptrS[k] = 0;
		}

		for (int n = 0; n < nWeights; n++)
		{
			if (mask[n])
			{
				ptrC[cLabel[n]] += cWeights[n];
				ptrS[cLabel[n]] += 1;
			}
		}

		for (int k = 0; k < nCluster; k++)
		{
			pCentroPos[ k] = ptrC[k];
			pClusterSize[k] = ptrS[k];
		}

		//reduction(global)
		for (int k = 0; k < nCluster; k++)
		{

			cCentro[k] = pCentroPos[k];
			cClusterSize[k] = pClusterSize[k];
	
			cCentro[k] /= cClusterSize[k];
		}

		iter++;
	//	cout << "Iteration: " << iter << " Distance: " << mCurDistance << endl;
		}
		//gather centroids
		//#pragma omp parallel for
		//for(int n=0; n<nNode; n++)
		//    cNodes[n] = cCentro[cLabel[n]];

		delete[] cDistance;
		delete[] cClusterSize;
		delete[] pClusterSize;
		delete[] pCentroPos;
		delete[] ptrC;
		delete[] ptrS;
}



#endif
