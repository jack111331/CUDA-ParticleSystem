//
// Created by Edge on 2021/6/15.
//

#ifndef PARTICLESYSTEM_FORCE_CUH
#define PARTICLESYSTEM_FORCE_CUH

#include "ParticleSystem.h"

__global__ void getLocationKernel(Particle *particleList, int particleSize, float *locationState);

__global__ void clearForceKernel(Particle *particleList, int particleSize);


#endif //PARTICLESYSTEM_FORCE_CUH
