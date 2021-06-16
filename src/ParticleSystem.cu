#include "ParticleSystem.h"


using namespace std;

__global__ void getLocationKernel(Particle *particleList, int particleSize, float *locationState) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].getLocation(&locationState[3 * particleIdx]);
}

void ParticleSystem::getLocation(float *locationState) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    getLocationKernel<<<blocksPerGrid, threadsPerBlock>>>(m_particleList, m_particleSize, locationState);

}

__global__ void getWholeStateKernel(Particle *particleList, int particleSize, float *wholeState) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].getWholeState(&wholeState[7 * particleIdx]);
}

void ParticleSystem::getWholeState(float *wholeState) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    getWholeStateKernel<<<blocksPerGrid, threadsPerBlock>>>(m_particleList, m_particleSize, wholeState);

}

__global__ void clearForceKernel(Particle *particleList, int particleSize) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].clearForce();
}

void ParticleSystem::clearForce() {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    clearForceKernel<<<blocksPerGrid, threadsPerBlock>>>(m_particleList, m_particleSize);
}

