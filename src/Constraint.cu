#include "Constraint.h"
#include "ParticleSystem.h"

__global__ void
pinConstraintApply(Particle *particle, int pairSize, const int *particleIdxList, const Vec3f *pinLocationList) {
    int pairIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (pairIdx >= pairSize) {
        return;
    }

    Particle &pinParticle = particle[particleIdxList[pairIdx]];
    pinParticle.m_velocity.zero();
    pinParticle.m_force.zero();
    pinParticle.m_location = pinLocationList[pairIdx];
}

void PinConstraint::applyConstraint(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    pinConstraintApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, m_pairSize,
                                                            m_particleIdxList, m_pinLocationList);
}

__global__ void
axisConstraintApply(Particle *particle, int pairSize, const int *particleIdxList, const Vec3f *axisVectorList) {
    int pairIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (pairIdx >= pairSize) {
        return;
    }

    Particle &axisParticle = particle[particleIdxList[pairIdx]];
    axisParticle.m_force.zero();
    axisParticle.m_velocity = axisParticle.m_velocity * axisVectorList[pairIdx];
}

void AxisConstraint::applyConstraint(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    axisConstraintApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, m_pairSize,
                                                             m_particleIdxList, m_axisVectorList);
}

__global__ void
planeConstraintApply(Particle *particle, int pairSize, const int *particleIdxList, const Vec3f *planeVectorList) {
    int pairIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (pairIdx >= pairSize) {
        return;
    }

    Particle &planeParticle = particle[particleIdxList[pairIdx]];
    planeParticle.m_force.zero();
    planeParticle.m_velocity = planeParticle.m_velocity * planeVectorList[pairIdx];
}

void PlaneConstraint::applyConstraint(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    planeConstraintApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, m_pairSize,
                                                             m_particleIdxList, m_planeVectorList);
}


__global__ void
angularConstraintApply(Particle *particle, int particleSize, int axisParticleIdx,  int particleIdx_1, int particleIdx_2, float minAngle, float maxAngle) {
    int pairIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (pairIdx >= particleSize) {
        return;
    }
    // TODO
}

void AngularConstraint::applyConstraint(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    angularConstraintApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, pSystem->m_particleSize, m_axisParticleIdx,
                                                              m_pairParticleIdx_1, m_pairParticleIdx_2, m_minAngle, m_maxAngle);
}