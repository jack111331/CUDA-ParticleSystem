
#include "Force.h"
#include "ParticleSystem.h"


__global__ void constantForceApply(Particle *particle, int particleSize, const Vec3f *constantForce) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particle[particleIdx].addForce(constantForce);
}

void ConstantForce::applyForce(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    constantForceApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, pSystem->m_particleSize, m_constant);
}

__global__ void dampingForceApply(Particle *particle, int particleSize, const float *dampingConstant) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    const Vec3f &particleVelocity = particle[particleIdx].m_velocity;
    Vec3f dampingForce = -*dampingConstant * particleVelocity;
    particle[particleIdx].addForce(&dampingForce);
}

void DampingForce::applyForce(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    dampingForceApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, pSystem->m_particleSize, m_constant);
}

__global__ void springForceApply(Particle *particle, int particleSize, const float *springConstant, const Vec3f *restLocation) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    const Vec3f &particleLocation = particle[particleIdx].m_location;
    Vec3f springForce = *springConstant * (*restLocation - particleLocation);
    particle[particleIdx].addForce(&springForce);
}

void SpringForce::applyForce(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    springForceApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, pSystem->m_particleSize, m_constant, m_restLocation);
}

__global__ void gravityForceApply(Particle *particle, int particleSize, const Vec3f *gravityConstant) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    float particleMass = particle[particleIdx].m_mass;
    Vec3f gravityForce = particleMass * -*gravityConstant;
    particle[particleIdx].addForce(&gravityForce);
}

void GravityForce::applyForce(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    gravityForceApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, pSystem->m_particleSize, m_constant);
}

__global__ void springTwoParticleForceApply(Particle *particle, int pairSize, const int *particleIdxList_1, const int *particleIdxList_2, const float *restLengthList, float *springConstant) {
    int pairIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (pairIdx >= pairSize) {
        return;
    }

    Particle &particle_1 = particle[particleIdxList_1[pairIdx]];
    Particle &particle_2 = particle[particleIdxList_2[pairIdx]];
    Vec3f &location_1 = particle_1.m_location;
    Vec3f &location_2 = particle_2.m_location;
    float restLength = restLengthList[pairIdx];
    Vec3f locationVector = location_1 - location_2;
    Vec3f spring_force = *springConstant * (restLength - locationVector.length()) * locationVector/locationVector.length();
    particle_1.addForce(&spring_force);
    spring_force = -spring_force;
    particle_2.addForce(&spring_force);

}

void SpringTwoParticleForce::applyForce(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    springTwoParticleForceApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, m_pairSize, m_particleIdxList_1, m_particleIdxList_2, m_restLengthList, m_constant);
}
