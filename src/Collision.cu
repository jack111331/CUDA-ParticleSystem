#include "Collision.h"
#include "ParticleSystem.h"

__global__ void
wallCollisionApply(Particle *particle, int particleSize, const Vec3f *wallLocation, const Vec3f *wallNormal) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    Vec3f &location = particle[particleIdx].m_location;
    float projection = (*wallLocation - location).dot(*wallNormal);
    bool isInsideWall = projection > 0;
    if (isInsideWall) {
        particle[particleIdx].m_location += (2 * projection) * *wallNormal;
        particle[particleIdx].m_velocity = particle[particleIdx].m_velocity.reflect(*wallNormal);
    }
}

void WallCollision::applyCollision(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(32, 32);
    dim3 threadsPerBlock(32, 32);
    wallCollisionApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, pSystem->m_particleSize,
                                                            m_location, m_normal);
}

__global__ void particleCollisionApply(Particle *particle, int particleSize) {
    int particleIdx_1 = threadIdx.x + blockIdx.x * blockDim.x;
    int particleIdx_2 = threadIdx.y + blockIdx.y * blockDim.y;
    if (particleIdx_1 >= particleSize || particleIdx_2 >= particleSize || particleIdx_1 >= particleIdx_2) {
        return;
    }
    Particle &particle_1 = particle[particleIdx_1];
    Particle &particle_2 = particle[particleIdx_2];
    if (particle_1.isCollision(particle_2)) {
        const Vec3f &velocity_1 = particle_1.m_velocity;
        const Vec3f &velocity_2 = particle_2.m_velocity;
        float mass_1 = particle_1.m_mass;
        float mass_2 = particle_2.m_mass;
        Vec3f normal = (particle_1.m_location - particle_2.m_location).normalize();
        float a = 2.0f * normal.dot(velocity_1 - velocity_2) / (1.0f / mass_1 + 1.0f / mass_2);

        particle_1.m_velocity -= ((a / mass_1) * normal);
        particle_2.m_velocity += ((a / mass_2) * normal);
    }
}

void ParticleCollision::applyCollision(ParticleSystem *pSystem) {
    dim3 blocksPerGrid(32, 32);
    dim3 threadsPerBlock(32, 32);
    particleCollisionApply<<< blocksPerGrid, threadsPerBlock>>>(pSystem->m_particleList, pSystem->m_particleSize);
}