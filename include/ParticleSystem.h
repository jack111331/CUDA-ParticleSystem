//
// Created by Edge on 2021/6/1.
//

#ifndef PARTICLESYSTEM_PARTICLESYSTEM_H
#define PARTICLESYSTEM_PARTICLESYSTEM_H


#include "Utility.h"
#include <vector>
#include <cuda_runtime.h>


class Particle {
public:
    Particle(const Vec3f &location = Vec3f(0.0f, 0.0f, 0.0f), const Vec3f &velocity = Vec3f(0.0f, 0.0f, 0.0f),
             const Vec3f &force = Vec3f(0.0f, 0.0f, 0.0f), float mass = 1.0f) :
            m_location(location), m_velocity(velocity), m_force(force), m_mass(mass) {

    }

    __device__ void clearForce() {
        m_force.zero();
    }

    __device__ void addForce(const Vec3f *force) {
        // Prevent from gpu memory access cpu memory...
        // m_force is in gpu memory, but force is in cpu memory...
        m_force += *force;
    }

    __device__ void derivativeEval(float *particleState) const {
        particleState[0] = m_velocity[0];
        particleState[1] = m_velocity[1];
        particleState[2] = m_velocity[2];
        particleState[3] = m_force[0] / m_mass;
        particleState[4] = m_force[1] / m_mass;
        particleState[5] = m_force[2] / m_mass;
    }

    __device__ void getState(float *particleState) const {
        particleState[0] = m_location[0];
        particleState[1] = m_location[1];
        particleState[2] = m_location[2];
        particleState[3] = m_velocity[0];
        particleState[4] = m_velocity[1];
        particleState[5] = m_velocity[2];
    }

    __device__ void getLocation(float *particleState) const {
        particleState[0] = m_location[0];
        particleState[1] = m_location[1];
        particleState[2] = m_location[2];
    }


    __device__ void getWholeState(float *wholeState) const {
        wholeState[0] = m_location[0];
        wholeState[1] = m_location[1];
        wholeState[2] = m_location[2];
        wholeState[3] = m_velocity[0];
        wholeState[4] = m_velocity[1];
        wholeState[5] = m_velocity[2];
        wholeState[6] = m_mass;
    }

    __device__ void setState(const float *particleState) {
        m_location[0] = particleState[0];
        m_location[1] = particleState[1];
        m_location[2] = particleState[2];
        m_velocity[0] = particleState[3];
        m_velocity[1] = particleState[4];
        m_velocity[2] = particleState[5];
    }

    __device__ bool isCollision(const Particle &anotherParticle) const {
        return (m_location - anotherParticle.m_location).lengthWithoutSquare() <=
               (m_mass + anotherParticle.m_mass) * (m_mass + anotherParticle.m_mass);
    }

    Vec3f m_location;
    Vec3f m_velocity;
    Vec3f m_force;
    float m_mass;
};

class Solver;

class Force;

class CoherentForce;

class Constraint;

class Collision;

class ParticleSystem {
public:
    void initialize(const std::string &jsonFilepath);

    void simulation(float dt);

    __device__ float *getState() const {
        float *cpuTmp = (float *) malloc(m_particleSize * 6 * sizeof(float));
        for (int i = 0; i < m_particleSize; ++i) {
            m_particleList[i].getState(&cpuTmp[6 * i]);
        }
        float *gpuTmp;
        cudaMalloc((void **) &gpuTmp, m_particleSize * 6 * sizeof(float));
        cudaMemcpy(gpuTmp, cpuTmp, m_particleSize * 6 * sizeof(float), cudaMemcpyHostToDevice);
        free(cpuTmp);
        return gpuTmp;
    }

    __device__ void setState(float *state) {
        for (int i = 0; i < m_particleSize; ++i) {
            m_particleList[i].setState(&state[6 * i]);
        }
    }

    void getLocation(float *locationState);

    void getWholeState(float *wholeState);

    void clearForce();

    __device__ float *derivativeEval() const {
        float *cpuTmp = (float *) malloc(m_particleSize * 6 * sizeof(float));
        for (int i = 0; i < m_particleSize; ++i) {
            m_particleList[i].derivativeEval(&cpuTmp[6 * i]);
        }
        float *gpuTmp;
        cudaMalloc((void **) &gpuTmp, m_particleSize * 6 * sizeof(float));
        cudaMemcpy(gpuTmp, cpuTmp, m_particleSize * 6 * sizeof(float), cudaMemcpyHostToDevice);
        free(cpuTmp);
        return gpuTmp;
    }

    void writeInitConfigToJson();

    void writeFrameToJson(int frameNumber);

    Particle *m_particleList;
    int m_particleSize;

    std::vector<Force *> m_forceList;
    std::vector<CoherentForce *> m_coherentForceList;
    std::vector<Constraint *> m_constraintList;
    std::vector<Collision *> m_collisionList;
    std::string m_outputDirectory;
    float m_timeStep;
    int m_frameStart=1, m_frameEnd = 251;
    Solver *m_solver;
};

#endif //PARTICLESYSTEM_PARTICLESYSTEM_H