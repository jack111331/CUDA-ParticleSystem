//
// Created by Edge on 2021/6/1.
//

#ifndef PARTICLESYSTEM_FORCE_H
#define PARTICLESYSTEM_FORCE_H

#include "Utility.h"
#include "nlohmann/json_fwd.hpp"

class ParticleSystem;


class Force {
public:
    // one cpu call match one gpu call
    virtual void applyForce(ParticleSystem *pSystem) = 0;

    virtual void loadForce(const nlohmann::json &jsonData) {}
};

class ConstantForce : public Force {
public:
    ConstantForce() {}

    // one cpu call match one gpu call
    virtual void applyForce(ParticleSystem *pSystem);

    virtual void loadForce(const nlohmann::json &jsonData);

    Vec3f *m_constant;
};

class DampingForce : public Force {
public:
    DampingForce() {}

    // one cpu call match one gpu call
    virtual void applyForce(ParticleSystem *pSystem);

    virtual void loadForce(const nlohmann::json &jsonData);

    float *m_constant;
};

class SpringForce : public Force {
public:
    SpringForce() {}

    // one cpu call match one gpu call
    virtual void applyForce(ParticleSystem *pSystem);

    virtual void loadForce(const nlohmann::json &jsonData);

    float *m_constant;
    Vec3f *m_restLocation;
};

class GravityForce : public Force {
public:
    GravityForce() {}

    // one cpu call match one gpu call
    virtual void applyForce(ParticleSystem *pSystem);

    virtual void loadForce(const nlohmann::json &jsonData);

    Vec3f *m_constant;
};

class CoherentForce {
public:
    // one cpu call match one gpu call
    virtual void applyForce(ParticleSystem *pSystem) = 0;

    virtual void loadForce(const nlohmann::json &jsonData) {}
};

class SpringTwoParticleForce : public CoherentForce {
public:
    virtual void applyForce(ParticleSystem *pSystem);

    virtual void loadForce(const nlohmann::json &jsonData);

    int *m_particleIdxList_1;
    int *m_particleIdxList_2;
    float *m_restLengthList;
    int m_pairSize;
    float *m_constant;
};

#endif //PARTICLESYSTEM_FORCE_H
