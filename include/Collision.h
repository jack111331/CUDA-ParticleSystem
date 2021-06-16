//
// Created by Edge on 2021/6/1.
//

#ifndef PARTICLESYSTEM_COLLISION_H
#define PARTICLESYSTEM_COLLISION_H

#include "Utility.h"
#include "nlohmann/json_fwd.hpp"

class ParticleSystem;

class Collision {
public:
    // one cpu call match one gpu call
    virtual void applyCollision(ParticleSystem *pSystem) = 0;

    virtual void loadCollision(const nlohmann::json &jsonData) {}
};

class WallCollision : public Collision {
public:
    // one cpu call match one gpu call
    virtual void applyCollision(ParticleSystem *pSystem);

    virtual void loadCollision(const nlohmann::json &jsonData);

    Vec3f *m_location;
    Vec3f *m_normal;
};

class ParticleCollision : public Collision {
public:
    // one cpu call match one gpu call
    virtual void applyCollision(ParticleSystem *pSystem);

    virtual void loadCollision(const nlohmann::json &jsonData) {}

};

#endif //PARTICLESYSTEM_COLLISION_H
