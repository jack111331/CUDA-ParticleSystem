//
// Created by Edge on 2021/6/1.
//

#ifndef PARTICLESYSTEM_CONSTRAINT_H
#define PARTICLESYSTEM_CONSTRAINT_H

#include "Utility.h"
#include "nlohmann/json_fwd.hpp"

class ParticleSystem;

class Constraint {
public:
    // one cpu call match one gpu call
    virtual void applyConstraint(ParticleSystem *pSystem) = 0;

    virtual void loadConstraint(const nlohmann::json &jsonData) {}


    enum ConstraintType {
        PRE,
        POST
    };

    virtual ConstraintType getConstraintType() const = 0;

};

class PinConstraint: public Constraint {
public:
    // one cpu call match one gpu call
    virtual void applyConstraint(ParticleSystem *pSystem);

    virtual void loadConstraint(const nlohmann::json &jsonData);

    virtual ConstraintType getConstraintType() const {
        return ConstraintType::PRE;
    }

    int m_pairSize;
    int *m_particleIdxList;
    Vec3f *m_pinLocationList;

};

class AxisConstraint: public Constraint {
public:
    // one cpu call match one gpu call
    virtual void applyConstraint(ParticleSystem *pSystem);

    virtual void loadConstraint(const nlohmann::json &jsonData);

    virtual ConstraintType getConstraintType() const {
        return ConstraintType::PRE;
    }

    int m_pairSize;
    int *m_particleIdxList;
    Vec3f *m_axisVectorList;

};

class PlaneConstraint: public Constraint {
public:
    // one cpu call match one gpu call
    virtual void applyConstraint(ParticleSystem *pSystem);

    virtual void loadConstraint(const nlohmann::json &jsonData);

    virtual ConstraintType getConstraintType() const {
        return ConstraintType::PRE;
    }

    int m_pairSize;
    int *m_particleIdxList;
    Vec3f *m_planeVectorList;

};

class AngularConstraint: public Constraint {
public:
    // one cpu call match one gpu call
    virtual void applyConstraint(ParticleSystem *pSystem);

    virtual void loadConstraint(const nlohmann::json &jsonData);

    virtual ConstraintType getConstraintType() const {
        return ConstraintType::POST;
    }


    int m_axisParticleIdx;
    int m_pairParticleIdx_1;
    int m_pairParticleIdx_2;
    float m_minAngle;
    float m_maxAngle;

};

#endif //PARTICLESYSTEM_CONSTRAINT_H
