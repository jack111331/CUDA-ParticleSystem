//
// Created by Edge on 2021/6/1.
//

#ifndef PARTICLESYSTEM_SOLVER_H
#define PARTICLESYSTEM_SOLVER_H


#include "ParticleSystem.h"

class Solver {
public:
    void refreshForce(ParticleSystem *particleSystem);
    virtual void solveStep(ParticleSystem *particleSystem, float step) = 0;
    virtual void resetSolver(ParticleSystem *particleSystem) {}

};

class ForwardEulerSolver: public Solver {
public:
    virtual void solveStep(ParticleSystem *particleSystem, float step);
};

class SecondOrderRkSolver: public Solver {
public:
    virtual void solveStep(ParticleSystem *particleSystem, float step);
};

class FourthOrderRkSolver: public Solver {
public:
    virtual void solveStep(ParticleSystem *particleSystem, float step);
};

#endif //PARTICLESYSTEM_SOLVER_H
