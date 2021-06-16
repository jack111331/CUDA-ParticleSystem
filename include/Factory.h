//
// Created by Edge on 2021/6/15.
//

#ifndef PARTICLESYSTEM_FACTORY_H
#define PARTICLESYSTEM_FACTORY_H

#include "Force.h"
#include "Solver.h"
#include "Collision.h"
#include "Constraint.h"

class ForceFactory {
public:
    static Force *generateForce(const std::string &forceName) {
        if (forceName == "constant_force") {
            return new ConstantForce();
        } else if (forceName == "damping_force") {
            return new DampingForce();
        } else if (forceName == "spring_force") {
            return new SpringForce();
        } else if (forceName == "gravity_force") {
            return new GravityForce();
        }
        return nullptr;
    }
};

class CoherentForceFactory {
public:
    static CoherentForce *generateCoherentForce(const std::string &coherentForceName) {
        if (coherentForceName == "spring_two_particle_force") {
            return new SpringTwoParticleForce();
        }
        return nullptr;
    }
};

class CollisionFactory {
public:
    static Collision *generateCollision(const std::string &collisionName) {
        if (collisionName == "wall_collision") {
            return new WallCollision();
        } else if (collisionName == "particle_collision") {
            return new ParticleCollision();
        }
        return nullptr;
    }
};

class ConstraintFactory {
public:
    static Constraint *generateConstraint(const std::string &constraintName) {
        if (constraintName == "pin_constraint") {
            return new PinConstraint();
        } else if (constraintName == "axis_constraint") {
            return new AxisConstraint();
        } else if (constraintName == "plane_constraint") {
            return new PlaneConstraint();
        } else if(constraintName == "angular_constraint") {
            return new AngularConstraint();
        }
        return nullptr;
    }
};

class SolverFactory {
public:
    static Solver *generateSolver(const std::string &solverName) {
        if (solverName == "forward_euler_solver") {
            return new ForwardEulerSolver();
        } else if (solverName == "second_order_rk_solver") {
            return new SecondOrderRkSolver();
        } else if (solverName == "fourth_order_rk_solver") {
            return new FourthOrderRkSolver();
        }
        return nullptr;
    }
};

#endif //PARTICLESYSTEM_FACTORY_H
