//
// Created by Edge on 2021/6/14.
//

#include "ParticleSystem.h"
#include "ParticleSystem.cuh"
#include <fstream>

#include <nlohmann/json.hpp>
#include <Factory.h>


using namespace std;
using namespace nlohmann;

void ParticleSystem::initialize(const string &jsonFilepath) {
    ifstream ifs(jsonFilepath);

    if (!ifs.good()) {
        std::cerr << "Error: ParticleSystem::initialize(): failed to read config file \"" <<
                  jsonFilepath << "\"!" << std::endl;
        exit(1);
    }

    json jsonData;
    ifs >> jsonData;
    ifs.close();

    m_solver = SolverFactory::generateSolver(jsonData["solver"]);

    m_particleSize = jsonData["particle_list"].size();
    Particle *cpuTmp = (Particle *) malloc(m_particleSize * sizeof(Particle));
    for (int i = 0; i < m_particleSize; ++i) {
        json &particleJson = jsonData["particle_list"][i];
        cpuTmp[i].m_location = Vec3f(particleJson["location"][0], particleJson["location"][1],
                                     particleJson["location"][2]);
        cpuTmp[i].m_velocity = Vec3f(particleJson["velocity"][0], particleJson["velocity"][1],
                                     particleJson["velocity"][2]);
        cpuTmp[i].m_force = Vec3f(0.0f, 0.0f, 0.0f);
        cpuTmp[i].m_mass = particleJson["mass"];
    }
    cudaMalloc((void **) &m_particleList, m_particleSize * sizeof(Particle));
    cudaMemcpy(m_particleList, cpuTmp, m_particleSize * sizeof(Particle), cudaMemcpyHostToDevice);
    for (auto forceJson: jsonData["force_list"]) {
        Force *newForce = ForceFactory::generateForce(forceJson["force_name"]);
        if (newForce == nullptr) {
            continue;
        }
        newForce->loadForce(forceJson);
        m_forceList.emplace_back(newForce);
    }

    for (auto coherentForceJson: jsonData["coherent_force_list"]) {
        CoherentForce *newCoherentForce = CoherentForceFactory::generateCoherentForce(
                coherentForceJson["coherent_force_name"]);
        if (newCoherentForce == nullptr) {
            continue;
        }
        newCoherentForce->loadForce(coherentForceJson);
        m_coherentForceList.emplace_back(newCoherentForce);
    }

    for (auto collisionJson: jsonData["collision_list"]) {
        Collision *newCollision = CollisionFactory::generateCollision(collisionJson["collision_name"]);
        if (newCollision == nullptr) {
            continue;
        }
        newCollision->loadCollision(collisionJson);
        m_collisionList.emplace_back(newCollision);
    }

    for (auto constraintJson: jsonData["constraint_list"]) {
        Constraint *newConstraint = ConstraintFactory::generateConstraint(constraintJson["constraint_name"]);
        if (newConstraint == nullptr) {
            continue;
        }
        newConstraint->loadConstraint(constraintJson);
        m_constraintList.emplace_back(newConstraint);
    }
}

void ParticleSystem::simulation(float dt) {
    writeInitConfigToJson();
    m_solver->resetSolver(this);
    writeFrameToJson(0);
    for (int i = m_frameStart; i < m_frameEnd; ++i) {
        m_solver->solveStep(this, dt);
        for (auto constraint: m_constraintList) {
            if (constraint->getConstraintType() == Constraint::ConstraintType::POST) {
                constraint->applyConstraint(this);
            }
        }
        for (auto collision: m_collisionList) {
            collision->applyCollision(this);
        }
        writeFrameToJson(i);
    }
}

void ParticleSystem::writeInitConfigToJson() {
    string filePath = m_outputDirectory + "config.json";
    float *cpuTmp = (float *) malloc(m_particleSize * 7 * sizeof(float));
    float *gpuTmp;
    cudaMalloc((void **) &gpuTmp, m_particleSize * 7 * sizeof(float));

    getWholeState(gpuTmp);
    cudaMemcpy(cpuTmp, gpuTmp, m_particleSize * 7 * sizeof(float), cudaMemcpyDeviceToHost);
    json jsonData;
    jsonData["frame_start"] = m_frameStart;
    jsonData["frame_end"] = m_frameEnd;
    for (int i = 0; i < m_particleSize; ++i) {
        json particleData;
        vector<float> locationList;
        locationList.emplace_back(cpuTmp[7 * i]);
        locationList.emplace_back(cpuTmp[7 * i + 1]);
        locationList.emplace_back(cpuTmp[7 * i + 2]);
        particleData["location"] = locationList;
        vector<float> velocityList;
        velocityList.emplace_back(cpuTmp[7 * i + 3]);
        velocityList.emplace_back(cpuTmp[7 * i + 4]);
        velocityList.emplace_back(cpuTmp[7 * i + 5]);
        particleData["velocity"] = velocityList;
        particleData["mass"] = cpuTmp[7 * i + 6];
        jsonData["particle_list"].emplace_back(particleData);
    }
    std::ofstream ofs(filePath);
    ofs << jsonData;
    ofs.close();
    free(cpuTmp);
    cudaFree(gpuTmp);
}


void ParticleSystem::writeFrameToJson(int frameNumber) {
    string filePath = m_outputDirectory + to_string(frameNumber) + ".json";
    float *cpuTmp = (float *) malloc(m_particleSize * 3 * sizeof(float));
    float *gpuTmp;
    cudaMalloc((void **) &gpuTmp, m_particleSize * 3 * sizeof(float));

    getLocation(gpuTmp);
    cudaMemcpy(cpuTmp, gpuTmp, m_particleSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    json jsonData;
    for (int i = 0; i < m_particleSize; ++i) {
        json particleData;
        vector<float> locationList;
        locationList.emplace_back(cpuTmp[3 * i]);
        locationList.emplace_back(cpuTmp[3 * i + 1]);
        locationList.emplace_back(cpuTmp[3 * i + 2]);
        particleData["location"] = locationList;
        jsonData["particle_list"].emplace_back(particleData);
    }
    std::ofstream ofs(filePath);
    ofs << jsonData;
    ofs.close();
    free(cpuTmp);
    cudaFree(gpuTmp);
}
