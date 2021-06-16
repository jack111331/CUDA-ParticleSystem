//
// Created by Edge on 2021/6/1.
//
#include <nlohmann/json.hpp>

#include "Force.h"


void ConstantForce::loadForce(const nlohmann::json &jsonData) {
    Vec3f constantForce = Vec3f(jsonData["constant_force"][0], jsonData["constant_force"][1],
                                jsonData["constant_force"][2]);
    cudaMalloc((void **) &m_constant, sizeof(Vec3f));
    cudaMemcpy(m_constant, &constantForce, sizeof(Vec3f), cudaMemcpyHostToDevice);
}

void DampingForce::loadForce(const nlohmann::json &jsonData) {
    float dampingConstant = jsonData["constant_damp"];
    cudaMalloc((void **) &m_constant, sizeof(float));
    cudaMemcpy(m_constant, &dampingConstant, sizeof(float), cudaMemcpyHostToDevice);
}

void SpringForce::loadForce(const nlohmann::json &jsonData) {
    float springConstant = jsonData["constant_spring"];
    cudaMalloc((void **) &m_constant, sizeof(float));
    cudaMemcpy(m_constant, &springConstant, sizeof(float), cudaMemcpyHostToDevice);

    Vec3f restLocation = Vec3f(jsonData["rest_location"][0], jsonData["rest_location"][1],
                               jsonData["rest_location"][2]);
    cudaMalloc((void **) &m_restLocation, sizeof(Vec3f));
    cudaMemcpy(m_restLocation, &restLocation, sizeof(Vec3f), cudaMemcpyHostToDevice);
}

void GravityForce::loadForce(const nlohmann::json &jsonData) {
    Vec3f gravityConstant = Vec3f(0.0, 0.0, 9.8f);
    cudaMalloc((void **) &m_constant, sizeof(Vec3f));
    cudaMemcpy(m_constant, &gravityConstant, sizeof(Vec3f), cudaMemcpyHostToDevice);
}

void SpringTwoParticleForce::loadForce(const nlohmann::json &jsonData) {
    m_pairSize = jsonData["coherent_particle_list"].size();
    cudaMalloc((void **) &m_particleIdxList_1, m_pairSize * sizeof(int));
    cudaMalloc((void **) &m_particleIdxList_2, m_pairSize * sizeof(int));
    cudaMalloc((void **) &m_restLengthList, m_pairSize * sizeof(float));

    int *cpuIntTmp = (int *) malloc(m_pairSize * sizeof(int));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuIntTmp[i] = jsonData["coherent_particle_list"][i]["coherent_particle_idx"][0];
    }
    cudaMemcpy(m_particleIdxList_1, cpuIntTmp, m_pairSize * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < m_pairSize; ++i) {
        cpuIntTmp[i] = jsonData["coherent_particle_list"][i]["coherent_particle_idx"][1];
    }
    cudaMemcpy(m_particleIdxList_2, cpuIntTmp, m_pairSize * sizeof(int), cudaMemcpyHostToDevice);

    float *cpuFloatTmp = (float *) malloc(m_pairSize * sizeof(float));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuFloatTmp[i] = jsonData["coherent_particle_list"][i]["rest_length"];
    }
    cudaMemcpy(m_restLengthList, cpuFloatTmp, m_pairSize * sizeof(float), cudaMemcpyHostToDevice);
    free(cpuIntTmp);
    free(cpuFloatTmp);

    float springConstant = jsonData["spring_constant"];
    cudaMalloc((void **) &m_constant, sizeof(float));
    cudaMemcpy(m_constant, &springConstant, sizeof(float), cudaMemcpyHostToDevice);
}