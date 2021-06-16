//
// Created by Edge on 2021/6/1.
//
#include <nlohmann/json.hpp>

#include "Constraint.h"

void PinConstraint::loadConstraint(const nlohmann::json &jsonData) {
    m_pairSize = jsonData["pin_list"].size();
    cudaMalloc((void **) &m_particleIdxList, m_pairSize * sizeof(int));
    cudaMalloc((void **) &m_pinLocationList, m_pairSize * sizeof(Vec3f));

    int *cpuIntTmp = (int *) malloc(m_pairSize * sizeof(int));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuIntTmp[i] = jsonData["pin_list"][i]["pin_particle_idx"];
    }
    cudaMemcpy(m_particleIdxList, cpuIntTmp, m_pairSize * sizeof(int), cudaMemcpyHostToDevice);

    Vec3f *cpuVec3fTmp = (Vec3f *) malloc(m_pairSize * sizeof(Vec3f));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuVec3fTmp[i] = Vec3f(jsonData["pin_list"][i]["pin_location"][0], jsonData["pin_list"][i]["pin_location"][1], jsonData["pin_list"][i]["pin_location"][2]);
    }
    cudaMemcpy(m_pinLocationList, cpuVec3fTmp, m_pairSize * sizeof(Vec3f), cudaMemcpyHostToDevice);
    free(cpuIntTmp);
    free(cpuVec3fTmp);
}

void AxisConstraint::loadConstraint(const nlohmann::json &jsonData) {
    m_pairSize = jsonData["axis_list"].size();
    cudaMalloc((void **) &m_particleIdxList, m_pairSize * sizeof(int));
    cudaMalloc((void **) &m_axisVectorList, m_pairSize * sizeof(Vec3f));

    int *cpuIntTmp = (int *) malloc(m_pairSize * sizeof(int));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuIntTmp[i] = jsonData["axis_list"][i]["axis_particle_idx"];
    }
    cudaMemcpy(m_particleIdxList, cpuIntTmp, m_pairSize * sizeof(int), cudaMemcpyHostToDevice);

    Vec3f *cpuVec3fTmp = (Vec3f *) malloc(m_pairSize * sizeof(Vec3f));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuVec3fTmp[i] = Vec3f(jsonData["axis_list"][i]["axis_vector"][0], jsonData["axis_list"][i]["axis_vector"][1], jsonData["axis_list"][i]["axis_vector"][2]);
    }
    cudaMemcpy(m_axisVectorList, cpuVec3fTmp, m_pairSize * sizeof(Vec3f), cudaMemcpyHostToDevice);
    free(cpuIntTmp);
    free(cpuVec3fTmp);
}

void PlaneConstraint::loadConstraint(const nlohmann::json &jsonData) {
    m_pairSize = jsonData["plane_list"].size();
    cudaMalloc((void **) &m_particleIdxList, m_pairSize * sizeof(int));
    cudaMalloc((void **) &m_planeVectorList, m_pairSize * sizeof(Vec3f));

    int *cpuIntTmp = (int *) malloc(m_pairSize * sizeof(int));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuIntTmp[i] = jsonData["plane_list"][i]["plane_particle_idx"];
    }
    cudaMemcpy(m_particleIdxList, cpuIntTmp, m_pairSize * sizeof(int), cudaMemcpyHostToDevice);

    Vec3f *cpuVec3fTmp = (Vec3f *) malloc(m_pairSize * sizeof(Vec3f));
    for (int i = 0; i < m_pairSize; ++i) {
        cpuVec3fTmp[i] = Vec3f(jsonData["plane_list"][i]["plane_vector"][0], jsonData["plane_list"][i]["plane_vector"][1], jsonData["plane_list"][i]["plane_vector"][2]);
    }
    cudaMemcpy(m_planeVectorList, cpuVec3fTmp, m_pairSize * sizeof(Vec3f), cudaMemcpyHostToDevice);
    free(cpuIntTmp);
    free(cpuVec3fTmp);
}

void AngularConstraint::loadConstraint(const nlohmann::json &jsonData) {
    cudaMalloc((void **) &m_axisParticleIdx, sizeof(int));
    cudaMalloc((void **) &m_pairParticleIdx_1, sizeof(int));
    cudaMalloc((void **) &m_pairParticleIdx_2, sizeof(int));
    cudaMalloc((void **) &m_minAngle, sizeof(float));
    cudaMalloc((void **) &m_maxAngle, sizeof(float));

    m_axisParticleIdx = jsonData["axis_particle_idx"];
    m_pairParticleIdx_1 = jsonData["pair_particle_idx_1"];
    m_pairParticleIdx_2 = jsonData["pair_particle_idx_2"];
    m_minAngle = jsonData["min_angle"];
    m_maxAngle = jsonData["max_angle"];

}