//
// Created by Edge on 2021/6/1.
//
#include <nlohmann/json.hpp>

#include "Collision.h"

void WallCollision::loadCollision(const nlohmann::json &jsonData) {
    Vec3f wallLocation = Vec3f(jsonData["wall_location"][0], jsonData["wall_location"][1],
                               jsonData["wall_location"][2]);
    cudaMalloc((void **) &m_location, sizeof(Vec3f));
    cudaMemcpy(m_location, &wallLocation, sizeof(Vec3f), cudaMemcpyHostToDevice);

    Vec3f wallNormal = Vec3f(jsonData["wall_normal"][0], jsonData["wall_normal"][1],
                               jsonData["wall_normal"][2]);
    cudaMalloc((void **) &m_normal, sizeof(Vec3f));
    cudaMemcpy(m_normal, &wallNormal, sizeof(Vec3f), cudaMemcpyHostToDevice);
}