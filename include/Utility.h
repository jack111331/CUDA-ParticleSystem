//
// Created by Edge on 2021/6/1.
//

#ifndef PARTICLESYSTEM_UTILITY_H
#define PARTICLESYSTEM_UTILITY_H

#include <cmath>
#include <iostream>
#include <vector>
#include <stdlib.h>

#include <cuda_runtime.h>

class Vec3f {
public:
    float x, y, z;

    __host__ __device__ Vec3f() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}

    friend std::istream &operator>>(std::istream &is, Vec3f &velocity) {
        is >> velocity.x >> velocity.y >> velocity.z;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const Vec3f &velocity) {
        os << "(" << velocity.x << ", " << velocity.y << ", " << velocity.z << ")";
        return os;
    }

    __device__ Vec3f &operator+=(const Vec3f &rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    __device__ Vec3f operator+(const Vec3f &rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    __host__ __device__ Vec3f &operator-=(const Vec3f &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    __host__ __device__ Vec3f operator-(const Vec3f &rhs) const {
        Vec3f copy(*this);
        copy -= rhs;
        return copy;
    }

    __host__ __device__ Vec3f operator-() const {
        return {-x, -y, -z};
    }

    __device__ float operator[](int dim) const {
        return dim == 0 ? x : dim == 1 ? y : z;
    };

    __device__ float &operator[](int dim) {
        return dim == 0 ? x : dim == 1 ? y : z;
    };

    Vec3f &operator*=(float rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    __host__ __device__ Vec3f operator*(float rhs) const {
        return {x * rhs, y * rhs, z * rhs};
    }

    __host__ __device__ Vec3f operator*(const Vec3f &rhs) const {
        return {x * rhs.x, y * rhs.y, z * rhs.z};
    }

    Vec3f operator/=(float rhs) {
        x /= rhs;
        y /= rhs;
        z /= rhs;
        return *this;
    }

    Vec3f operator/(const Vec3f &rhs) const {
        return {x / rhs.x, y / rhs.y, z / rhs.z};
    }

    __device__ Vec3f operator/(float rhs) const {
        return {x / rhs, y / rhs, z / rhs};
    }

    bool operator>=(const Vec3f &rhs) const {
        return x >= rhs.x && y >= rhs.y && z >= rhs.z;
    }

    bool operator<=(const Vec3f &rhs) const {
        return x <= rhs.x && y <= rhs.y && z <= rhs.z;
    }

    __device__ void zero() {
        x = y = z = 0.0f;
    }

    __device__ float length() const {
        return sqrt(x * x + y * y + z * z);
    }

    __device__ float lengthWithoutSquare() const {
        return x * x + y * y + z * z;
    }

    __device__ float dot(const Vec3f &rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    Vec3f cross(const Vec3f &rhs) const {
        return {y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z, x * rhs.y - y * rhs.x};
    }

    __device__ Vec3f reflect(const Vec3f &normal) const {
        Vec3f reflectedVelocity(*this);
        float dot = reflectedVelocity.dot(normal);
        if (dot > 0.0) {
            return reflectedVelocity + 2.0 * dot * normal;
        }
        return reflectedVelocity - 2.0 * dot * normal;
    }

    __device__ bool refract(const Vec3f &normal, float niOverNt, Vec3f &refracted) const {
        Vec3f unitVector = (*this);
        unitVector = unitVector.normalize();
        float dt = unitVector.dot(normal);
        float discriminant = 1.0 - niOverNt * niOverNt * (1.0 - dt * dt);
        if (discriminant > 0.0) {
            refracted = niOverNt * (unitVector - normal * dt) - normal * sqrt(discriminant);
            return true;
        } else {
            return false;
        }
    }

    __device__ Vec3f normalize() const {
        Vec3f normalizedVelocity(*this);
        return normalizedVelocity.normalize();
    }

    __device__ Vec3f &normalize() {
        float velocityLength = length();
        x /= velocityLength;
        y /= velocityLength;
        z /= velocityLength;
        return *this;
    }

    __host__ __device__ friend Vec3f operator*(float lhs, const Vec3f &rhs) {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }

    bool nearZero() const {
        return x <= 1e-6 && y <= 1e-6 && z <= 1e-6;
    }

    static Vec3f toVec3f(const std::vector<float> &floatList) {
        if (floatList.size() == 3) {
            return {floatList[0], floatList[1], floatList[2]};
        } else {
            return Vec3f(0, 0, 0);
        }
    }

    void clamp(float minVal, float maxVal) {
        x = x > maxVal ? maxVal : x < minVal ? minVal : x;
        y = y > maxVal ? maxVal : y < minVal ? minVal : y;
        z = z > maxVal ? maxVal : z < minVal ? minVal : z;
    }

};



#endif //PARTICLESYSTEM_UTILITY_H
