//
// Created by Edge on 2020/10/14.
//

#include <time.h>
#include "Utility.h"

namespace Util {
    float schlickApprox(float cosine, float referenceIndex) {
        float r0 = (1 - referenceIndex) / (1 + referenceIndex);
        r0 *= r0;
        return r0 + (1 - r0) * pow((1.0 - cosine), 5);
    }

    float randomInUnit() {
        srand(time(nullptr));
        return rand() / (RAND_MAX + 1.0f);
    }

    Vec3f randomInHemisphere() {
        float r1 = randomInUnit(), r2 = randomInUnit();
        float PI = acos(-1);
        float x = cos(2 * PI * r1) * 2 * sqrt(r2 * (2 - r2));
        float y = sin(2 * PI * r1) * 2 * sqrt(r2 * (2 - r2));
        float z = 1 - r2;
        return Vec3f(x, y, z);
    }

    Vec3f randomCosineDirection() {
        float r1 = randomInUnit();
        auto r2 = randomInUnit();
        auto z = sqrt(1 - r2);
        float PI = acos(-1);
        auto phi = 2 * PI * r1;
        auto x = cos(phi) * sqrt(r2);
        auto y = sin(phi) * sqrt(r2);

        return Vec3f(x, y, z);
    }

    float randomIn(float min, float max) {
        return min + (max - min) * randomInUnit();
    }

    Vec3f randomSphere() {
        Vec3f result = {2.0f * randomInUnit() - 1.0f, 2.0f * randomInUnit() - 1.0f, 2.0f * randomInUnit() - 1.0f};
        result.normalize();
        return result;
    }

};