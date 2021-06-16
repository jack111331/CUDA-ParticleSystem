
#include "Solver.h"
#include "Constraint.h"
#include "Force.h"

void Solver::refreshForce(ParticleSystem *particleSystem) {
    particleSystem->clearForce();
    for (auto force: particleSystem->m_forceList) {
        force->applyForce(particleSystem);
    }
    for (auto coherentForce: particleSystem->m_coherentForceList) {
        coherentForce->applyForce(particleSystem);
    }
    for (auto constraint: particleSystem->m_constraintList) {
        if (constraint->getConstraintType() == Constraint::ConstraintType::PRE) {
            constraint->applyConstraint(particleSystem);

        }
    }
}

__global__ void
forwardEulerSolveStepKernel(Particle *particleList, float *currentStateBuf, float *derivativeBuf, int particleSize,
                            float step) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].getState(&currentStateBuf[6 * particleIdx]);
    particleList[particleIdx].derivativeEval(&derivativeBuf[6 * particleIdx]);
    for (int i = 0; i < 6; ++i) {
        currentStateBuf[6 * particleIdx + i] += step * derivativeBuf[6 * particleIdx + i];
    }
    particleList[particleIdx].setState(&currentStateBuf[6 * particleIdx]);
}

void ForwardEulerSolver::solveStep(ParticleSystem *particleSystem, float step) {
    // should do per particle kernel
    // but should claim temp memory for later use
    float *currentStateBuf;
    float *derivativeBuf;
    cudaMalloc((void **) &currentStateBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    cudaMalloc((void **) &derivativeBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    // target
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    refreshForce(particleSystem);
    forwardEulerSolveStepKernel<<<blocksPerGrid, threadsPerBlock>>>(particleSystem->m_particleList, currentStateBuf,
                                                                    derivativeBuf,
                                                                    particleSystem->m_particleSize, step);
    cudaFree(currentStateBuf);
    cudaFree(derivativeBuf);
}

__global__ void
secondOrderRkSolveStepKernel_1(Particle *particleList, float *originStateBuf, float *currentStateBuf,
                               float *derivativeBuf, int particleSize, float step) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].getState(&originStateBuf[6 * particleIdx]);
    particleList[particleIdx].derivativeEval(&derivativeBuf[6 * particleIdx]);
    for (int i = 0; i < 6; ++i) {
        currentStateBuf[6 * particleIdx + i] =
                originStateBuf[6 * particleIdx + i] + 0.5f * step * derivativeBuf[6 * particleIdx + i];
    }
    particleList[particleIdx].setState(&currentStateBuf[6 * particleIdx]);

}

__global__ void
secondOrderRkSolveStepKernel_2(Particle *particleList, float *originStateBuf, float *currentStateBuf,
                               float *derivativeBuf, int particleSize, float step) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }

    particleList[particleIdx].derivativeEval(&derivativeBuf[6 * particleIdx]);
    for (int i = 0; i < 6; ++i) {
        currentStateBuf[6 * particleIdx + i] =
                originStateBuf[6 * particleIdx + i] + step * derivativeBuf[6 * particleIdx + i];
    }
    particleList[particleIdx].setState(&currentStateBuf[6 * particleIdx]);

}


void SecondOrderRkSolver::solveStep(ParticleSystem *particleSystem, float step) {
    // should do per particle kernel
    // but should claim temp memory for later use
    float *originStateBuf;
    float *currentStateBuf;
    float *derivativeBuf;
    cudaMalloc((void **) &originStateBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    cudaMalloc((void **) &currentStateBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    cudaMalloc((void **) &derivativeBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    // target
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    refreshForce(particleSystem);
    secondOrderRkSolveStepKernel_1<<<blocksPerGrid, threadsPerBlock>>>(particleSystem->m_particleList, originStateBuf,
                                                                       currentStateBuf,
                                                                       derivativeBuf,
                                                                       particleSystem->m_particleSize, step);
    refreshForce(particleSystem);
    secondOrderRkSolveStepKernel_2<<<blocksPerGrid, threadsPerBlock>>>(particleSystem->m_particleList, originStateBuf,
                                                                       currentStateBuf,
                                                                       derivativeBuf,
                                                                       particleSystem->m_particleSize, step);
    cudaFree(originStateBuf);
    cudaFree(currentStateBuf);
    cudaFree(derivativeBuf);
}


__global__ void
fourthOrderRkSolveStepKernel_1(Particle *particleList, float *originStateBuf, float *currentStateBuf,
                               float *derivativeBuf, float *currentDerivativeBuf, int particleSize, float step) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].getState(&originStateBuf[6 * particleIdx]);
    particleList[particleIdx].derivativeEval(&derivativeBuf[6 * particleIdx]);
    for (int i = 0; i < 6; ++i) {
        currentStateBuf[6 * particleIdx + i] =
                originStateBuf[6 * particleIdx + i] + 0.5f * step * derivativeBuf[6 * particleIdx + i];
        currentDerivativeBuf[6 * particleIdx + i] = derivativeBuf[6 * particleIdx + i];
    }
    particleList[particleIdx].setState(&currentStateBuf[6 * particleIdx]);

}

__global__ void
fourthOrderRkSolveStepKernel_2(Particle *particleList, float *originStateBuf, float *currentStateBuf,
                               float *derivativeBuf, float *currentDerivativeBuf, int particleSize, float step) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].derivativeEval(&derivativeBuf[6 * particleIdx]);
    for (int i = 0; i < 6; ++i) {
        currentStateBuf[6 * particleIdx + i] =
                originStateBuf[6 * particleIdx + i] + 0.5f * step * derivativeBuf[6 * particleIdx + i];
        currentDerivativeBuf[6 * particleIdx + i] += 2.0f * derivativeBuf[6 * particleIdx + i];

    }
    particleList[particleIdx].setState(&currentStateBuf[6 * particleIdx]);

}

__global__ void
fourthOrderRkSolveStepKernel_3(Particle *particleList, float *originStateBuf, float *currentStateBuf,
                               float *derivativeBuf, float *currentDerivativeBuf, int particleSize, float step) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }

    particleList[particleIdx].derivativeEval(&derivativeBuf[6 * particleIdx]);
    for (int i = 0; i < 6; ++i) {
        currentStateBuf[6 * particleIdx + i] =
                originStateBuf[6 * particleIdx + i] + step * derivativeBuf[6 * particleIdx + i];
        currentDerivativeBuf[6 * particleIdx + i] += 2.0f * derivativeBuf[6 * particleIdx + i];

    }
    particleList[particleIdx].setState(&currentStateBuf[6 * particleIdx]);

}

__global__ void
fourthOrderRkSolveStepKernel_4(Particle *particleList, float *originStateBuf, float *currentStateBuf,
                               float *derivativeBuf, float *currentDerivativeBuf, int particleSize, float step) {
    int particleIdx = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    if (particleIdx >= particleSize) {
        return;
    }
    particleList[particleIdx].derivativeEval(&derivativeBuf[6 * particleIdx]);
    for (int i = 0; i < 6; ++i) {
        currentDerivativeBuf[6 * particleIdx + i] =
                (currentDerivativeBuf[6 * particleIdx + i] + derivativeBuf[6 * particleIdx + i]) / 6.0f;
        currentStateBuf[6 * particleIdx + i] =
                originStateBuf[6 * particleIdx + i] + step * currentDerivativeBuf[6 * particleIdx + i];
    }

    particleList[particleIdx].setState(&currentStateBuf[6 * particleIdx]);

}


void FourthOrderRkSolver::solveStep(ParticleSystem *particleSystem, float step) {
    // should do per particle kernel
    // but should claim temp memory for later use
    float *originStateBuf;
    float *currentStateBuf;
    float *derivativeBuf;
    float *currentDerivativeStateBuf;
    cudaMalloc((void **) &originStateBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    cudaMalloc((void **) &currentStateBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    cudaMalloc((void **) &derivativeBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    cudaMalloc((void **) &currentDerivativeStateBuf, particleSystem->m_particleSize * 6 * sizeof(float));
    // target
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock(16, 16);
    refreshForce(particleSystem);
    fourthOrderRkSolveStepKernel_1<<<blocksPerGrid, threadsPerBlock>>>(particleSystem->m_particleList, originStateBuf,
                                                                       currentStateBuf,
                                                                       derivativeBuf, currentDerivativeStateBuf,
                                                                       particleSystem->m_particleSize, step);
    refreshForce(particleSystem);
    fourthOrderRkSolveStepKernel_2<<<blocksPerGrid, threadsPerBlock>>>(particleSystem->m_particleList, originStateBuf,
                                                                       currentStateBuf,
                                                                       derivativeBuf, currentDerivativeStateBuf,
                                                                       particleSystem->m_particleSize, step);
    refreshForce(particleSystem);
    fourthOrderRkSolveStepKernel_3<<<blocksPerGrid, threadsPerBlock>>>(particleSystem->m_particleList, originStateBuf,
                                                                       currentStateBuf,
                                                                       derivativeBuf, currentDerivativeStateBuf,
                                                                       particleSystem->m_particleSize, step);
    refreshForce(particleSystem);
    fourthOrderRkSolveStepKernel_4<<<blocksPerGrid, threadsPerBlock>>>(particleSystem->m_particleList, originStateBuf,
                                                                       currentStateBuf,
                                                                       derivativeBuf, currentDerivativeStateBuf,
                                                                       particleSystem->m_particleSize, step);
    cudaFree(originStateBuf);
    cudaFree(currentStateBuf);
    cudaFree(derivativeBuf);
    cudaFree(currentDerivativeStateBuf);
}