cmake_minimum_required(VERSION 3.1.3)
project(ParticleSystem)
set(PROFILE_SUPPORT ON)

include_directories("./include")

find_package(CUDA 10.1 REQUIRED)
if (CUDA_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "")
endif (CUDA_FOUND)

if (PROFILE_SUPPORT)
    add_compile_definitions(PROFILE_PROJECT)
endif ()

set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -DGNU_SOURCE -gencode arch=compute_61,code=sm_61")

set(SOURCE_LIST)
set(HEADER_LIST)

list(APPEND SOURCE_LIST
        src/Collision.cpp
        src/Constraint.cpp
        src/Force.cpp
        src/ParticleSystem.cpp
        src/Solver.cpp
        src/Utility.cpp
        )
list(APPEND HEADER_LIST
        include/Collision.h
        include/Constraint.h
        include/Force.h
        include/ParticleSystem.h
        include/Solver.h
        include/Utility.h
        include/Factory.h
        )
list(APPEND CUDA_SOURCE_LIST
        include/ParticleSystem.cuh
        src/ParticleSystem.cu
        src/Force.cu
        src/Collision.cu
        src/Constraint.cu
        src/Solver.cu
        #        include/Force.cuh
        )

#cuda_add_library(mylib SHARED mycudalib.cu)
cuda_add_executable(ParticleSystem src/main.cpp ${SOURCE_LIST} ${CUDA_SOURCE_LIST} ${HEADER_LIST})
target_compile_options(ParticleSystem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)