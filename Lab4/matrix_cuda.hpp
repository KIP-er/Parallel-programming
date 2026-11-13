#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int initializeCuda(char* errorMessage, int errorMessageSize);

int multiplyCuda(
    const double* hostA,
    const double* hostB,
    double* hostC,
    int matrixSize,
    int blockX,
    int blockY,
    double* kernelTimeMs,
    double* totalTimeMs,
    char* errorMessage,
    int errorMessageSize);

#ifdef __cplusplus
}
#endif
