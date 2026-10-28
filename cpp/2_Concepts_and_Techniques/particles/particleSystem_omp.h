/*
 * Versione CPU multi-thread (OpenMP) del passo di simulazione di ParticleSystem.
 *
 * Riproduce una per una le fasi di ParticleSystem::update() (integrate, hash, sort,
 * reorder, collide) con gli stessi algoritmi dei kernel CUDA, cosi' che il confronto
 * CPU/GPU e lo studio di scalabilita' al variare dei thread siano fatti a parita' di
 * algoritmo (Step 1 del progetto).
 */

#ifndef __PARTICLESYSTEM_OMP_H__
#define __PARTICLESYSTEM_OMP_H__

#include <vector>

#include "particleSystem.h"
#include "particles_kernel.cuh"
#include "vector_types.h"

class ParticleSystemOmp
{
public:
    ParticleSystemOmp(uint numParticles, const SimParams &params);

    // stato iniziale (float4 per particella), copiato da ParticleSystem dopo reset()
    void setState(const float *pos, const float *vel);

    void update(float deltaTime);

    const float *getPos() const { return (const float *)m_pos.data(); }

    // tempi per fase (omp_get_wtime) accumulati dall'ultima resetPhaseTiming()
    void resetPhaseTiming();
    int  getPhaseTiming(float *phaseMs) const; // ms totali per fase, ritorna i frame misurati

private:
    void integrate(float deltaTime);
    void calcHash();
    void sortParticles();
    void reorderDataAndFindCellStart();
    void collide();

    uint      m_numParticles;
    uint      m_numGridCells;
    SimParams m_params;

    // stessi array (e stesso layout float4, AoS) della versione CUDA
    std::vector<float4> m_pos, m_vel;
    std::vector<float4> m_sortedPos, m_sortedVel;
    std::vector<uint>   m_gridParticleHash, m_gridParticleIndex;
    std::vector<uint>   m_cellStart, m_cellEnd;
    std::vector<unsigned long long> m_sortKeys;

    double m_phaseSec[ParticleSystem::NUM_PHASES];
    int    m_timedFrames;
};

#endif // __PARTICLESYSTEM_OMP_H__
