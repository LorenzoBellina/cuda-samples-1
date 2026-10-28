/*
 * Versione CPU multi-thread (OpenMP) del passo di simulazione: vedi particleSystem_omp.h.
 *
 * Modello di riferimento [Lez02 parte3 - MIMD, architetture con shared-memory]: i core
 * della CPU sono processori MIMD che condividono la stessa memoria (UMA su un desktop
 * single-socket, NUMA su server multi-socket), quindi tutti i thread vedono gli stessi
 * array e non servono trasferimenti host<->device come nella versione CUDA.
 *
 * Modello di programmazione [Lez04 OpenMP - OpenMP programming model, direttiva parallel]:
 * fork-join. Ogni "#pragma omp parallel for" apre una regione parallela (fork di un team
 * di thread), divide le iterazioni del for tra i thread (work-sharing, direttiva for) e
 * termina con una barrier implicita (join). Le variabili dichiarate dentro il corpo del
 * loop sono private di ogni thread, gli array membri della classe sono shared.
 *
 * Il numero di thread si sceglie con omp_set_num_threads() (opzione -threads=N del driver)
 * o con la variabile d'ambiente OMP_NUM_THREADS.
 *
 * Il parallelismo e' quello "per particella" della versione CUDA: un'iterazione del loop
 * OpenMP corrisponde a un thread CUDA. La differenza e' la granularita': sulla GPU ci sono
 * decine di migliaia di thread leggeri schedulati in warp SIMT [Lez03 parte1], sulla CPU
 * pochi thread pesanti, ognuno dei quali esegue un blocco di iterazioni.
 */

#include "particleSystem_omp.h"

#include <algorithm>
#include <cstring>
#include <math.h>
#include <omp.h>

#include "helper_math.h"

// ---------------------------------------------------------------------------------------
// Funzioni per singola particella: copia host delle funzioni __device__ di
// particles_kernel_impl.cuh (cudaParams in constant memory -> parametro p)
// ---------------------------------------------------------------------------------------

static inline int3 calcGridPos(float3 pos, const SimParams &p)
{
    int3 gridPos;
    gridPos.x = (int)floorf((pos.x - p.worldOrigin.x) / p.cellSize.x);
    gridPos.y = (int)floorf((pos.y - p.worldOrigin.y) / p.cellSize.y);
    gridPos.z = (int)floorf((pos.z - p.worldOrigin.z) / p.cellSize.z);
    return gridPos;
}

static inline uint calcGridHash(int3 gridPos, const SimParams &p)
{
    gridPos.x = gridPos.x & (p.gridSize.x - 1); // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (p.gridSize.y - 1);
    gridPos.z = gridPos.z & (p.gridSize.z - 1);
    return (gridPos.z * p.gridSize.y) * p.gridSize.x + gridPos.y * p.gridSize.x + gridPos.x;
}

static inline float3 collideSpheres(float3           posA,
                                    float3           posB,
                                    float3           velA,
                                    float3           velB,
                                    float            radiusA,
                                    float            radiusB,
                                    float            attraction,
                                    const SimParams &p)
{
    float3 relPos      = posB - posA;
    float  dist        = length(relPos);
    float  collideDist = radiusA + radiusB;
    float3 force       = make_float3(0.0f);

    if (dist < collideDist) {
        float3 norm   = relPos / dist;
        float3 relVel = velB - velA;
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        force = -p.spring * (collideDist - dist) * norm;
        force += p.damping * relVel;
        force += p.shear * tanVel;
        force += attraction * relPos;
    }

    return force;
}

// ---------------------------------------------------------------------------------------

ParticleSystemOmp::ParticleSystemOmp(uint numParticles, const SimParams &params)
    : m_numParticles(numParticles)
    , m_numGridCells(params.numCells)
    , m_params(params)
    , m_pos(numParticles)
    , m_vel(numParticles)
    , m_sortedPos(numParticles)
    , m_sortedVel(numParticles)
    , m_gridParticleHash(numParticles)
    , m_gridParticleIndex(numParticles)
    , m_cellStart(params.numCells)
    , m_cellEnd(params.numCells)
    , m_sortKeys(numParticles)
{
    resetPhaseTiming();
}

void ParticleSystemOmp::setState(const float *pos, const float *vel)
{
    memcpy(m_pos.data(), pos, sizeof(float4) * m_numParticles);
    memcpy(m_vel.data(), vel, sizeof(float4) * m_numParticles);
}

void ParticleSystemOmp::resetPhaseTiming()
{
    for (int p = 0; p < ParticleSystem::NUM_PHASES; p++) {
        m_phaseSec[p] = 0.0;
    }

    m_timedFrames = 0;
}

int ParticleSystemOmp::getPhaseTiming(float *phaseMs) const
{
    for (int p = 0; p < ParticleSystem::NUM_PHASES; p++) {
        phaseMs[p] = (float)(1.0e3 * m_phaseSec[p]);
    }

    return m_timedFrames;
}

// Ogni fase e' una regione parallela distinta, quindi tra due fasi c'e' sempre la barrier
// implicita di fine "parallel for" [Lez04 - direttiva for]: e' cio' che rende corretta la
// sequenza (es. sort deve vedere tutti gli hash calcolati) ed e' l'equivalente CPU della
// sincronizzazione implicita tra kernel lanciati sullo stesso stream [Lez03 parte4].
// omp_get_wtime() dopo la barrier misura quindi il tempo dell'intera fase.
void ParticleSystemOmp::update(float deltaTime)
{
    double t[ParticleSystem::NUM_PHASES + 1];

    t[0] = omp_get_wtime();
    integrate(deltaTime);
    t[1] = omp_get_wtime();
    calcHash();
    t[2] = omp_get_wtime();
    sortParticles();
    t[3] = omp_get_wtime();
    reorderDataAndFindCellStart();
    t[4] = omp_get_wtime();
    collide();
    t[5] = omp_get_wtime();

    for (int p = 0; p < ParticleSystem::NUM_PHASES; p++) {
        m_phaseSec[p] += t[p + 1] - t[p];
    }

    m_timedFrames++;
}

// Fase 1 - integrazione (integrate_functor con thrust::for_each nella versione CUDA).
// Embarrassingly parallel [Lez01]: ogni iterazione legge e scrive solo la particella i,
// non ci sono dipendenze tra iterazioni ne' race condition.
// Il lavoro per iterazione e' costante, quindi schedule(static) [Lez04 - Ripartire le
// iterazioni tra i thread]: un chunk contiguo di ~N/T iterazioni per thread, nessun
// overhead di assegnazione dinamica e accessi sequenziali in memoria per ogni thread.
// Nota: integrate e calcHash si potrebbero fondere in un'unica regione parallela con due
// "omp for" e nowait [Lez04 - Inibire la barrier in uscita], perche' con due loop static
// di uguale lunghezza la stessa iterazione i va allo stesso thread; qui restano separate
// per poter misurare il tempo di ogni fase.
void ParticleSystemOmp::integrate(float deltaTime)
{
    const SimParams &p = m_params;
    const int        n = (int)m_numParticles;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float3 pos = make_float3(m_pos[i]);
        float3 vel = make_float3(m_vel[i]);

        vel += p.gravity * deltaTime;
        vel *= p.globalDamping;

        pos += vel * deltaTime;

        if (pos.x > 1.0f - p.particleRadius) {
            pos.x = 1.0f - p.particleRadius;
            vel.x *= p.boundaryDamping;
        }

        if (pos.x < -1.0f + p.particleRadius) {
            pos.x = -1.0f + p.particleRadius;
            vel.x *= p.boundaryDamping;
        }

        if (pos.y > 1.0f - p.particleRadius) {
            pos.y = 1.0f - p.particleRadius;
            vel.y *= p.boundaryDamping;
        }

        if (pos.z > 1.0f - p.particleRadius) {
            pos.z = 1.0f - p.particleRadius;
            vel.z *= p.boundaryDamping;
        }

        if (pos.z < -1.0f + p.particleRadius) {
            pos.z = -1.0f + p.particleRadius;
            vel.z *= p.boundaryDamping;
        }

        if (pos.y < -1.0f + p.particleRadius) {
            pos.y = -1.0f + p.particleRadius;
            vel.y *= p.boundaryDamping;
        }

        m_pos[i] = make_float4(pos, m_pos[i].w);
        m_vel[i] = make_float4(vel, m_vel[i].w);
    }
}

// Fase 2 - hash della cella (calcHashD). Stesso pattern della fase 1: map indipendente
// per particella, schedule(static).
void ParticleSystemOmp::calcHash()
{
    const int n = (int)m_numParticles;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int3 gridPos           = calcGridPos(make_float3(m_pos[i]), m_params);
        m_gridParticleHash[i]  = calcGridHash(gridPos, m_params);
        m_gridParticleIndex[i] = i;
    }
}

// Fase 3 - ordinamento per hash (thrust::sort_by_key, radix sort, nella versione CUDA).
// Questa fase e' lasciata volutamente SERIALE: e' la frazione s della legge di Amdahl
// [Lez01 - Amdahl's law e strong scalability] e limita lo speedup ottenibile a
// 1 / (s + (1-s)/T). Il tempo per fase stampato dal driver permette di stimare s e di
// confrontare il limite teorico con lo speedup misurato.
// Solo l'impacchettamento delle chiavi (hash << 32 | index) e lo spacchettamento sono
// parallelizzati (map indipendenti). Ordinare la coppia (hash, index) da' lo stesso ordine
// del radix sort stabile della GPU: le particelle di una cella restano in ordine di
// indice, quindi l'ordine delle somme in collide() coincide con quello della GPU.
void ParticleSystemOmp::sortParticles()
{
    const int n = (int)m_numParticles;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        m_sortKeys[i] = ((unsigned long long)m_gridParticleHash[i] << 32) | m_gridParticleIndex[i];
    }

    std::sort(m_sortKeys.begin(), m_sortKeys.end()); // parte seriale

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        m_gridParticleHash[i]  = (uint)(m_sortKeys[i] >> 32);
        m_gridParticleIndex[i] = (uint)(m_sortKeys[i] & 0xffffffffu);
    }
}

// Fase 4 - riordino e inizio/fine delle celle (cudaMemset + reorderDataAndFindCellStartD).
// Un'unica regione parallela con due "omp for": i thread vengono creati una volta sola
// (meno overhead di fork/join) e la barrier implicita alla fine del primo for [Lez04 -
// direttiva for] garantisce che tutte le celle siano marcate vuote prima che il secondo
// loop scriva cellStart/cellEnd. Qui il nowait NON si puo' usare: il secondo loop scrive
// celle arbitrarie, non solo quelle azzerate dallo stesso thread.
// Nel kernel CUDA ogni blocco carica in shared memory l'hash del vicino precedente per
// non leggerlo due volte da global memory [Lez03 parte3c - shared memory come cache
// gestita via software]; sulla CPU questo riuso lo fa automaticamente la cache hardware
// [Lez02 parte1 - Cache], quindi basta leggere m_gridParticleHash[i - 1].
// Nessuna race condition [Lez03 parte5 - race condition, sezione critica]: l'inizio di una
// cella e' scritto solo dalla prima particella della cella e la fine solo dalla prima
// della cella successiva, quindi ogni locazione di cellStart/cellEnd ha un solo scrittore
// e non servono critical/atomic [Lez04 - Sincronizzazione].
void ParticleSystemOmp::reorderDataAndFindCellStart()
{
    const int n        = (int)m_numParticles;
    const int numCells = (int)m_numGridCells;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int c = 0; c < numCells; c++) {
            m_cellStart[c] = 0xffffffff;
        }

#pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            uint hash = m_gridParticleHash[i];

            if (i == 0 || hash != m_gridParticleHash[i - 1]) {
                m_cellStart[hash] = i;

                if (i > 0) {
                    m_cellEnd[m_gridParticleHash[i - 1]] = i;
                }
            }

            if (i == n - 1) {
                m_cellEnd[hash] = i + 1;
            }

            // gather: lettura indiretta (non sequenziale), scrittura sequenziale
            uint sortedIndex = m_gridParticleIndex[i];
            m_sortedPos[i]   = m_pos[sortedIndex];
            m_sortedVel[i]   = m_vel[sortedIndex];
        }
    }
}

// Fase 5 - collisioni (collideD). E' la fase con piu' calcolo ed e' la versione "base"
// (gather, sync-free) del confronto del punto 3: ogni iterazione calcola la forza sulla
// SOLA particella i leggendo le 27 celle vicine e scrive solo la velocita' di i, quindi le
// scritture di thread diversi non vanno mai sulla stessa locazione e non servono
// atomic/critical [Lez04 - direttive critical e atomic]. Il prezzo e' che ogni coppia
// (i, j) viene calcolata due volte, una da i e una da j.
//
// Il lavoro per iterazione NON e' costante: dipende da quante particelle ci sono nelle
// 27 celle vicine (zone dense vs vuote, la pila sul fondo per effetto della gravita').
// Per questo lo scheduling e' schedule(runtime) [Lez04 - Ripartire le iterazioni tra i
// thread]: la politica (static, dynamic, guided e il chunk) si sceglie a run time con
// omp_set_schedule() (opzione -schedule del driver) o con OMP_SCHEDULE, senza
// ricompilare, cosi' da confrontare:
//  - static: overhead minimo ma possibile sbilanciamento del carico tra i thread;
//  - dynamic,chunk: bilanciamento migliore, costo di assegnazione per ogni chunk;
//  - guided,chunk: chunk decrescenti, compromesso tra i due.
// Particelle consecutive sono nella stessa zona di spazio (sono ordinate per cella), quindi
// chunk contigui mantengono anche la localita' in cache delle celle vicine.
//
// Scrittura m_vel[originalIndex]: l'indice originale e' "sparso", quindi thread diversi
// possono scrivere float4 vicini nella stessa linea di cache (64 byte = 4 float4) e
// causare false sharing (invalidazioni dovute alla coerenza delle cache [Lez02 parte3 -
// cache coherent NUMA]). E' un costo di prestazioni, non di correttezza.
void ParticleSystemOmp::collide()
{
    const SimParams &p = m_params;
    const int        n = (int)m_numParticles;

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        // variabili private del thread: equivalgono ai registri del thread CUDA
        float3 pos     = make_float3(m_sortedPos[i]);
        float3 vel     = make_float3(m_sortedVel[i]);
        int3   gridPos = calcGridPos(pos, p);
        float3 force   = make_float3(0.0f);

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    uint gridHash   = calcGridHash(gridPos + make_int3(x, y, z), p);
                    uint startIndex = m_cellStart[gridHash];

                    if (startIndex == 0xffffffff) { // cell is empty
                        continue;
                    }

                    uint endIndex = m_cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        if (j != (uint)i) { // check not colliding with self
                            force += collideSpheres(pos,
                                                    make_float3(m_sortedPos[j]),
                                                    vel,
                                                    make_float3(m_sortedVel[j]),
                                                    p.particleRadius,
                                                    p.particleRadius,
                                                    p.attraction,
                                                    p);
                        }
                    }
                }
            }
        }

        // collide with cursor sphere
        force += collideSpheres(
            pos, p.colliderPos, vel, make_float3(0.0f), p.particleRadius, p.colliderRadius, 0.0f, p);

        // write new velocity back to original unsorted location
        uint originalIndex = m_gridParticleIndex[i];
        m_vel[originalIndex] = make_float4(vel + force, 0.0f);
    }
}
