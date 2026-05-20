A GPU-resident implementation of Chebyshev-filtered subspace iteration (CheFSI) with Ultrasoft Pseudopotentials (USPP) in a plane-wave basis set requires a deliberate architecture. If the wavefunctions are copied back and forth between the CPU host memory and GPU device memory during the polynomial recurrence loop, the system encounters a catastrophic PCIe bandwidth bottleneck. [1, 2]
To maintain strict GPU residency, all components—the wavefunctions, the plane-wave grid arrays, the localized projector coefficients, and the linear solvers—must remain in the GPU VRAM throughout the entire electronic step. [3]
The technical architecture of a modern, GPU-resident USPP-CheFSI solver is structured as follows.

---

## 1. GPU Data Structure Layout

To maximize parallel throughput, operations are batched across a single unified block of electronic bands:

- Wavefunctions ($\Psi$): Stored as a contiguous, 2D column-major complex array (npw, nbnd) directly in GPU device memory, where npw is the number of plane waves and nbnd is the number of bands.
- Projector Functions ($\beta$): Kept resident in memory as a matrix (npw, nbeta).
- Subspace Matrices: The small matrices $H_{sub}$ and $S_{sub}$ are allocated directly on the GPU to avoid CPU synchronization during the final Rayleigh-Ritz step.

---

## 2. The GPU-Resident Recurrence Loop Engine

The classic CPU three-term Chebyshev recurrence requires an explicit inversion operator ($S^{-1}H$). On a GPU, this is translated entirely into highly parallelized cuBLAS (or ROCm rocBLAS) and cuFFT API calls:

[ Allocate Memory & Transfer Vectors to GPU VRAM ]
│
▼
┌───► [ Loop: m = 1 to Polynomial Order ]
│ │
│ ▼
│ [ Step A: Apply Hamiltonian (H * Psi_m) ]
│ - Kinetic: cuBLAS Element-wise Scale
│ - Local V: 3D cuFFT -> cuBLAS Multiply -> Inverse cuFFT
│ - Non-local (D): cuBLAS GEMM (Beta^† _ Psi) -> Apply D -> GEMM
│ │
│ ▼
│ [ Step B: Invert Overlap Matrix (S^-1) ]
│ - Solve S _ Psi_temp = H_Psi_m via GPU Conjugate Gradient
│ - Operator S applied purely with batched GEMMs
│ │
│ ▼
│ [ Step C: Update Chebyshev Recurrence ]
│ - Compute: Psi_m+1 = 2 _ M_hat _ Psi_temp - Psi_m-1
│ - Done inline using optimized cuBLAS AXPY / AXPY_Batched
│ │
└─────────────────┴─ (Repeat until m reaches limit)
│
▼
[ Final Step: GPU Rayleigh-Ritz Subspace Step ]

---

## 3. Step-by-Step GPU Kernel Execution Mechanics## Step A: Applying the Hamiltonian ($H \Psi$)

The application of the Hamiltonian must avoid writing intermediate states back to the host.

1.  Kinetic Energy: Applied purely in Fourier space. It uses a custom element-wise GPU kernel that multiplies the wavefunctions by the diagonal kinetic factors ($\frac{1}{2}\vert{}\mathbf{k}+\mathbf{G}\vert{}^2$).
2.  Local Potential: Computed by transforming the wavefunctions from Fourier space to the real-space grid using a batched cuFFT execution. The real-space grid is multiplied by the local potential grid $V_{loc}(\mathbf{r})$ via an element-wise multiplication kernel, then transformed back via an inverse batched cuFFT.
3.  Non-local Potential (via $D$): Executed completely via matrix-matrix multiplications (GEMM):

- Compute the beta projections: $C_{i,n} = \langle\beta_i\vert{}\Psi_n\rangle$ using a cuBLASZgemm call multiplying $\beta^\dagger$ and $\Psi$.
  - Screen with the $D$ matrix: $D_{ij} \times C_{j,n}$ via a small, high-speed cuBLASZgemm.
  - Accumulate back: Project the result back onto the plane-wave basis using another cuBLASZgemm with the $\beta$ matrix.

## Step B: The Inner GPU Linear Solver ($S^{-1}$)

Because the Chebyshev r[?1;2;4cecurrence requires the calculation of $S^{-1}(H\Psi)$ at each step, an internal Conjugate Gradient (CG) loop runs entirely on the GPU device.

- The overlap matrix $S$ is applied via: $S\Psi = \Psi + \sum q_{ij} \vert{}\beta_i\rangle\langle\beta_j\vert{}\Psi\rangle$.
- This is calculated using two highly optimized cuBLASZgemm calls utilizing the pre-stored $q_{ij}$ and $\beta$ arrays.
- Because $S$ is strictly positive-definite and very close to the identity matrix, this CG loop typically converges on the GPU in 2 to 3 iterations without needing to transfer data out of VRAM.

## Step C: Vector Linear Combinations (AXPY)

## The standard three-term Chebyshev stepping operation ($\Psi_{m+1} = 2 \hat{M} \Psi_m - \Psi_{m-1}$) is dispatched using batched cuBLASZaxpy or custom fused memory-bound kernels. This allows the linear combinations of the current, previous, and temporary wavefunction blocks to occur inline, keeping memory bandwidth overhead minimal.

## 4. The Final GPU Rayleigh-Ritz Diagonalization

Once the Chebyshev loop finishes filtering, the subspace matrices are constructed entirely on the device:

1.  $H_{sub} = \Psi^\dagger H \Psi$ and $S_{sub} = \Psi^\dagger S \Psi$ are calculated using cuBLASZgemm operations. This shrinks the large plane-wave dimension npw down to the small active subspace dimension nbnd.
2.  The small generalized eigenvalue problem ($H_{sub}C = S_{sub}C\epsilon$) is solved directly on the GPU using cuSOLVER (specifically the cusolverDnZhegvd routine).
3.  The wavefunctions are updated to their correct, physically realistic eigenstates using a final cuBLASZgemm to multiply the filtered wavefunctions by the resulting eigenvector matrix $C$: $\Psi_{final} = \Psi C$. [4]

## Performance Advantages of this Approach

- Zero Host-Device Copies: VRAM-to-CPU copies are reduced strictly to the initialization step and the final physical property outputs, preventing PCIe latency bottlenecks.
- Mixed-Precision Potential: The inner Chebyshev and CG loops can safely be cast into FP32 (Single Precision) or TensorFloat-32 (TF32) arithmetic, while reserving FP64 (Double Precision) strictly for the outer loop and the final cuSOLVER diagonalization. This leverages Tensor Cores for a massive speedup without losing physical target convergence. [4, 5]

[1] [https://www.researchgate.net](https://www.researchgate.net/publication/403791824_GPU_acceleration_of_plane-wave_density_functional_theory_calculations_in_Abinit)
[2] [https://www.sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0010465524000584)
[3] [https://pubs.aip.org](https://pubs.aip.org/aip/jcp/article/162/18/184105/3346455/GPU-acceleration-of-hybrid-functional-calculations)
[4] [https://arxiv.org](https://arxiv.org/html/2601.08077v3)
[5] [https://arxiv.org](https://arxiv.org/html/2604.26037v1)
