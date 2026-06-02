// ---------------------------------------------------------------------------
// Integration tests for apply_preconditioner — TPA preconditioner apply step
// ---------------------------------------------------------------------------
//
// Tests outcome C1-C3 from TASK-D-2:
//
//   C1: For norm-conserving PP (no augmentation): precon = (H−E·ψ) · R(G)
//   C2: All values finite (no NaN, no Inf)
//   C3: Output dimensions match input (n_pw × n_bands)
//
// The discriminator: the OLD wrong preconditioner used 1/(T−λ) which
// diverges when T ≈ λ. The correct TPA formula always produces finite
// values bounded by R(G) ∈ [0, 1]. Any test that verifies the output
// matches the explicit elementwise formula (hpsi − e·psi) · R(G) will
// catch the wrong formula.
// ---------------------------------------------------------------------------

use chemrust_scf::apply_preconditioner;
use chemrust_scf::device::CudaComplex;
use chemrust_scf::PreconditionerVector;
use chemrust_scf::PwCoefficients;
use chemrust_scf::TpaPreconditioner;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;

// -----------------------------------------------------------------------
// Helper: check GPU availability
// -----------------------------------------------------------------------

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
}

// -----------------------------------------------------------------------
// C1–C3: TPA preconditioner apply with no augmentation
// -----------------------------------------------------------------------

/// C1 + C2 + C3: TPA preconditioner apply matches elementwise formula.
///
/// Creates synthetic psi, hpsi, eigenvalues, and R vector on GPU, calls
/// `apply_preconditioner`, downloads the result, and verifies elementwise
/// equality with the manual CPU computation:
///
///   expected[G,b] = (hpsi[G,b] - e[b] * psi[G,b]) * R(G)
///
/// This is the norm-conserving PP path (no NL correction). The test
/// verifies that the fused GPU kernel produces bit-identical results
/// to the CPU reference.
#[test]
fn test_tpa_apply_no_augment() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    // Parameters (same as unit test in preconditioner.rs for cross-validation)
    let n_pw = 4;
    let n_bands = 3;

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // Create TpaPreconditioner (compiles CUDA kernels via NVRTC)
    let tpa_precon = TpaPreconditioner::new(&ctx).expect("TpaPreconditioner::new");

    // Synthetic eigenvalues and R(G) vector
    let eigenvalues_cpu: Vec<f64> = vec![0.5, 1.0, 1.5];
    let r_vector_cpu: Vec<f64> = vec![0.8, 0.6, 0.4, 0.2];

    // Synthetic psi and hpsi data (band-major: i = b * n_pw + g)
    //
    // Band 0: psi=(1,0), hpsi=(2,0.5)
    // Band 1: psi=(0,1), hpsi=(-1,2)
    // Band 2: psi=(0.5,-0.5), hpsi=(0,0)
    //
    // The data from each band is contiguous in PW order.
    let psi_cpu: Vec<CudaComplex> = vec![
        // Band 0 (b=0): psi = [1+0i, 0-1i, 0.5+0.5i, -0.5+1i]
        CudaComplex { x: 1.0, y: 0.0 },
        CudaComplex { x: 0.0, y: -1.0 },
        CudaComplex { x: 0.5, y: 0.5 },
        CudaComplex { x: -0.5, y: 1.0 },
        // Band 1 (b=1): psi = [0+1i, 1+0i, -1+1i, 0.5-0.5i]
        CudaComplex { x: 0.0, y: 1.0 },
        CudaComplex { x: 1.0, y: 0.0 },
        CudaComplex { x: -1.0, y: 1.0 },
        CudaComplex { x: 0.5, y: -0.5 },
        // Band 2 (b=2): psi = [0.5-0.5i, 0+0.5i, 1+0i, -1+0i]
        CudaComplex { x: 0.5, y: -0.5 },
        CudaComplex { x: 0.0, y: 0.5 },
        CudaComplex { x: 1.0, y: 0.0 },
        CudaComplex { x: -1.0, y: 0.0 },
    ];
    // Verify total length
    assert_eq!(psi_cpu.len(), n_pw * n_bands);

    let hpsi_cpu: Vec<CudaComplex> = vec![
        // Band 0 (b=0): hpsi = [2+0.5i, -1+2i, 0+0i, 3+1i]
        CudaComplex { x: 2.0, y: 0.5 },
        CudaComplex { x: -1.0, y: 2.0 },
        CudaComplex { x: 0.0, y: 0.0 },
        CudaComplex { x: 3.0, y: 1.0 },
        // Band 1 (b=1): hpsi = [1-0.5i, 0.5+1.5i, 2-1i, -1+0i]
        CudaComplex { x: 1.0, y: -0.5 },
        CudaComplex { x: 0.5, y: 1.5 },
        CudaComplex { x: 2.0, y: -1.0 },
        CudaComplex { x: -1.0, y: 0.0 },
        // Band 2 (b=2): hpsi = [-1+1i, 1-2i, 0.5+0.5i, 2+1.5i]
        CudaComplex { x: -1.0, y: 1.0 },
        CudaComplex { x: 1.0, y: -2.0 },
        CudaComplex { x: 0.5, y: 0.5 },
        CudaComplex { x: 2.0, y: 1.5 },
    ];
    assert_eq!(hpsi_cpu.len(), n_pw * n_bands);

    // Upload data to GPU
    let psi_dev = PwCoefficients::new(
        stream
            .clone_htod(&psi_cpu)
            .expect("upload psi to GPU"),
    );
    let hpsi_dev = PwCoefficients::new(
        stream
            .clone_htod(&hpsi_cpu)
            .expect("upload hpsi to GPU"),
    );
    let eigenvalues_dev: CudaSlice<f64> = stream
        .clone_htod(&eigenvalues_cpu)
        .expect("upload eigenvalues to GPU");
    let r_slice: CudaSlice<f64> = stream
        .clone_htod(&r_vector_cpu)
        .expect("upload r_vector to GPU");
    let r_vec = PreconditionerVector::new(r_slice);

    // Call apply_preconditioner
    let precon = unsafe {
        apply_preconditioner()
            .psi(&psi_dev)
            .hpsi(&hpsi_dev)
            .eigenvalues(&eigenvalues_dev)
            .r_vector(&r_vec)
            .tpa_preconditioner(&tpa_precon)
            .n_bands(n_bands)
            .n_pw(n_pw)
            .stream(&stream)
            .call()
            .expect("apply_preconditioner failed")
    };

    // Download result from GPU
    let result: Vec<CudaComplex> = stream
        .clone_dtoh(&*precon)
        .expect("download precon from GPU");

    // C3: Output dimensions match input (n_pw × n_bands)
    assert_eq!(
        result.len(),
        n_pw * n_bands,
        "C3 FAIL: output length {} != expected {}",
        result.len(),
        n_pw * n_bands,
    );

    // Compute expected output on CPU: expected[G,b] = (hpsi[G,b] - e[b]*psi[G,b]) * R[G]
    let mut expected = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pw * n_bands];
    for b in 0..n_bands {
        let e = eigenvalues_cpu[b];
        for g in 0..n_pw {
            let i = b * n_pw + g;
            let r = r_vector_cpu[g];
            let hx = hpsi_cpu[i].x;
            let hy = hpsi_cpu[i].y;
            let px = psi_cpu[i].x;
            let py = psi_cpu[i].y;
            expected[i].x = (hx - e * px) * r;
            expected[i].y = (hy - e * py) * r;
        }
    }

    // C1: Verify elementwise agreement with CPU reference
    // C2: All values finite (no NaN, no Inf) — covered by the comparison
    let mut max_diff_x = 0.0_f64;
    let mut max_diff_y = 0.0_f64;
    let mut max_diff_idx = 0_usize;
    for i in 0..(n_pw * n_bands) {
        assert!(
            result[i].x.is_finite(),
            "C2 FAIL: result[{i}].x is not finite: {}",
            result[i].x,
        );
        assert!(
            result[i].y.is_finite(),
            "C2 FAIL: result[{i}].y is not finite: {}",
            result[i].y,
        );
        let diff_x = (result[i].x - expected[i].x).abs();
        let diff_y = (result[i].y - expected[i].y).abs();
        if diff_x > max_diff_x {
            max_diff_x = diff_x;
            max_diff_idx = i;
        }
        if diff_y > max_diff_y {
            max_diff_y = diff_y;
        }
        let b = i / n_pw;
        let g = i % n_pw;
        assert!(
            diff_x < 1e-14,
            "C1 FAIL: element [{b},{g}] x mismatch: expected {exp:.6e}, got {got:.6e}, diff={d:.2e}",
            b = b,
            g = g,
            exp = expected[i].x,
            got = result[i].x,
            d = diff_x,
        );
        assert!(
            diff_y < 1e-14,
            "C1 FAIL: element [{b},{g}] y mismatch: expected {exp:.6e}, got {got:.6e}, diff={d:.2e}",
            b = b,
            g = g,
            exp = expected[i].y,
            got = result[i].y,
            d = diff_y,
        );
    }
    eprintln!(
        "[PASS] TPA apply: max_diff_x={:.2e} max_diff_y={:.2e} at index {}",
        max_diff_x, max_diff_y, max_diff_idx,
    );
}
