// ---------------------------------------------------------------------------
// Mechanism test: cuBLAS host-pointer results and stream choice
// ---------------------------------------------------------------------------
//
// `cublasZdotc_v2` in host-pointer mode enqueues a dot kernel plus a
// small D2H staging copy of the 16-byte result. The call returns after
// enqueueing. Two hazards interact:
//
//   1. Host-read hazard: a host read of the result pointer before the
//      kernel completes sees the stale value.
//   2. Internal-stream hazard: on the LEGACY default stream, cuBLAS may
//      run work on internal streams, so a zdotc enqueued "after" a GEMM
//      is not guaranteed to run after it. It can read GEMM output mid-
//      write.
//
// This probe makes both visible with a data-dependent setup:
//
//   - a = 6000x6000 matrix of 1.0. GEMM c = a*a settles to 6000 per
//     element.
//   - zdotc(c, c) enqueued right after the GEMM.
//   - Immediate host read vs the settled value.
//
// On a fresh non-blocking stream the zdotc is stream-ordered after the
// GEMM: the immediate read is stale (0.0), the post-sync re-read
// matches. On the legacy default stream the behavior may differ (cuBLAS
// internal streams), which is exactly the out-of-order class behind the
// block Davidson "load-bearing wall". See
// docs/load-bearing-diagnostic-overhead.md.
//
// Companion probe: tests/async_dtoh_probe.rs shows `clone_dtoh` into a
// plain (pageable) host Vec blocks the host until the DMA data is
// available. So D2H of plain Vecs is NOT the racy class here.

use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::sys::{cublasZdotc_v2, cuDoubleComplex};
use cudarc::driver::{CudaContext, CudaStream, DevicePtr};
use cudarc::cufft::sys::double2 as CudaComplex;

use chemrust_scf::device::blas::{op, BlasHandle, ZgemmConfig};

fn gpu_ctx() -> Option<Arc<CudaContext>> {
    match CudaContext::new(0) {
        Ok(ctx) => Some(ctx),
        Err(_) => {
            eprintln!("no GPU, skip");
            None
        }
    }
}

/// GEMM delay + data-dependent zdotc + blocking read. cuBLAS in host-
/// pointer mode blocks the CPU until the dot completes (cuBLAS doc
/// 2.1.5). The wall time of the zdotc call therefore reveals the stream
/// ordering:
///
///   enqueue: GEMM1 (c), GEMM2 (c2), zdotc(c, c)
///
/// In-order on `stream`: zdotc waits for GEMM1 + GEMM2.
/// Out-of-order (internal streams): zdotc only needs GEMM1, and runs in
/// parallel with GEMM2. The call returns ~T1 instead of ~T1 + T2.
fn probe(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, label: &str) -> f64 {
    let m: i32 = 6000;
    let nm = (m as usize).pow(2);
    let a_host: Vec<CudaComplex> = vec![CudaComplex { x: 1.0, y: 0.0 }; nm];
    let blas = BlasHandle::new(stream.clone()).expect("blas");
    let handle = blas.raw_handle();
    let a = stream.clone_htod(&a_host).expect("H2D a");
    let mut c = stream.alloc_zeros::<CudaComplex>(nm).expect("alloc c");
    let mut c2 = stream.alloc_zeros::<CudaComplex>(nm).expect("alloc c2");
    stream.synchronize().expect("prime");
    let expected = 6000.0_f64 * 6000.0 * nm as f64; // sum_ij c[i]*c[j]

    let t0 = Instant::now();
    unsafe {
        blas.gemm_c64(gemm_cfg(m), &a, &a, &mut c).expect("gemm1");
        blas.gemm_c64(gemm_cfg(m), &a, &a, &mut c2).expect("gemm2");
    }
    let enq_s = t0.elapsed().as_secs_f64();

    let (c_ptr, _) = c.device_ptr(stream);
    let mut dot = CudaComplex { x: 0.0, y: 0.0 };
    let t1 = Instant::now();
    unsafe {
        cublasZdotc_v2(
            handle,
            nm as i32,
            c_ptr as *const cuDoubleComplex,
            1,
            c_ptr as *const cuDoubleComplex,
            1,
            &mut dot as *mut _ as *mut cuDoubleComplex,
        )
        .result()
        .expect("zdotc call");
    }
    // Host-pointer mode: the call blocks until the dot + result copy.
    let zdotc_s = t1.elapsed().as_secs_f64();
    let settled = (dot.x - expected).abs() <= expected * 1e-9;
    let read_val = dot.x;
    // In-order: zdotc_s ≈ T1 + T2 (both GEMMs ahead on the stream).
    // Out-of-order: zdotc_s ≈ T1 only (GEMM2 overlapped on an internal
    // stream). Compare against the enqueue wall time as a T1 proxy.
    eprintln!(
        "[zdotc_race] {label}: enq2gemm={enq_s:.2}s zdotc_block={zdotc_s:.2}s \
         settled={settled} read={read_val:.6e} expected={expected:.6e}"
    );
    zdotc_s
}

fn gemm_cfg(m: i32) -> ZgemmConfig {
    ZgemmConfig {
        transa: op::N,
        transb: op::N,
        m,
        n: m,
        k: m,
        alpha: CudaComplex { x: 1.0, y: 0.0 },
        lda: m,
        ldb: m,
        ldc: m,
        beta: CudaComplex { x: 0.0, y: 0.0 },
    }
}

#[test]
fn zdotc_data_dependent_race_probe() {
    let ctx = match gpu_ctx() {
        Some(ctx) => ctx,
        None => return,
    };
    let legacy = probe(&ctx, &ctx.default_stream(), "legacy-stream");
    let fresh = ctx.new_stream().expect("new stream");
    let fresh_block = probe(&ctx, &fresh, "fresh-stream");
    eprintln!(
        "[zdotc_race] legacy_block={legacy:.2}s fresh_block={fresh_block:.2}s \
         (block ≈ T1+T2 means in-order; block ≈ T1 means GEMM2 overlapped \
         on an internal stream)"
    );
}
