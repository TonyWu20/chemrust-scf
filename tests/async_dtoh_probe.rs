// ---------------------------------------------------------------------------
// Probe: does `clone_dtoh` block the host, and on which stream?
// ---------------------------------------------------------------------------
//
// cudarc docs claim `clone_dtoh` enqueues `cuMemcpyDtoHAsync` and returns
// immediately. Earlier experiment on the LEGACY default stream showed the
// D2H enqueue blocking the host for the whole GEMM delay. This probe
// separates the two stream kinds:
//
//   * fresh stream (cuStreamCreate): expected truly async D2H.
//   * legacy default stream: measure whether the D2H enqueue blocks.
//
// Run: cargo test --release --test async_dtoh_probe -- --nocapture

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice};
use cudarc::cufft::sys::double2 as CudaComplex;

use chemrust_scf::device::blas::{op, BlasHandle, ZgemmConfig};

fn make_expected(n: usize) -> Vec<CudaComplex> {
    (0..n)
        .map(|i| CudaComplex {
            x: (i % 997) as f64 * 1e-3,
            y: -((i % 881) as f64 * 1e-3),
        })
        .collect()
}

fn count_mismatch(a: &[CudaComplex], b: &[CudaComplex]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.x != y.x || x.y != y.y)
        .count()
}

fn enqueue_gemm_delay(stream: &Arc<cudarc::driver::CudaStream>) {
    let m: i32 = 6000;
    let a = stream.alloc_zeros::<CudaComplex>((m as usize).pow(2)).expect("alloc a");
    let mut c = stream.alloc_zeros::<CudaComplex>((m as usize).pow(2)).expect("alloc c");
    let blas = BlasHandle::new(stream.clone()).expect("blas");
    unsafe {
        blas.gemm_c64(
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
            },
            &a,
            &a,
            &mut c,
        )
        .expect("gemm");
    }
}

fn probe(
    label: &str,
    stream: &Arc<cudarc::driver::CudaStream>,
) {
    let n = 1_000_000;
    let expected = make_expected(n);
    let dev: CudaSlice<CudaComplex> = stream.clone_htod(&expected).expect("H2D");
    stream.synchronize().expect("prime sync");

    // A: D2H with NO preceding GPU delay.
    let t0 = Instant::now();
    let v = stream.clone_dtoh(&dev).expect("D2H");
    let enq_us = t0.elapsed().as_micros();
    let mismatch = count_mismatch(&v, &expected);
    eprintln!(
        "[probe] {label}: no-delay d2h enqueue={enq_us}us \
         immediate_read_mismatch={mismatch}/{n}"
    );

    // B: D2H behind a GEMM delay.
    let t1 = Instant::now();
    enqueue_gemm_delay(stream);
    let gemm_us = t1.elapsed().as_micros();
    let t2 = Instant::now();
    let v2 = stream.clone_dtoh(&dev).expect("D2H behind GEMM");
    let enq2_us = t2.elapsed().as_micros();
    let mismatch2 = count_mismatch(&v2, &expected);
    eprintln!(
        "[probe] {label}: behind-gemm gemm_enqueue={gemm_us}us \
         d2h_enqueue={enq2_us}us immediate_read_mismatch={mismatch2}/{n}"
    );
    stream.synchronize().expect("sync");
}

#[test]
fn probe_dtoh_blocking() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("no GPU, skip");
            return;
        }
    };
    probe("legacy-stream", &ctx.default_stream());
    let fresh = ctx.new_stream().expect("new stream");
    probe("fresh-stream", &fresh);
}
