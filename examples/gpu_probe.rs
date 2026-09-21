fn main() {
    match cudarc::driver::CudaContext::new(0) {
        Ok(_) => println!("GPU OK"),
        Err(e) => println!("GPU FAIL: {e:?}"),
    }
}
