// https://gist.github.com/GabrielMajeri/545042ee4f956d5b2141105eb6a505a9
// Potentially improves the performance of SIMD floating-point math
// by flushing denormals/underflow to zero.
pub unsafe fn flush_denormals() {
    use std::arch::x86_64::*;

    let mut mxcsr = _mm_getcsr();

    // Denormals & underflows are flushed to zero
    mxcsr |= (1 << 15) | (1 << 6);

    // All exceptions are masked
    mxcsr |= ((1 << 6) - 1) << 7;

    _mm_setcsr(mxcsr);
}

fn main() {}
