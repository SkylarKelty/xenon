//! Minimal cuBLASLt FFI + a BF16 matmul wrapper.
//!
//! cuBLASLt works in column-major. Our higher layers think in row-major. We
//! convert by swapping operands: row-major `D[M,N] = A[M,K] * B[K,N]` becomes
//! col-major `D^T[N,M] = B^T[N,K] * A^T[K,M]`, which cuBLASLt computes
//! directly on the same buffers (since reinterpreting a row-major matrix in
//! col-major *is* its transpose).
//!
//! We bind only the subset we need: create/destroy the handle, describe an op
//! + layouts, ask the heuristic for an algo, run it. Scoped to BF16 operands
//! + fp32 accumulate; other dtypes can be added when we need them.

use std::ffi::c_void;

use half::bf16;

use crate::cuda::{CudaError, DeviceBuffer, Stream};

// Opaque handle types.
type CublasLtHandleRaw = *mut c_void;
type MatmulDescRaw     = *mut c_void;
type MatrixLayoutRaw   = *mut c_void;
type MatmulPreferenceRaw = *mut c_void;

/// `cublasLtMatmulAlgo_t` is an opaque 8*8 = 64-byte blob. We never read its
/// fields; we just hand it back to `cublasLtMatmul`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulAlgo {
    data: [u64; 8],
}

#[repr(C)]
struct MatmulHeuristicResult {
    algo: MatmulAlgo,
    workspace_size: usize,
    state: i32,
    waves_count: f32,
    _reserved: [i32; 4],
}

// cudaDataType enum values (from library_types.h).
const CUDA_R_16BF: i32 = 14;
const CUDA_R_32F:  i32 = 0;

// cublasComputeType_t values (from cublas_api.h).
const CUBLAS_COMPUTE_32F: i32 = 68;

// cublasOperation_t values.
const CUBLAS_OP_N: i32 = 0;
#[allow(dead_code)] // wanted once we do transposed / FP4 operand GEMMs.
const CUBLAS_OP_T: i32 = 1;

// cublasLtMatmulDescAttributes_t.
const CUBLASLT_MATMUL_DESC_TRANSA: u32 = 3;
const CUBLASLT_MATMUL_DESC_TRANSB: u32 = 4;

// cublasLtMatmulPreferenceAttributes_t.
const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: u32 = 1;

#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    fn cublasLtCreate(handle: *mut CublasLtHandleRaw) -> i32;
    fn cublasLtDestroy(handle: CublasLtHandleRaw) -> i32;

    fn cublasLtMatmulDescCreate(
        desc: *mut MatmulDescRaw,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: MatmulDescRaw) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: MatmulDescRaw,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;

    fn cublasLtMatrixLayoutCreate(
        layout: *mut MatrixLayoutRaw,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: MatrixLayoutRaw) -> i32;

    fn cublasLtMatmulPreferenceCreate(pref: *mut MatmulPreferenceRaw) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: MatmulPreferenceRaw) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: MatmulPreferenceRaw,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;

    fn cublasLtMatmulAlgoGetHeuristic(
        handle: CublasLtHandleRaw,
        desc: MatmulDescRaw,
        a_layout: MatrixLayoutRaw,
        b_layout: MatrixLayoutRaw,
        c_layout: MatrixLayoutRaw,
        d_layout: MatrixLayoutRaw,
        pref: MatmulPreferenceRaw,
        requested: i32,
        results: *mut MatmulHeuristicResult,
        returned: *mut i32,
    ) -> i32;

    fn cublasLtMatmul(
        handle: CublasLtHandleRaw,
        desc: MatmulDescRaw,
        alpha: *const c_void,
        a: *const c_void,
        a_layout: MatrixLayoutRaw,
        b: *const c_void,
        b_layout: MatrixLayoutRaw,
        beta: *const c_void,
        c: *const c_void,
        c_layout: MatrixLayoutRaw,
        d: *mut c_void,
        d_layout: MatrixLayoutRaw,
        algo: *const MatmulAlgo,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// cuBLASLt status code (non-zero). `0` is success.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CublasError(pub i32);

impl std::fmt::Debug for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cublas error (status {})", self.0)
    }
}

impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cublas error (status {})", self.0)
    }
}

impl std::error::Error for CublasError {}

fn ck(s: i32) -> Result<(), CublasError> {
    if s == 0 {
        Ok(())
    } else {
        Err(CublasError(s))
    }
}

/// Error returned by the GEMM wrappers: either cuBLASLt or underlying CUDA.
#[derive(Debug)]
pub enum GemmError {
    Cublas(CublasError),
    Cuda(CudaError),
    NoAlgorithm,
}

impl std::fmt::Display for GemmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GemmError::Cublas(e) => write!(f, "{e}"),
            GemmError::Cuda(e) => write!(f, "{e}"),
            GemmError::NoAlgorithm => write!(f, "no cuBLASLt algorithm found for the requested shapes"),
        }
    }
}

impl std::error::Error for GemmError {}

impl From<CublasError> for GemmError {
    fn from(e: CublasError) -> Self {
        GemmError::Cublas(e)
    }
}

impl From<CudaError> for GemmError {
    fn from(e: CudaError) -> Self {
        GemmError::Cuda(e)
    }
}

/// Owning handle to cuBLASLt with a pre-allocated workspace.
pub struct CublasLt {
    handle: CublasLtHandleRaw,
    workspace: DeviceBuffer<u8>,
}

unsafe impl Send for CublasLt {}

impl CublasLt {
    /// Create a handle and reserve a 32 MiB workspace on the current device.
    pub fn new() -> Result<Self, GemmError> {
        let mut h: CublasLtHandleRaw = std::ptr::null_mut();
        ck(unsafe { cublasLtCreate(&mut h) })?;
        let workspace = DeviceBuffer::<u8>::new(32 * 1024 * 1024)?;
        Ok(Self { handle: h, workspace })
    }

    /// `D[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]`, all row-major bf16.
    /// Uses fp32 accumulation and fp32 alpha/beta.
    pub fn matmul_bf16_rm(
        &mut self,
        d: &mut DeviceBuffer<bf16>,
        a: &DeviceBuffer<bf16>,
        b: &DeviceBuffer<bf16>,
        c: Option<&DeviceBuffer<bf16>>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        stream: Option<&Stream>,
    ) -> Result<(), GemmError> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(d.len(), m * n);
        if let Some(c) = c {
            assert_eq!(c.len(), m * n);
        }

        // Row-major to col-major: compute D^T[N,M] = B^T[N,K] * A^T[K,M].
        // cuBLASLt sees the buffers unchanged; only the layout metadata swaps.
        //
        //   "A_cublas" = our B, viewed col-major as [N, K], ld=N
        //   "B_cublas" = our A, viewed col-major as [K, M], ld=K
        //   "D_cublas" = our D, viewed col-major as [N, M], ld=N
        //
        // No transpose ops needed; we're just reinterpreting the layout.

        let mut desc: MatmulDescRaw = std::ptr::null_mut();
        ck(unsafe { cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) })?;
        let guard_desc = DescGuard(desc);

        let op_n = CUBLAS_OP_N;
        ck(unsafe {
            cublasLtMatmulDescSetAttribute(
                desc, CUBLASLT_MATMUL_DESC_TRANSA,
                &op_n as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            )
        })?;
        ck(unsafe {
            cublasLtMatmulDescSetAttribute(
                desc, CUBLASLT_MATMUL_DESC_TRANSB,
                &op_n as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            )
        })?;

        let mut a_layout: MatrixLayoutRaw = std::ptr::null_mut();
        let mut b_layout: MatrixLayoutRaw = std::ptr::null_mut();
        let mut d_layout: MatrixLayoutRaw = std::ptr::null_mut();
        ck(unsafe { cublasLtMatrixLayoutCreate(&mut a_layout, CUDA_R_16BF, n as u64, k as u64, n as i64) })?;
        let _g_a = LayoutGuard(a_layout);
        ck(unsafe { cublasLtMatrixLayoutCreate(&mut b_layout, CUDA_R_16BF, k as u64, m as u64, k as i64) })?;
        let _g_b = LayoutGuard(b_layout);
        ck(unsafe { cublasLtMatrixLayoutCreate(&mut d_layout, CUDA_R_16BF, n as u64, m as u64, n as i64) })?;
        let _g_d = LayoutGuard(d_layout);

        let mut pref: MatmulPreferenceRaw = std::ptr::null_mut();
        ck(unsafe { cublasLtMatmulPreferenceCreate(&mut pref) })?;
        let _g_pref = PrefGuard(pref);
        let ws_bytes: u64 = self.workspace.bytes() as u64;
        ck(unsafe {
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const u64 as *const c_void,
                std::mem::size_of::<u64>(),
            )
        })?;

        let mut heur = MatmulHeuristicResult {
            algo: MatmulAlgo { data: [0; 8] },
            workspace_size: 0,
            state: 0,
            waves_count: 0.0,
            _reserved: [0; 4],
        };
        let mut returned: i32 = 0;
        ck(unsafe {
            cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc,
                a_layout, b_layout, d_layout, d_layout,
                pref,
                1,
                &mut heur,
                &mut returned,
            )
        })?;
        if returned == 0 {
            return Err(GemmError::NoAlgorithm);
        }

        let c_ptr: *const c_void = match c {
            Some(c) => c.as_device_ptr() as *const c_void,
            None => d.as_device_ptr() as *const c_void,
        };
        let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());

        ck(unsafe {
            cublasLtMatmul(
                self.handle,
                desc,
                &alpha as *const f32 as *const c_void,
                b.as_device_ptr() as *const c_void, a_layout,
                a.as_device_ptr() as *const c_void, b_layout,
                &beta as *const f32 as *const c_void,
                c_ptr, d_layout,
                d.as_device_ptr(), d_layout,
                &heur.algo as *const MatmulAlgo,
                self.workspace.as_device_ptr(),
                self.workspace.bytes(),
                stream_ptr,
            )
        })?;

        drop(guard_desc);
        Ok(())
    }
}

impl Drop for CublasLt {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { cublasLtDestroy(self.handle) };
        }
    }
}

struct DescGuard(MatmulDescRaw);
impl Drop for DescGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let _ = unsafe { cublasLtMatmulDescDestroy(self.0) };
        }
    }
}

struct LayoutGuard(MatrixLayoutRaw);
impl Drop for LayoutGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let _ = unsafe { cublasLtMatrixLayoutDestroy(self.0) };
        }
    }
}

struct PrefGuard(MatmulPreferenceRaw);
impl Drop for PrefGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let _ = unsafe { cublasLtMatmulPreferenceDestroy(self.0) };
        }
    }
}

/// Reference host-side bf16 matmul (fp32 accumulate) for correctness tests.
pub fn matmul_bf16_reference(
    a: &[bf16],
    b: &[bf16],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    c: Option<&[bf16]>,
) -> Vec<bf16> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut d = vec![bf16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l].to_f32() * b[l * n + j].to_f32();
            }
            let cv = c.map(|c| c[i * n + j].to_f32()).unwrap_or(0.0);
            d[i * n + j] = bf16::from_f32(alpha * acc + beta * cv);
        }
    }
    d
}
