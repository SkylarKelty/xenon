//! Minimal CUDA runtime FFI + safe Rust wrappers.
//!
//! We bind only what the inference engine needs (device selection, streams,
//! device allocations, H2D/D2H copies, query free/total memory). Adding a
//! heavier dep (cudarc, etc.) is deferred until we have a real reason to.

use std::ffi::{c_void, CStr};
use std::marker::PhantomData;

#[link(name = "cudart", kind = "dylib")]
unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMallocAsync(ptr: *mut *mut c_void, size: usize, stream: *mut c_void) -> i32;
    fn cudaFreeAsync(ptr: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetErrorString(err: i32) -> *const i8;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

    // Graph capture / instantiation / replay.
    fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiate(
        graph_exec: *mut *mut c_void,
        graph: *mut c_void,
        error_node: *mut *mut c_void,
        log_buf: *mut i8,
        log_size: usize,
    ) -> i32;
    fn cudaGraphLaunch(graph_exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(graph_exec: *mut c_void) -> i32;
}

/// `cudaStreamCaptureMode` — how strictly the capture validates stream-
/// crossing and memory ops during capture. `Relaxed` accepts anything that
/// would otherwise trip the validator (we've manually verified our decode
/// step is free of forbidden ops — see project_xenon_cuda_graphs_scope).
pub const CAPTURE_MODE_GLOBAL: i32 = 0;
pub const CAPTURE_MODE_THREAD_LOCAL: i32 = 1;
pub const CAPTURE_MODE_RELAXED: i32 = 2;

const MEMCPY_HOST_TO_DEVICE: i32 = 1;
const MEMCPY_DEVICE_TO_HOST: i32 = 2;
const MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

/// A CUDA runtime error code (non-zero). `0` is success and never wrapped.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub i32);

impl CudaError {
    fn check(code: i32) -> Result<(), CudaError> {
        if code == 0 {
            Ok(())
        } else {
            Err(CudaError(code))
        }
    }

    pub fn message(&self) -> String {
        unsafe {
            CStr::from_ptr(cudaGetErrorString(self.0))
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl std::fmt::Debug for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaError({}: {})", self.0, self.message())
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error {}: {}", self.0, self.message())
    }
}

impl std::error::Error for CudaError {}

/// A CUDA device handle. Just wraps an ordinal; cheap to copy.
#[derive(Clone, Copy, Debug)]
pub struct Device(pub i32);

impl Device {
    pub fn count() -> Result<i32, CudaError> {
        let mut n = 0;
        CudaError::check(unsafe { cudaGetDeviceCount(&mut n) })?;
        Ok(n)
    }

    pub fn current() -> Result<Device, CudaError> {
        let mut d = 0;
        CudaError::check(unsafe { cudaGetDevice(&mut d) })?;
        Ok(Device(d))
    }

    pub fn set(self) -> Result<(), CudaError> {
        CudaError::check(unsafe { cudaSetDevice(self.0) })
    }
}

/// `(free_bytes, total_bytes)` on the current device.
pub fn mem_info() -> Result<(usize, usize), CudaError> {
    let mut free = 0usize;
    let mut total = 0usize;
    CudaError::check(unsafe { cudaMemGetInfo(&mut free, &mut total) })?;
    Ok((free, total))
}

pub fn device_synchronize() -> Result<(), CudaError> {
    CudaError::check(unsafe { cudaDeviceSynchronize() })
}

/// CUDA stream handle. Dropped automatically.
pub struct Stream {
    raw: *mut c_void,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

/// `cudaStreamNonBlocking`: stream does NOT implicitly synchronize with the
/// legacy default stream. Required for CUDA-graph capture — a blocking
/// stream that depends on the default stream aborts capture with
/// `cudaErrorStreamCaptureIsolation` (906).
const STREAM_NON_BLOCKING: u32 = 0x1;

impl Stream {
    pub fn new() -> Result<Self, CudaError> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaStreamCreate(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Non-blocking stream — does NOT implicitly synchronize with the
    /// legacy default stream. Required for CUDA-graph capture (a blocking
    /// stream with a dependency on the default stream aborts capture with
    /// cudaErrorStreamCaptureIsolation = 906).
    pub fn new_nonblocking() -> Result<Self, CudaError> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaStreamCreateWithFlags(&mut raw, STREAM_NON_BLOCKING) })?;
        Ok(Self { raw })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        CudaError::check(unsafe { cudaStreamSynchronize(self.raw) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudaStreamDestroy(self.raw) };
        }
    }
}

// -------------------- CUDA Graphs --------------------
//
// Stream capture records the sequence of kernel launches + memcpy operations
// issued on a stream into a `Graph`, which can then be `instantiate()`d into
// a reusable `GraphExec`. Replaying a `GraphExec` on a stream re-runs the
// captured work as a single submission with per-step launch overhead
// amortized across one CUDA driver call.
//
// Xenon scope: graphs are used only by `xenon-server`'s decode loop, one
// `GraphExec` per batch size, cached for the engine's lifetime. Never
// captured from `xenon-cli` — see project_xenon_cuda_graphs_scope memory.

/// An immutable record of a captured stream's operations.
pub struct CudaGraph {
    raw: *mut c_void,
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Instantiate this graph into a runnable `GraphExec`. The graph remains
    /// valid afterwards and could in principle be instantiated again, but
    /// xenon only instantiates once per captured graph.
    pub fn instantiate(&self) -> Result<GraphExec, CudaError> {
        let mut exec: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe {
            cudaGraphInstantiate(
                &mut exec,
                self.raw,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        })?;
        Ok(GraphExec { raw: exec })
    }

    pub fn as_raw(&self) -> *mut c_void { self.raw }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudaGraphDestroy(self.raw) };
        }
    }
}

/// A runnable, instantiated graph. Replay with `launch(&stream)`.
pub struct GraphExec {
    raw: *mut c_void,
}

unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// Submit all recorded operations to `stream`. Non-blocking; follow
    /// with `stream.synchronize()` if you need to read results back.
    pub fn launch(&self, stream: &Stream) -> Result<(), CudaError> {
        CudaError::check(unsafe { cudaGraphLaunch(self.raw, stream.raw) })
    }

    pub fn as_raw(&self) -> *mut c_void { self.raw }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudaGraphExecDestroy(self.raw) };
        }
    }
}

impl Stream {
    /// Begin recording operations issued on this stream into a CUDA graph.
    /// The caller issues the work to capture, then calls `end_capture` to
    /// get the `CudaGraph`.
    ///
    /// `mode = CAPTURE_MODE_RELAXED` accepts cross-stream operations and
    /// other advanced patterns; xenon's single-stream decode step is fine
    /// with any mode but `Relaxed` is the most permissive.
    pub fn begin_capture(&self, mode: i32) -> Result<(), CudaError> {
        CudaError::check(unsafe { cudaStreamBeginCapture(self.raw, mode) })
    }

    /// Finish capture and return the recorded graph. Must be paired with
    /// a prior `begin_capture`; otherwise the CUDA driver errors.
    pub fn end_capture(&self) -> Result<CudaGraph, CudaError> {
        let mut graph: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaStreamEndCapture(self.raw, &mut graph) })?;
        Ok(CudaGraph { raw: graph })
    }
}

/// A typed device allocation. `T` must be POD (bytemuck::Pod) so we can safely
/// round-trip raw bytes from host slices.
///
/// Allocated via either `cudaMalloc` (default) or `cudaMallocAsync` on a
/// given stream — the latter hits CUDA's default stream-ordered memory
/// pool, which caches block sizes and is ~10-100x faster for the
/// alloc/free churn in transient scratch buffers. `free_stream` remembers
/// which path was used so Drop picks the matching API.
pub struct DeviceBuffer<T: bytemuck::Pod> {
    ptr: *mut c_void,
    len: usize,
    free_stream: *mut c_void, // null => cudaFree, non-null => cudaFreeAsync
    _p: PhantomData<T>,
}

unsafe impl<T: bytemuck::Pod + Send> Send for DeviceBuffer<T> {}
unsafe impl<T: bytemuck::Pod + Sync> Sync for DeviceBuffer<T> {}

impl<T: bytemuck::Pod> DeviceBuffer<T> {
    pub fn new(n: usize) -> Result<Self, CudaError> {
        if n == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                free_stream: std::ptr::null_mut(),
                _p: PhantomData,
            });
        }
        let bytes = n.checked_mul(std::mem::size_of::<T>()).expect("overflow");
        let mut ptr: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaMalloc(&mut ptr, bytes) })?;
        Ok(Self {
            ptr,
            len: n,
            free_stream: std::ptr::null_mut(),
            _p: PhantomData,
        })
    }

    /// Stream-ordered allocation from CUDA's default memory pool.
    /// For transient buffers reused each step, this is much faster than
    /// `cudaMalloc` because the pool caches freed blocks of the same size.
    pub fn new_async(n: usize, stream: &Stream) -> Result<Self, CudaError> {
        if n == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                free_stream: std::ptr::null_mut(),
                _p: PhantomData,
            });
        }
        let bytes = n.checked_mul(std::mem::size_of::<T>()).expect("overflow");
        let mut ptr: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaMallocAsync(&mut ptr, bytes, stream.as_raw()) })?;
        Ok(Self {
            ptr,
            len: n,
            free_stream: stream.as_raw(),
            _p: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<(), CudaError> {
        assert_eq!(src.len(), self.len, "DeviceBuffer::copy_from_host length mismatch");
        if self.len == 0 {
            return Ok(());
        }
        CudaError::check(unsafe {
            cudaMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                self.bytes(),
                MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    /// Upload raw bytes. Length must equal `self.bytes()`.
    pub fn copy_from_host_bytes(&mut self, src: &[u8]) -> Result<(), CudaError> {
        assert_eq!(src.len(), self.bytes(), "DeviceBuffer::copy_from_host_bytes byte length mismatch");
        if self.len == 0 {
            return Ok(());
        }
        CudaError::check(unsafe {
            cudaMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    /// Async upload raw bytes on a stream. With pageable host memory, CUDA
    /// copies the host bytes into a pinned staging buffer synchronously on
    /// the caller, then runs the staging→device DMA asynchronously on the
    /// stream. For overlap across streams, that's still useful: the DMA
    /// can run concurrently with kernels on other streams, and the host
    /// thread isn't blocked for the full transfer.
    pub fn copy_from_host_bytes_async(&mut self, src: &[u8], stream: &Stream) -> Result<(), CudaError> {
        assert_eq!(src.len(), self.bytes(), "DeviceBuffer::copy_from_host_bytes_async byte length mismatch");
        if self.len == 0 {
            return Ok(());
        }
        CudaError::check(unsafe {
            cudaMemcpyAsync(
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MEMCPY_HOST_TO_DEVICE,
                stream.as_raw(),
            )
        })
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), CudaError> {
        assert_eq!(dst.len(), self.len, "DeviceBuffer::copy_to_host length mismatch");
        if self.len == 0 {
            return Ok(());
        }
        CudaError::check(unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                self.bytes(),
                MEMCPY_DEVICE_TO_HOST,
            )
        })
    }

    /// Async D2H: enqueue `cudaMemcpyAsync` into `dst`. Caller must sync
    /// the stream before reading `dst`. Unlike `copy_to_host`, this is
    /// captureable by `cudaStreamBeginCapture` — the destination pointer
    /// is recorded in the graph node and reads happen at replay time, so
    /// the caller must keep `dst` alive at a stable address across
    /// replays.
    pub fn copy_to_host_async(&self, dst: &mut [T], stream: &Stream) -> Result<(), CudaError> {
        assert_eq!(dst.len(), self.len, "DeviceBuffer::copy_to_host_async length mismatch");
        if self.len == 0 {
            return Ok(());
        }
        CudaError::check(unsafe {
            cudaMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.bytes(),
                MEMCPY_DEVICE_TO_HOST,
                stream.as_raw(),
            )
        })
    }

    pub fn copy_to_host_vec(&self) -> Result<Vec<T>, CudaError> {
        let mut v = vec![T::zeroed(); self.len];
        self.copy_to_host(&mut v)?;
        Ok(v)
    }

    /// D2D copy from another buffer of identical length.
    pub fn copy_from_device(&mut self, src: &DeviceBuffer<T>) -> Result<(), CudaError> {
        assert_eq!(src.len, self.len, "DeviceBuffer::copy_from_device length mismatch");
        if self.len == 0 {
            return Ok(());
        }
        CudaError::check(unsafe {
            cudaMemcpy(self.ptr, src.ptr, self.bytes(), MEMCPY_DEVICE_TO_DEVICE)
        })
    }

    /// Async D2D copy via `cudaMemcpyAsync` on `stream`. Required for
    /// paths that run under CUDA graph capture: the synchronous
    /// `cudaMemcpy` used by `copy_from_device` runs on the legacy default
    /// stream and trips `cudaErrorStreamCaptureIsolation` (906) when the
    /// capturing stream tries to depend on it.
    pub fn copy_from_device_async(
        &mut self, src: &DeviceBuffer<T>, stream: &Stream,
    ) -> Result<(), CudaError> {
        assert_eq!(src.len, self.len, "copy_from_device_async: length mismatch");
        if self.len == 0 { return Ok(()); }
        CudaError::check(unsafe {
            cudaMemcpyAsync(
                self.ptr, src.ptr, self.bytes(),
                MEMCPY_DEVICE_TO_DEVICE, stream.as_raw(),
            )
        })
    }

    /// D2D copy of `src` into this buffer starting at `dst_offset` elements.
    /// `dst_offset + src.len()` must not exceed `self.len()`.
    pub fn copy_slice_from_device(
        &mut self,
        dst_offset: usize,
        src: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        self.copy_region_from_device(dst_offset, src, 0, src.len)
    }

    /// Async version of `copy_slice_from_device` — see `copy_from_device_async`
    /// for why the sync variant breaks graph capture.
    pub fn copy_slice_from_device_async(
        &mut self,
        dst_offset: usize,
        src: &DeviceBuffer<T>,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        self.copy_region_from_device_async(dst_offset, src, 0, src.len, stream)
    }

    /// General D2D region copy: copy `len` elements from `src[src_offset..]`
    /// into `self[dst_offset..]`.
    pub fn copy_region_from_device(
        &mut self,
        dst_offset: usize,
        src: &DeviceBuffer<T>,
        src_offset: usize,
        len: usize,
    ) -> Result<(), CudaError> {
        let dst_end = dst_offset.checked_add(len).expect("overflow");
        let src_end = src_offset.checked_add(len).expect("overflow");
        assert!(dst_end <= self.len, "copy_region: dst slice out of range");
        assert!(src_end <= src.len, "copy_region: src slice out of range");
        if len == 0 {
            return Ok(());
        }
        let esz = std::mem::size_of::<T>();
        let dst_ptr = unsafe { (self.ptr as *mut u8).add(dst_offset * esz) as *mut c_void };
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset * esz) as *const c_void };
        CudaError::check(unsafe {
            cudaMemcpy(dst_ptr, src_ptr, len * esz, MEMCPY_DEVICE_TO_DEVICE)
        })
    }

    /// Async version of `copy_region_from_device` — runs on the provided
    /// stream via `cudaMemcpyAsync`, capture-safe.
    pub fn copy_region_from_device_async(
        &mut self,
        dst_offset: usize,
        src: &DeviceBuffer<T>,
        src_offset: usize,
        len: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        let dst_end = dst_offset.checked_add(len).expect("overflow");
        let src_end = src_offset.checked_add(len).expect("overflow");
        assert!(dst_end <= self.len, "copy_region_async: dst slice out of range");
        assert!(src_end <= src.len, "copy_region_async: src slice out of range");
        if len == 0 { return Ok(()); }
        let esz = std::mem::size_of::<T>();
        let dst_ptr = unsafe { (self.ptr as *mut u8).add(dst_offset * esz) as *mut c_void };
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset * esz) as *const c_void };
        CudaError::check(unsafe {
            cudaMemcpyAsync(
                dst_ptr, src_ptr, len * esz,
                MEMCPY_DEVICE_TO_DEVICE, stream.as_raw(),
            )
        })
    }
}

impl<T: bytemuck::Pod> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if self.free_stream.is_null() {
                let _ = unsafe { cudaFree(self.ptr) };
            } else {
                let _ = unsafe { cudaFreeAsync(self.ptr, self.free_stream) };
            }
        }
    }
}

/// Page-locked (pinned) host buffer. Unlike a regular `Vec<T>`, these bytes
/// are pinned in physical memory and can be transferred to/from device
/// memory via `cudaMemcpyAsync` *truly* asynchronously — no CUDA-managed
/// staging buffer. One-time allocation cost trades for per-call H2D that
/// can run concurrently with compute on another stream.
pub struct PinnedBuffer<T: bytemuck::Pod> {
    ptr: *mut c_void,
    len: usize,
    _p: PhantomData<T>,
}

unsafe impl<T: bytemuck::Pod + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: bytemuck::Pod + Sync> Sync for PinnedBuffer<T> {}

impl<T: bytemuck::Pod> PinnedBuffer<T> {
    pub fn new(n: usize) -> Result<Self, CudaError> {
        if n == 0 {
            return Ok(Self { ptr: std::ptr::null_mut(), len: 0, _p: PhantomData });
        }
        let bytes = n.checked_mul(std::mem::size_of::<T>()).expect("overflow");
        let mut ptr: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaMallocHost(&mut ptr, bytes) })?;
        Ok(Self { ptr, len: n, _p: PhantomData })
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn bytes(&self) -> usize { self.len * std::mem::size_of::<T>() }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.bytes()) }
    }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.bytes()) }
    }
}

impl<T: bytemuck::Pod> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudaFreeHost(self.ptr) };
        }
    }
}
