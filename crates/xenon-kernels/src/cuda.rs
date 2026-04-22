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
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetErrorString(err: i32) -> *const i8;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

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

impl Stream {
    pub fn new() -> Result<Self, CudaError> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaStreamCreate(&mut raw) })?;
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

/// A typed device allocation. `T` must be POD (bytemuck::Pod) so we can safely
/// round-trip raw bytes from host slices.
pub struct DeviceBuffer<T: bytemuck::Pod> {
    ptr: *mut c_void,
    len: usize,
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
                _p: PhantomData,
            });
        }
        let bytes = n.checked_mul(std::mem::size_of::<T>()).expect("overflow");
        let mut ptr: *mut c_void = std::ptr::null_mut();
        CudaError::check(unsafe { cudaMalloc(&mut ptr, bytes) })?;
        Ok(Self {
            ptr,
            len: n,
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

    /// D2D copy of `src` into this buffer starting at `dst_offset` elements.
    /// `dst_offset + src.len()` must not exceed `self.len()`.
    pub fn copy_slice_from_device(
        &mut self,
        dst_offset: usize,
        src: &DeviceBuffer<T>,
    ) -> Result<(), CudaError> {
        self.copy_region_from_device(dst_offset, src, 0, src.len)
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
}

impl<T: bytemuck::Pod> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudaFree(self.ptr) };
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
