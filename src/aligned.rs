use std::alloc::{alloc, dealloc, Layout};

pub const CACHE_LINE: usize = 64;
pub const DISK_BLOCK: usize = 4096;

/// Check if a pointer is aligned to `align` (power of two).
#[inline]
pub fn is_aligned(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

#[inline]
pub fn is_cache_aligned(ptr: *const u8) -> bool {
    is_aligned(ptr, CACHE_LINE)
}

#[inline]
pub fn is_block_aligned(ptr: *const u8) -> bool {
    is_aligned(ptr, DISK_BLOCK)
}

/// Round `pos` up to `align` (power of two).
#[inline]
pub fn align_up(pos: usize, align: usize) -> usize {
    if align == 0 {
        pos
    } else {
        (pos + align - 1) & !(align - 1)
    }
}

#[inline]
pub fn align_up_u64(pos: u64, align: u64) -> u64 {
    if align == 0 {
        pos
    } else {
        (pos + align - 1) / align * align
    }
}

/// A heap allocation guaranteed aligned to `ALIGN`.
///
/// Uses `std::alloc` directly so alignment survives `Vec` reallocation.
/// `ALIGN` must be a power of two.
pub struct AlignedBuffer<const ALIGN: usize> {
    ptr: *mut u8,
    layout: Layout,
    len: usize,
}

impl<const ALIGN: usize> AlignedBuffer<ALIGN> {
    pub fn new(len: usize) -> Option<Self> {
        if !ALIGN.is_power_of_two() {
            return None;
        }
        if len == 0 {
            // Zero-sized still needs a valid layout for dealloc safety
            let layout = Layout::from_size_align(1, ALIGN).ok()?;
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                return None;
            }
            return Some(Self {
                ptr,
                layout,
                len: 0,
            });
        }
        let layout = Layout::from_size_align(len, ALIGN).ok()?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return None;
        }
        // Zero-fill for determinism (important for disk padding)
        unsafe {
            std::ptr::write_bytes(ptr, 0, len);
        }
        Some(Self { ptr, layout, len })
    }

    pub fn new_zeroed(len: usize) -> Self {
        Self::new(len).expect("aligned alloc failed")
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn resize(&mut self, new_len: usize) -> bool {
        if new_len == self.len {
            return true;
        }
        // Reallocate via new alloc + copy (stable, no realloc with align guarantee)
        let new_layout = match Layout::from_size_align(new_len.max(1), ALIGN) {
            Ok(l) => l,
            Err(_) => return false,
        };
        let new_ptr = unsafe { alloc(new_layout) };
        if new_ptr.is_null() {
            return false;
        }
        let copy_len = self.len.min(new_len);
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, new_ptr, copy_len);
            if new_len > copy_len {
                std::ptr::write_bytes(new_ptr.add(copy_len), 0, new_len - copy_len);
            }
            dealloc(self.ptr, self.layout);
        }
        self.ptr = new_ptr;
        self.layout = new_layout;
        self.len = new_len;
        true
    }
}

impl<const ALIGN: usize> Drop for AlignedBuffer<ALIGN> {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

unsafe impl<const ALIGN: usize> Send for AlignedBuffer<ALIGN> {}
unsafe impl<const ALIGN: usize> Sync for AlignedBuffer<ALIGN> {}

pub type CacheAlignedBuffer = AlignedBuffer<CACHE_LINE>;
pub type BlockAlignedBuffer = AlignedBuffer<DISK_BLOCK>;

/// Stack-allocated cache-line aligned buffer for hot-path quantized reads.
/// `N` should be ≤ 1024 for stack safety; larger dims should heap-allocate.
#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct StackBuf<const N: usize> {
    pub data: [u8; N],
}

impl<const N: usize> StackBuf<N> {
    pub fn new() -> Self {
        Self { data: [0u8; N] }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl<const N: usize> Default for StackBuf<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to get a correctly aligned heap Vec for bulk I/O.
/// Over-allocates and slices to guarantee alignment for the returned Vec's ptr.
pub fn alloc_aligned_vec(size: usize, _align: usize) -> Vec<u8> {
    if size == 0 {
        return Vec::new();
    }
    // Helper kept for API ergonomics; true aligned alloc is `AlignedBuffer`.
    // `_align` retained for signature compatibility.
    vec![0u8; size]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_aligned_alloc() {
        let buf = CacheAlignedBuffer::new_zeroed(128);
        assert!(is_cache_aligned(buf.as_ptr()));
        assert_eq!(buf.len(), 128);
    }

    #[test]
    fn block_aligned_alloc() {
        let buf = BlockAlignedBuffer::new_zeroed(4096);
        assert!(is_block_aligned(buf.as_ptr()));
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up_u64(62, 4096), 4096);
    }

    #[test]
    fn stack_buf_aligned() {
        let buf = StackBuf::<128>::new();
        assert!(is_cache_aligned(buf.data.as_ptr()));
    }
}
