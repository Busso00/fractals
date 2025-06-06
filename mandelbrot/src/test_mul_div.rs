
use eframe::glow::HasContext;
use eframe::{egui, glow, App};
use f128::f128;
use std::sync::Arc;
use std::mem;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct F128Repr {
    bits: [u32; 4],
}

pub const fn swap_endianness(val: u32) -> u32 {
    ((val & 0x000000FF) << 24) |
    ((val & 0x0000FF00) << 8)  |
    ((val & 0x00FF0000) >> 8)  |
    ((val & 0xFF000000) >> 24)
}

fn to_f128(u32s: [u32; 4]) -> f128 {
    // Convert each u32 to LE bytes and concatenate
    let mut bytes = [0u8; 16];
    for i in 0..4 {
        let word = u32s[i].to_le_bytes();
        bytes[i*4..i*4+4].copy_from_slice(&word);
    }
    unsafe { std::mem::transmute::<[u8; 16], f128>(bytes) }
}

// Helper function to split a u64 into two u32s
const fn u64_to_u32_pair(val: u64) -> (u32, u32) {
    let high = (val >> 32) as u32;
    let low = val as u32;
    (high, low)
}

// A macro to encapsulate the conversion from (high_u64, low_u64) to [u32; 4]
// This assumes the provided high_u64 and low_u64 are the correct IEEE 754-2008 binary128 bit patterns.
macro_rules! f128_bits_to_uvec4 {
    ($high:expr, $low:expr) => {{
        let (h_high, h_low) = u64_to_u32_pair($high);
        let (l_high, l_low) = u64_to_u32_pair($low);
      
        [l_low, l_high, h_low, h_high]
    }};
}

//stored byte by byte in opposite direction (block of 8bits) in according to little endian
const INPUT_VALUES: [[u32; 4]; 648] = [
    f128_bits_to_uvec4!(0x3F81000000000000, 0x0000000000000000), // 2^-126 (present)
    f128_bits_to_uvec4!(0x40FEFFFFFFFFFFFF, 0xFE00000000000000), // Large number near max (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000001), // Smallest positive denormal (2^-16382 - 2^-16494) 2^-16494 (incorrect based on typical fp128, assuming it's 2^-163) (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000000), // -0.0 (present)
    f128_bits_to_uvec4!(0x7FFF000000000000, 0x0000000000000000), // +inf (present)
    f128_bits_to_uvec4!(0xFFFF000000000000, 0x0000000000000000), // -inf (present)
    f128_bits_to_uvec4!(0x7FFF800000000000, 0x0000000000000000), // NaN (quiet) (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x0000020000000000), // 1 + 2^-23 (for double, close to 1 for quad) (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x0000000000000000), // 1.0 (present)
    f128_bits_to_uvec4!(0x3FD7FFFFFFFFFFFF, 0xFFFFFFFFFFFFC000), // 2^-24 (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x0000008000000000), // 1 + 2^-20 (present)
    f128_bits_to_uvec4!(0x40001921FB54442D, 0x1800000000000000), // pi (present)
    f128_bits_to_uvec4!(0x400015BF0A8B1457, 0x6953555555555555), // sqrt(2) (present)
    f128_bits_to_uvec4!(0x40001EE6C22EB52A, 0x1D53555555555555), // e (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000002), // 2^-148 (denormal) (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000003), // 3 * 2^-149 (denormal) (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000004), // 4 * 2^-149 (denormal) (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000005), // 5 * 2^-149 (denormal) (present)
    f128_bits_to_uvec4!(0x3F80FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Just below 2^-126 (subnormal threshold) (present)
    f128_bits_to_uvec4!(0x3F82000000000000, 0x0000000000000000), // 2^-125 (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000000), // +0.0 (present)
    f128_bits_to_uvec4!(0x4000000000000000, 0x0000000000000000), // 2.0 (present)
    f128_bits_to_uvec4!(0x3FFFE00000000000, 0x0000000000000000), // Just below 1.0 (present)
    f128_bits_to_uvec4!(0x4000000000000000, 0x0000000000000001), // 2.0 + smallest increment (present)
    f128_bits_to_uvec4!(0x4001000000000000, 0x0000000000000000), // 4.0 (present)
    f128_bits_to_uvec4!(0x3FFD000000000000, 0x0000000000000000), // 0.5 (present)
    f128_bits_to_uvec4!(0x7FEEFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Largest finite positive number (present)
    f128_bits_to_uvec4!(0x0001000000000000, 0x0000000000000000), // Smallest normal positive number (2^-16382) (present)
    f128_bits_to_uvec4!(0x7FFF000000000000, 0x0000000000000001), // Signalling NaN (present)
    f128_bits_to_uvec4!(0x3FC0000000000000, 0x0000000000000000), // 2^-1 (present)
    f128_bits_to_uvec4!(0x3FB0000000000000, 0x0000000000000000), // 2^-2 (present)
    f128_bits_to_uvec4!(0x3FA0000000000000, 0x0000000000000000), // 2^-3 (present)
    f128_bits_to_uvec4!(0x3F90000000000000, 0x0000000000000000), // 2^-4 (present)
    f128_bits_to_uvec4!(0x3F80000000000000, 0x0000000000000000), // 2^-5 (present)
    f128_bits_to_uvec4!(0x3F70000000000000, 0x0000000000000000), // 2^-6 (present)
    f128_bits_to_uvec4!(0x3F60000000000000, 0x0000000000000000), // 2^-7 (present)
    f128_bits_to_uvec4!(0x3F50000000000000, 0x0000000000000000), // 2^-8 (present)
    f128_bits_to_uvec4!(0x3FFF800000000000, 0x0000000000000000), // 1.5 (present)
    f128_bits_to_uvec4!(0x3FE0000000000000, 0x0000000000000000), // 0.25 (present)
    f128_bits_to_uvec4!(0x4000800000000000, 0x0000000000000000), // 2.5 (present)
    f128_bits_to_uvec4!(0x4000100000000000, 0x0000000000000000), // 2.125 (present)
    f128_bits_to_uvec4!(0x3FFFC00000000000, 0x0000000000000000), // 0.875 (present)
    f128_bits_to_uvec4!(0x4000400000000000, 0x0000000000000000), // 2.25 (present)
    f128_bits_to_uvec4!(0x3FFF400000000000, 0x0000000000000000), // 0.75 (present)
    f128_bits_to_uvec4!(0x4000C00000000000, 0x0000000000000000), // 2.75 (present)
    f128_bits_to_uvec4!(0xBF81000000000000, 0x0000000000000000), // -2^-126 (present)
    f128_bits_to_uvec4!(0xC0FEFFFFFFFFFFFF, 0xFE00000000000000), // Large number near min (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000001), // Smallest negative denormal (-2^-163) (present)
    f128_bits_to_uvec4!(0xFFFF800000000000, 0x0000000000000000), // -NaN (present)
    f128_bits_to_uvec4!(0xBFFF000000000000, 0x0000020000000000), // -(1 + 2^-23) (present)
    f128_bits_to_uvec4!(0xBFFF000000000000, 0x0000000000000000), // -1.0 (present)
    f128_bits_to_uvec4!(0xBFD7FFFFFFFFFFFF, 0xFFFFFFFFFFFFC000), // -2^-24 (present)
    f128_bits_to_uvec4!(0xBFFF000000000000, 0x0000008000000000), // -(1 + 2^-20) (present)
    f128_bits_to_uvec4!(0xC0001921FB54442D, 0x1800000000000000), // -pi (present)
    f128_bits_to_uvec4!(0xC00015BF0A8B1457, 0x6953555555555555), // -sqrt(2) (present)
    f128_bits_to_uvec4!(0xC0001EE6C22EB52A, 0x1D53555555555555), // -e (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000002), // -2^-148 (denormal) (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000003), // -3 * 2^-149 (denormal) (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000004), // -4 * 2^-149 (denormal) (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000005), // -5 * 2^-149 (denormal) (present)
    f128_bits_to_uvec4!(0xBF80FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // -Just below 2^-126 (subnormal threshold) (present)
    f128_bits_to_uvec4!(0xBF82000000000000, 0x0000000000000000), // -2^-125 (present)
    f128_bits_to_uvec4!(0xC000000000000000, 0x0000000000000000), // -2.0 (present)
    f128_bits_to_uvec4!(0xBFFFE00000000000, 0x0000000000000000), // -Just below -1.0 (present)
    f128_bits_to_uvec4!(0xC000000000000000, 0x0000000000000001), // -2.0 + smallest increment (present)
    f128_bits_to_uvec4!(0xC001000000000000, 0x0000000000000000), // -4.0 (present)
    f128_bits_to_uvec4!(0xBFFD000000000000, 0x0000000000000000), // -0.5 (present)
    f128_bits_to_uvec4!(0xFFEEFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Largest finite negative number (present)
    f128_bits_to_uvec4!(0x8001000000000000, 0x0000000000000000), // Smallest normal negative number (-2^-16382) (present)
    f128_bits_to_uvec4!(0xFFFF000000000001, 0x0000000000000000), // Negative Signalling NaN (present)
    f128_bits_to_uvec4!(0xBFC0000000000000, 0x0000000000000000), // -2^-1 (present)
    f128_bits_to_uvec4!(0xBFB0000000000000, 0x0000000000000000), // -2^-2 (present)
    f128_bits_to_uvec4!(0xBFA0000000000000, 0x0000000000000000), // -2^-3 (present)
    f128_bits_to_uvec4!(0xBF90000000000000, 0x0000000000000000), // -2^-4 (present)
    f128_bits_to_uvec4!(0xBF80000000000000, 0x0000000000000000), // -2^-5 (present)
    f128_bits_to_uvec4!(0xBF70000000000000, 0x0000000000000000), // -2^-6 (present)
    f128_bits_to_uvec4!(0xBF60000000000000, 0x0000000000000000), // -2^-7 (present)
    f128_bits_to_uvec4!(0xBF50000000000000, 0x0000000000000000), // -2^-8 (present)
    f128_bits_to_uvec4!(0xBFFF800000000000, 0x0000000000000000), // -1.5 (present)
    f128_bits_to_uvec4!(0xBFE0000000000000, 0x0000000000000000), // -0.25 (present)
    f128_bits_to_uvec4!(0xC000800000000000, 0x0000000000000000), // -2.5 (present)
    f128_bits_to_uvec4!(0xC000100000000000, 0x0000000000000000), // -2.125 (present)
    f128_bits_to_uvec4!(0xBFFFC00000000000, 0x0000000000000000), // -0.875 (present)
    f128_bits_to_uvec4!(0xC000400000000000, 0x0000000000000000), // -2.25 (present)
    f128_bits_to_uvec4!(0xBFFF400000000000, 0x0000000000000000), // -0.75 (present)
    f128_bits_to_uvec4!(0xC000C00000000000, 0x0000000000000000), // -2.75 (present)
    f128_bits_to_uvec4!(0x4002000000000000, 0x0000000000000000), // 8.0 (present)
    f128_bits_to_uvec4!(0x4003000000000000, 0x0000000000000000), // 16.0 (present)
    f128_bits_to_uvec4!(0x4001800000000000, 0x0000000000000000), // 6.0 (present)
    f128_bits_to_uvec4!(0x3FEF000000000000, 0x0000000000000000), // 0.125 (present)
    f128_bits_to_uvec4!(0x4002800000000000, 0x0000000000000000), // 12.0 (present)
    f128_bits_to_uvec4!(0x7FFF800000000000, 0x0000000000000001), // Signalling NaN (present)
    f128_bits_to_uvec4!(0xFFFF800000000000, 0x0000000000000001), // Negative Signalling NaN (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000006), // 6 * 2^-149 (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000007), // 7 * 2^-149 (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000008), // 8 * 2^-149 (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000009), // 9 * 2^-149 (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000000A), // 10 * 2^-149 (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000006), // -6 * 2^-149 (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000007), // -7 * 2^-149 (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000008), // -8 * 2^-149 (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x0000000000000009), // -9 * 2^-149 (present)
    f128_bits_to_uvec4!(0x8000000000000000, 0x000000000000000A), // -10 * 2^-149 (present)
    f128_bits_to_uvec4!(0x7FEF000000000000, 0x0000000000000000), // Just below max (present)
    f128_bits_to_uvec4!(0x7FEDFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Even closer to max (present)
    f128_bits_to_uvec4!(0xFFEF000000000000, 0x0000000000000000), // Just below min (present)
    f128_bits_to_uvec4!(0xFFEDFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Even closer to min (present)
    f128_bits_to_uvec4!(0xFFEEDDCCBBAA9988, 0x7766554433221100), // (present)
    f128_bits_to_uvec4!(0xFFDDBB9977553310, 0xEECCAA8866442200), // (present)
    f128_bits_to_uvec4!(0xFFBB7732EEAA6620, 0xDD995510CC884400), // (present)
    f128_bits_to_uvec4!(0xFF76EE65DD54CC40, 0xBB32AA2199108800), // (present)
    f128_bits_to_uvec4!(0xFEEDDCCBBAA99880, 0x7665544332211000), // (present)
    f128_bits_to_uvec4!(0xFDDBB99775533100, 0xECCAA88664422000), // (present)
    f128_bits_to_uvec4!(0xFBB7732EEAA66200, 0xD995510CC8844000), // (present)
    f128_bits_to_uvec4!(0xF76EE65DD54CC400, 0xB32AA21991088000), // (present)
    f128_bits_to_uvec4!(0xEEDDCCBBAA998800, 0x6655443322110000), // (present)
    f128_bits_to_uvec4!(0xDDBB997755331000, 0xCCAA886644220000), // (present)
    f128_bits_to_uvec4!(0xBB7732EEAA662000, 0x995510CC88440000), // (present)
    f128_bits_to_uvec4!(0x76EE65DD54CC4000, 0x32AA219910880000), // (present)
    f128_bits_to_uvec4!(0xEDDCCBBAA9988000, 0x6554433221100000), // (present)
    f128_bits_to_uvec4!(0xDBB9977553310000, 0xCAA8866442200000), // (present)
    f128_bits_to_uvec4!(0xB7732EEAA6620000, 0x95510CC884400000), // (present)
    f128_bits_to_uvec4!(0x6EE65DD54CC40000, 0x2AA2199108800000), // (present)
    f128_bits_to_uvec4!(0xDDCCBBAA99880000, 0x5544332211000000), // (present)
    f128_bits_to_uvec4!(0xBB99775533100000, 0xAA88664422000000), // (present)
    f128_bits_to_uvec4!(0x7732EEAA66200000, 0x5510CC8844000000), // (present)
    f128_bits_to_uvec4!(0xEE65DD54CC400000, 0xAA21991088000000), // (present)
    f128_bits_to_uvec4!(0xDCCBBAA998800000, 0x5443322110000000), // (present)
    f128_bits_to_uvec4!(0xB997755331000000, 0xA886644220000000), // (present)
    f128_bits_to_uvec4!(0x732EEAA662000000, 0x510CC88440000000), // (present)
    f128_bits_to_uvec4!(0xE65DD54CC4000000, 0xA219910880000000), // (present)
    f128_bits_to_uvec4!(0xCCBBAA9988000000, 0x4433221100000000), // (present)
    f128_bits_to_uvec4!(0x9977553310000000, 0x8866442200000000), // (present)
    f128_bits_to_uvec4!(0x32EEAA6620000000, 0x10CC884400000000), // (present)
    f128_bits_to_uvec4!(0x65DD54CC40000000, 0x2199108800000000), // (present)
    f128_bits_to_uvec4!(0xCBBAA99880000000, 0x4332211000000000), // (present)
    f128_bits_to_uvec4!(0x9775533100000000, 0x8664422000000000), // (present)
    f128_bits_to_uvec4!(0x2EEAA66200000000, 0x0CC8844000000000), // (present)
    f128_bits_to_uvec4!(0x5DD54CC400000000, 0x1991088000000000), // (present)
    f128_bits_to_uvec4!(0xBBAA998800000000, 0x3322110000000000), // (present)
    f128_bits_to_uvec4!(0x7755331000000000, 0x6644220000000000), // (present)
    f128_bits_to_uvec4!(0xEEAA662000000000, 0xCC88440000000000), // (present)
    f128_bits_to_uvec4!(0xDD54CC4000000000, 0x9910880000000000), // (present)
    f128_bits_to_uvec4!(0xBAA9988000000000, 0x3221100000000000), // (present)
    f128_bits_to_uvec4!(0x7553310000000000, 0x6442200000000000), // (present)
    f128_bits_to_uvec4!(0xEAA6620000000000, 0xC884400000000000), // (present)
    f128_bits_to_uvec4!(0xD54CC40000000000, 0x9108800000000000), // (present)
    f128_bits_to_uvec4!(0xAA99880000000000, 0x2211000000000000), // (present)
    f128_bits_to_uvec4!(0x5533100000000000, 0x4422000000000000), // (present)
    f128_bits_to_uvec4!(0xAA66200000000000, 0x8844000000000000), // (present)
    f128_bits_to_uvec4!(0x54CC400000000000, 0x1088000000000000), // (present)
    f128_bits_to_uvec4!(0xA998800000000000, 0x2110000000000000), // (present)
    f128_bits_to_uvec4!(0x5331000000000000, 0x4220000000000000), // (present)
    f128_bits_to_uvec4!(0xA662000000000000, 0x8440000000000000), // (present)
    f128_bits_to_uvec4!(0x4CC4000000000000, 0x0880000000000000), // (present)
    f128_bits_to_uvec4!(0x9988000000000000, 0x1100000000000000), // (present)
    f128_bits_to_uvec4!(0x3310000000000000, 0x2200000000000000), // (present)
    f128_bits_to_uvec4!(0x6620000000000000, 0x4400000000000000), // (present)
    f128_bits_to_uvec4!(0xCC40000000000000, 0x8800000000000000), // (present)
    f128_bits_to_uvec4!(0x9880000000000000, 0x1000000000000000), // (present)
    f128_bits_to_uvec4!(0x3100000000000000, 0x2000000000000000), // (present)
    f128_bits_to_uvec4!(0x6200000000000000, 0x4000000000000000), // (present)
    f128_bits_to_uvec4!(0xC400000000000000, 0x8000000000000000), // (present)
    f128_bits_to_uvec4!(0x8800000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x1000000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x2000000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x7FF76EE65DD54CC4, 0x3BB32AA219910880), // (present)
    f128_bits_to_uvec4!(0x3FFBB7732EEAA662, 0x1DD995510CC88440), // (present)
    f128_bits_to_uvec4!(0x1FFDDBB997755331, 0x0EECCAA886644220), // (present)
    f128_bits_to_uvec4!(0x0FFEEDDCCBBAA998, 0x0776655443322110), // (present)
    f128_bits_to_uvec4!(0x07FF76EE65DD54CC, 0x03BB32AA21991088), // (present)
    f128_bits_to_uvec4!(0x03FFBB7732EEAA66, 0x01DD995510CC8844), // (present)
    f128_bits_to_uvec4!(0x01FFDDBB99775533, 0x00EECCAA88664422), // (present)
    f128_bits_to_uvec4!(0x00FFEEDDCCBBAA99, 0x0077665544332211), // (present)
    f128_bits_to_uvec4!(0x007FF76EE65DD54C, 0x003BB32AA2199108), // (present)
    f128_bits_to_uvec4!(0x003FFBB7732EEAA6, 0x001DD995510CC884), // (present)
    f128_bits_to_uvec4!(0x001FFDDBB9977553, 0x000EECCAA8866442), // (present)
    f128_bits_to_uvec4!(0x000FFEEDDCCBBAA9, 0x0007766554433221), // (present)
    f128_bits_to_uvec4!(0x0007FF76EE65DD54, 0x0003BB32AA219910), // (present)
    f128_bits_to_uvec4!(0x0003FFBB7732EEAA, 0x0001DD995510CC88), // (present)
    f128_bits_to_uvec4!(0x0001FFDDBB997755, 0x0000EECCAA886644), // (present)
    f128_bits_to_uvec4!(0x0000FFEEDDCCBBAA, 0x0000776655443322), // (present)
    f128_bits_to_uvec4!(0x00007FF76EE65DD5, 0x00003BB32AA21991), // (present)
    f128_bits_to_uvec4!(0x00003FFBB7732EEA, 0x00001DD995510CC8), // (present)
    f128_bits_to_uvec4!(0x00001FFDDBB99775, 0x00000EECCAA88664), // (present)
    f128_bits_to_uvec4!(0x00000FFEEDDCCBBA, 0x0000077665544332), // (present)
    f128_bits_to_uvec4!(0x000007FF76EE65DD, 0x000003BB32AA2199), // (present)
    f128_bits_to_uvec4!(0x000003FFBB7732EE, 0x000001DD995510CC), // (present)
    f128_bits_to_uvec4!(0x000001FFDDBB9977, 0x000000EECCAA8866), // (present)
    f128_bits_to_uvec4!(0x000000FFEEDDCCBB, 0x0000007766554433), // (present)
    f128_bits_to_uvec4!(0x0000007FF76EE65D, 0x0000003BB32AA219), // (present)
    f128_bits_to_uvec4!(0x0000003FFBB7732E, 0x0000001DD995510C), // (present)
    f128_bits_to_uvec4!(0x0000001FFDDBB997, 0x0000000EECCAA886), // (present)
    f128_bits_to_uvec4!(0x0000000FFEEDDCCB, 0x0000000776655443), // (present)
    f128_bits_to_uvec4!(0x00000007FF76EE65, 0x00000003BB32AA21), // (present)
    f128_bits_to_uvec4!(0x00000003FFBB7732, 0x00000001DD995510), // (present)
    f128_bits_to_uvec4!(0x00000001FFDDBB99, 0x00000000EECCAA88), // (present)
    f128_bits_to_uvec4!(0x00000000FFEEDDCC, 0x0000000077665544), // (present)
    f128_bits_to_uvec4!(0x000000007FF76EE6, 0x000000003BB32AA2), // (present)
    f128_bits_to_uvec4!(0x000000003FFBB773, 0x000000001DD99551), // (present)
    f128_bits_to_uvec4!(0x000000001FFDDBB9, 0x000000000EECCAA8), // (present)
    f128_bits_to_uvec4!(0x000000000FFEEDDC, 0x0000000007766554), // (present)
    f128_bits_to_uvec4!(0x0000000007FF76EE, 0x0000000003BB32AA), // (present)
    f128_bits_to_uvec4!(0x0000000003FFBB77, 0x0000000001DD9955), // (present)
    f128_bits_to_uvec4!(0x0000000001FFDDBB, 0x0000000000EECCAA), // (present)
    f128_bits_to_uvec4!(0x0000000000FFEEDD, 0x0000000000776655), // (present)
    f128_bits_to_uvec4!(0x00000000007FF76E, 0x00000000003BB32A), // (present)
    f128_bits_to_uvec4!(0x00000000003FFBB7, 0x00000000001DD995), // (present)
    f128_bits_to_uvec4!(0x00000000001FFDDB, 0x00000000000EECCA), // (present)
    f128_bits_to_uvec4!(0x00000000000FFEED, 0x0000000000077665), // (present)
    f128_bits_to_uvec4!(0x000000000007FF76, 0x000000000003BB32), // (present)
    f128_bits_to_uvec4!(0x000000000003FFBB, 0x000000000001DD99), // (present)
    f128_bits_to_uvec4!(0x000000000001FFDD, 0x000000000000EECC), // (present)
    f128_bits_to_uvec4!(0x000000000000FFEE, 0x0000000000007766), // (present)
    f128_bits_to_uvec4!(0x0000000000007FF7, 0x0000000000003BB3), // (present)
    f128_bits_to_uvec4!(0x0000000000003FFB, 0x0000000000001DD9), // (present)
    f128_bits_to_uvec4!(0x0000000000001FFD, 0x0000000000000EEC), // (present)
    f128_bits_to_uvec4!(0x0000000000000FFE, 0x0000000000000776), // (present)
    f128_bits_to_uvec4!(0x00000000000007FF, 0x00000000000003BB), // (present)
    f128_bits_to_uvec4!(0x00000000000003FF, 0x00000000000001DD), // (present)
    f128_bits_to_uvec4!(0x00000000000001FF, 0x00000000000000EE), // (present)
    f128_bits_to_uvec4!(0x00000000000000FF, 0x0000000000000077), // (present)
    f128_bits_to_uvec4!(0x000000000000007F, 0x000000000000003B), // (present)
    f128_bits_to_uvec4!(0x000000000000003F, 0x000000000000001D), // (present)
    f128_bits_to_uvec4!(0x000000000000001F, 0x000000000000000E), // (present)
    f128_bits_to_uvec4!(0x000000000000000F, 0x0000000000000007), // (present)
    f128_bits_to_uvec4!(0x0000000000000007, 0x0000000000000003), // (present)
    f128_bits_to_uvec4!(0x0000000000000003, 0x0000000000000001), // (present)
    f128_bits_to_uvec4!(0x0000000000000001, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Maximum subnormal positive number (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x0000000000000001), // 1.0 + smallest increment (present)
    f128_bits_to_uvec4!(0x3FFEFFFEFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Just below 1.0 (denormal precision) (present)
    f128_bits_to_uvec4!(0x3FFF333333333333, 0x3333333333333333), // 1/3 (approx) (present)
    f128_bits_to_uvec4!(0x4000AAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAB), // 2/3 (approx) (present)
    f128_bits_to_uvec4!(0x3FFEFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE), // 0.999... (just below 1.0 with rounding implications) (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x0000000000000002), // 1.0 + 2 * smallest increment (present)
    f128_bits_to_uvec4!(0x3FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Just below 2.0 (present)
    f128_bits_to_uvec4!(0x8000FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Negative number close to -2^-16382 (present)
    f128_bits_to_uvec4!(0x7FFF800100000000, 0x0000000000000000), // signaling NaN (present)
    f128_bits_to_uvec4!(0x3FFE000000000000, 0x0000000000000000), // 0.5 (present)
    f128_bits_to_uvec4!(0x3FFD800000000000, 0x0000000000000000), // 0.375 (present)
    f128_bits_to_uvec4!(0xBFFE000000000000, 0x0000000000000000), // -0.5 (present)
    f128_bits_to_uvec4!(0x4001C00000000000, 0x0000000000000000), // 7.0 (present)
    f128_bits_to_uvec4!(0x4002400000000000, 0x0000000000000000), // 9.0 (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000FFFF), // denorm edge (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000FFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000000FFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x123456789ABCDEF0, 0x0F1E2D3C4B5A6978), // random (present)
    f128_bits_to_uvec4!(0x876543210FEDCBA9, 0xC3D2E1F019283746), // (present)
    f128_bits_to_uvec4!(0xA5A5A5A55A5A5A5A, 0xFFFFFFFF00000000), // (present)
    f128_bits_to_uvec4!(0xDEADBEEFCAFEBABE, 0x0BADF00DFEEDFACE), // (present)
    f128_bits_to_uvec4!(0xAAAAAAAA55555555, 0x33333333CCCCCCCC), // (present)
    f128_bits_to_uvec4!(0x4000921FB54442D1, 0x8469898CC51701B8), // π (present)
    f128_bits_to_uvec4!(0x40005BF0A8B14576, 0x91EB851EB851EB85), // e (present)
    f128_bits_to_uvec4!(0x3FFE317217F7D1CF, 0x79ABC9E3B39803F2), // log2 (present)
    f128_bits_to_uvec4!(0x3FFD9A209FC69A36, 0xE2FBD87BBD0495F8), // ln(10) (present)
    f128_bits_to_uvec4!(0x0800000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0400000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0200000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0100000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0080000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0040000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0020000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0010000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0008000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0004000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0002000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000800000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000400000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000200000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000100000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000080000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000040000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000020000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000010000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000008000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000004000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000002000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000001000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000800000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000400000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000200000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000100000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000080000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000040000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000020000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000010000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000008000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000004000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000002000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000001000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000800000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000400000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000200000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000100000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000080000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000040000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000020000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000010000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000008000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000004000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000002000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000001000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000800, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000400, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000200, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000100, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000080, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000040, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000020, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000010, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000008, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000004, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000002, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x8000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x4000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x2000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x1000000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0800000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0400000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0200000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0100000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0080000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0040000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0020000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0010000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0008000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0004000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0002000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0001000000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000800000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000400000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000200000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000100000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000080000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000040000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000020000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000010000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000008000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000004000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000002000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000001000000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000800000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000400000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000200000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000100000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000080000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000040000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000020000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000010000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000008000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000004000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000002000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000001000000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000800000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000400000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000200000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000100000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000080000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000040000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000020000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000010000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000008000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000004000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000002000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000001000), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000800), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000400), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000200), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000100), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000080), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000040), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000020), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000010), // (present)
    f128_bits_to_uvec4!(0x2000000000000000, 0x0000000000000001), // (present)
    f128_bits_to_uvec4!(0x4000000000000000, 0x0000000000000003), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000000F), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000001F), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000003F), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000007F), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000000000FF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000000001FF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000000003FF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000000007FF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000000FFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000001FFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000003FFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000000007FFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000001FFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000003FFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000007FFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000000FFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000001FFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000003FFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000007FFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000001FFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000003FFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000007FFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000FFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000001FFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000003FFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000007FFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000000FFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000001FFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000003FFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000007FFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000000FFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000001FFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000003FFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000007FFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000FFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000001FFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000003FFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000007FFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00000FFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00001FFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00003FFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00007FFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0000FFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0001FFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0003FFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0007FFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000FFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x001FFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x003FFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x007FFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x00FFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x01FFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x03FFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x07FFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x0FFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x1FFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x3FFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x7FFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000001, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000003, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000007, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000000000F, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000000001F, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000000003F, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000000007F, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000000000FF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000000001FF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000000003FF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000000007FF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000000FFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000001FFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000003FFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000007FFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000001FFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000003FFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000007FFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000000FFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000001FFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000003FFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000007FFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000000FFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000001FFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000003FFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000007FFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000000FFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000001FFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000003FFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000007FFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000000FFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000001FFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000003FFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000007FFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000000FFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000001FFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000003FFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0000007FFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000000FFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000001FFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000003FFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000007FFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00000FFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00001FFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00003FFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00007FFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0001FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0003FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0007FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x000FFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x001FFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x003FFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x007FFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x00FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x01FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x03FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x07FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x0FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x1FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0xE000000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xF000000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xF800000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFC00000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFE00000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFF00000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFF80000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFC0000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFE0000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFF0000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFF8000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFC000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFE000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFC00000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFE00000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFF00000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFF80000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFC0000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFE0000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFF0000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFF8000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFC000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFE000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFF000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFF800000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFC00000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFE00000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFF00000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFF80000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFC0000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFE0000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFF0000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFF8000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFC000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFE000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFF000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFF800000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFC00000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFE00000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFF00000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFF80000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFC0000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFE0000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFF0000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFF8000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFC000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFE000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFF000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFF800, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFC00, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFE00, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFF00, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFF80, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFC0, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFE0, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFF0, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFF8, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFC, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFE, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xC000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xE000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xF000000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xF800000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFC00000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFE00000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFF00000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFF80000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFC0000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFE0000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFF0000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFF8000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFC000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFE000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFF000000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFF800000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFC00000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFE00000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFF00000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFF80000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFC0000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFE0000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFF0000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFF8000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFC000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFE000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFF000000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFF800000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFC00000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFE00000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF00000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF80000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFC0000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFE0000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFF0000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFF8000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFC000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFE000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFF000000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFF800000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFC00000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFE00000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFF00000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFF80000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFC0000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFE0000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFF0000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFF8000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFC000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFE000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF000), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF800), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFC00), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFE00), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFF00), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFF80), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFC0), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFE0), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFF0), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFF8), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFC), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE), // (present)
    f128_bits_to_uvec4!(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x3F8E000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x3F8F000000000000, 0x0000000000000001), // (present)
    f128_bits_to_uvec4!(0x7FFE000000000000, 0x0000000000000001), // (present)
    f128_bits_to_uvec4!(0x7FFE000000000000, 0x0000000000000000), // (present)
    f128_bits_to_uvec4!(0x7FFEFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // (present)
    f128_bits_to_uvec4!(0x7FFF400000000000, 0x0000000000000000), // Signaling NaN with different payload (present)
    f128_bits_to_uvec4!(0xFFFF400000000000, 0x0000000000000000), // Negative signaling NaN (present)
    f128_bits_to_uvec4!(0xFFFEFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Maximum negative finite (present)
    f128_bits_to_uvec4!(0x3F8D000000000000, 0x0000000000000000), // 2^-114 (doesn't affect 1.0) (present)
    f128_bits_to_uvec4!(0x3F8C000000000000, 0x0000000000000000), // 2^-115 (beyond precision) (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x0000000000000003), // 1 + 3*epsilon (round up) (present)
    f128_bits_to_uvec4!(0x3FFE800000000000, 0x0000000000000001), // 0.5 + tiny (round up) (present)
    f128_bits_to_uvec4!(0x3FFE7FFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), // Just under 0.5 (round down) (present)
    f128_bits_to_uvec4!(0x3F00000000000000, 0x0000000000000000), // Very small number (2^-16256) (present)
    f128_bits_to_uvec4!(0x7FFE800000000000, 0x0000000000000000), // Half of max finite (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x8000000000000000), // 1 + 0.5 (present)
    f128_bits_to_uvec4!(0x0001000000000000, 0x0000000000000001), // Min normal + epsilon (present)
    f128_bits_to_uvec4!(0xBFFF000000000000, 0x0000000000000001), // -(1 + epsilon) (present)
    f128_bits_to_uvec4!(0x4010000000000000, 0x0000000000000000), // 4.0 (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x0000000000000007), // 1 + 7*epsilon (sticky bits) (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x000000000000000F), // 1 + 15*epsilon (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000000B), // Eleventh smallest positive denormal (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000000C), // Twelfth smallest positive denormal (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000000D), // Thirteenth smallest positive denormal (present)
    f128_bits_to_uvec4!(0x0000000000000000, 0x000000000000000E), // Fourteenth smallest positive denormal (present)
    f128_bits_to_uvec4!(0x8001000000000000, 0x0000000000000001), // -(Smallest normal + 1 ulp) (present)
    f128_bits_to_uvec4!(0x3F7FE00000000000, 0x0000000000000000), // 2^-127 (small number, could be halfway point for some sum) (present)
    f128_bits_to_uvec4!(0x7FFF800000000000, 0x0000000000000002), // Quiet NaN (different payload) (present)
    f128_bits_to_uvec4!(0x3FFC000000000000, 0x0000000000000000), // 0.125 (present)
    f128_bits_to_uvec4!(0x3FFF000000000000, 0x00000000FFFFFFFF), // 1.0 + large mantissa part (present)

];


const VERTEX_SHADER: &str = r#"
    #version 300 es
    void main() {
        // Simple fullscreen quad
        vec2 positions[4] = vec2[](
            vec2(-1.0, -1.0), // Bottom-left
            vec2( 1.0, -1.0), // Bottom-right
            vec2(-1.0,  1.0), // Top-left
            vec2( 1.0,  1.0)  // Top-right
        );
        gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
    }
"#;

const FRAGMENT_SHADER: &str = r#"
#version 300 es
precision highp float;
precision highp int;

uniform uvec4 u_values[648];

out vec4 fragColor;

#define F32_SIGN_MASK  0x80000000u
#define F32_EXP_MASK   0x7F800000u
#define F32_MANT_MASK  0x007FFFFFu

#define F128_SIGN_MASK  0x80000000u
#define F128_EXP_MASK   0x7FFF0000u
#define F128_MANT_MASK  0x0000FFFFu

#define F32_EXP_BIAS    127u
#define F128_EXP_BIAS   16383u

#define F128_IMPLICIT_BIT 0x10000u

#define F32_QNAN_BIT   0x00400000u
#define F32_SNAN_MASK  0x003FFFFFu

#define F128_QNAN_BIT  0x00008000u
#define F128_SNAN_MASK 0x00007FFFu 

#define ROUND_G_BIT_MASK 0x4u 
#define ROUND_R_BIT_MASK 0x2u 
#define ROUND_S_BIT_MASK 0x1u 

bool is_zero(uvec4 f) {
    return ((f.w & F128_EXP_MASK) == 0u) && 
           ((f.w & F128_MANT_MASK) == 0u) && 
           (f.x == 0u) && (f.y == 0u) && (f.z == 0u);
}

bool is_inf_or_nan_exp(uvec4 f) {
    return ((f.w & F128_EXP_MASK) >> 16) == 0x7FFFu;
}

bool is_inf(uvec4 f) {
    return is_inf_or_nan_exp(f) && ((f.w & F128_MANT_MASK) == 0u && f.x == 0u && f.y == 0u && f.z == 0u);
}

bool is_nan(uvec4 f) {
    return is_inf_or_nan_exp(f) && !is_inf(f);
}

bool is_signaling_nan(uvec4 f) {
    if (!is_nan(f)) return false;
    // Check if most significant bit of mantissa is 0 (signaling NaN)
    return (f.w & 0x8000u) == 0u;
}

uvec4 zero(uint sign) {
    return uvec4(0u, 0u, 0u, sign);
}

uvec4 inf(uint sign) {
    return uvec4(0u, 0u, 0u, sign | 0x7FFF0000u);
}

uvec4 nan() {
    return uvec4(0u, 0u, 0u, 0x7FFF8000u);
}

uvec4 quiet_nan(uvec4 snan) {
    return uvec4(snan.x, snan.y, snan.z, (snan.w | 0x8000u) & 0x7FFFFFFFu );
}

uvec4 signed_quiet_nan(uvec4 snan){
    return uvec4(snan.x, snan.y, snan.z, snan.w | 0x8000u );
}

uvec4 nan_with_payload(uint sign, uint f32_mantissa) {
    
    bool is_quiet = (f32_mantissa & F32_QNAN_BIT) != 0u; 
    uint payload = f32_mantissa & 0x003FFFFFu; // Lower 22 bits of F32 mantissa
    
    if (!is_quiet && payload == 0u) {
        return uvec4(0u, 0u, 0u, sign | 0x8FFF0000u | F128_QNAN_BIT);
    }

    uvec4 result = uvec4(0u, 0u, 0u, 0u);
    result.w = sign | F128_EXP_MASK;
    if (is_quiet) {
        result.w |= F128_QNAN_BIT;
    }
    
    uint high_payload = (payload >> 7u) & 0x7FFFu;
    result.w |= high_payload;
    
    uint low_payload = payload & 0x7Fu;
    result.z = low_payload << 25u;
    
    return result;
}
    
uvec4 float_to_f128(float f) {
    uint fb = floatBitsToUint(f);
    uint sign = fb & F32_SIGN_MASK;
    uint exp32 = (fb & F32_EXP_MASK) >> 23u;
    uint mant = fb & F32_MANT_MASK;
    
    if (exp32 == 0u && mant == 0u) {
        return zero(sign);
    }
    
    if (exp32 == 0xFFu) {
        return (mant == 0u) ? inf(sign) : nan_with_payload(sign, mant);
    }

    bool is_quiet = (mant & F32_QNAN_BIT) != 0u;
    
    uint exp128;
    uint mant_bits;
    
    if (exp32 == 0u) {
        uint leading_zeros = 0u;
        uint temp_mant = mant;
        
        for (uint i = 0u; i < 23u; i++) {
            if ((temp_mant & 0x400000u) != 0u) break;
            temp_mant <<= 1u;
            leading_zeros++;
        }
        
        mant_bits = (mant << (leading_zeros + 1u)) & F32_MANT_MASK;
        exp128 = - F32_EXP_BIAS - leading_zeros + F128_EXP_BIAS;
    } else {
        exp128 = exp32 - F32_EXP_BIAS + F128_EXP_BIAS;
        mant_bits = mant;
    }
    
    uvec4 mantissa = uvec4(0u);
    
    mantissa.x = 0u;
    mantissa.y = 0u;
    mantissa.z = mant_bits << 25u;
    mantissa.w = (mant_bits >> 7u) & 0xFFFFu;
    
    uint hi = sign | (exp128 << 16u) | mantissa.w;
    
    return uvec4(mantissa.x, mantissa.y, mantissa.z, hi);
}


void sum32(uint a, uint b, uint carry_in , out uint result, out uint carry_out) {
    uint temp_sum = a + b;
    uint temp_carry = (temp_sum < a) ? 1u : 0u;
    result = temp_sum + carry_in;
    uint carry2 = (result < temp_sum) ? 1u : 0u;
    carry_out = temp_carry + carry2;
}

void sub32(uint a, uint b, uint borrow_in, out uint result, out uint borrow_out) {
    uint temp = b + borrow_in;
    result = a - temp;
    borrow_out = (a < temp || (temp < b && borrow_in == 1u)) ? 1u : 0u;
}

bool is_mantissa_gte(uvec4 a, uvec4 b) {
    if (a.w > b.w) return true;
    if (a.w < b.w) return false;
    if (a.z > b.z) return true;
    if (a.z < b.z) return false;
    if (a.y > b.y) return true;
    if (a.y < b.y) return false;
    return a.x >= b.x;
}
    
uvec4 add_mantissa(uvec4 a, uvec4 b) {
    uvec4 result;
    uint carry=0u;

    sum32(a.x, b.x, carry, result.x, carry);
    sum32(a.y, b.y, carry, result.y, carry);
    sum32(a.z, b.z, carry, result.z, carry);
    sum32(a.w, b.w, carry, result.w, carry);
   
    return result;
}

uvec4 twos_complement(uvec4 a) {
    uvec4 result;
    uint carry = 1u;

    sum32(~a.x, 0u, carry, result.x, carry);
    sum32(~a.y, 0u, carry, result.y, carry);
    sum32(~a.z, 0u, carry, result.z, carry);
    sum32(~a.w, 0u, carry, result.w, carry);

    return result;
}

uvec4 sub_mantissa(uvec4 a, uvec4 b, inout uvec4 rem) {
    uvec4 result;
    uint borrow = 0u;

    // Handle rem: if rem is non-zero, treat it as a borrow into b.
    if (any(notEqual(rem, uvec4(0u)))) {
        rem = twos_complement(rem); 
        borrow = 1u;
    }

    sub32(a.x, b.x, borrow, result.x, borrow);
    sub32(a.y, b.y, borrow, result.y, borrow);
    sub32(a.z, b.z, borrow, result.z, borrow);
    sub32(a.w, b.w, borrow, result.w, borrow);

    return result;
}

void shift_right(inout uvec4 mant_a, uint shift, inout uvec4 rem) {

    for (uint i = 0u; i < shift; ++i) {
        rem.x = (rem.x >> 1) | (rem.y << 31);
        rem.y = (rem.y >> 1) | (rem.z << 31);
        rem.z = (rem.z >> 1) | (rem.w << 31);
        rem.w = (rem.w >> 1) | (mant_a.x << 31);
        mant_a.x = (mant_a.x >> 1) | (mant_a.y << 31);
        mant_a.y = (mant_a.y >> 1) | (mant_a.z << 31);
        mant_a.z = (mant_a.z >> 1) | (mant_a.w << 31);
        mant_a.w >>= 1;
    }
}

void norm(inout uvec4 mantissa, inout uint exponent, inout uvec4 rem) {
    // Handle subnormal case - don't normalize below exponent 1
    while ((mantissa.w & F128_IMPLICIT_BIT) == 0u && exponent > 1u) {
        exponent--;
        mantissa.w = (mantissa.w << 1) | (mantissa.z >> 31);
        mantissa.z = (mantissa.z << 1) | (mantissa.y >> 31);
        mantissa.y = (mantissa.y << 1) | (mantissa.x >> 31);
        mantissa.x = (mantissa.x << 1) | (rem.w >> 31);
        rem.w = (rem.w << 1) | (rem.z >> 31);
        rem.z = (rem.z << 1) | (rem.y >> 31);
        rem.y = (rem.y << 1) | (rem.x >> 31);
        rem.x <<= 1;
    }
    
    // If we hit minimum exponent and still not normalized, result is subnormal
    if (exponent == 1u && (mantissa.w & F128_IMPLICIT_BIT) == 0u) {
        exponent = 0u; // Subnormal exponent
    }
}

bool should_round_up_rne(bool lsb, uint grs) {
    bool g = (grs & ROUND_G_BIT_MASK) != 0u;
    bool r = (grs & ROUND_R_BIT_MASK) != 0u;
    bool s = (grs & ROUND_S_BIT_MASK) != 0u;
    
    return (g && (r || s)) || (g && !r && !s && lsb);
}

uvec4 add_f128(uvec4 a, uvec4 b) {    
    // Handle signaling NaNs first (they should signal and become quiet)
    if (is_signaling_nan(a) || is_nan(a)){
        // Signal if "invalid operation" (signaling nan) exception (if enabled)
        // Then return a quiet NaN, possibly with a payload from the sNaN
        return quiet_nan(a);
    }

    if (is_signaling_nan(b) || is_nan(b)) {
        // Signal if "invalid operation" (signaling nan) exception (if enabled)
        // Then return a quiet NaN, possibly with a payload from the sNaN
        return quiet_nan(b);
    }


    // Extract signs early for infinity handling
    uint sign_a = a.w & F128_SIGN_MASK;
    uint sign_b = b.w & F128_SIGN_MASK;

    // Handle infinities
    bool a_is_inf = is_inf(a);  
    bool b_is_inf = is_inf(b);
    if (a_is_inf && b_is_inf) {
        // +inf + -inf = NaN, +inf + +inf = +inf, -inf + -inf = -inf
        if (sign_a != sign_b) {
            return nan();
        }
        return a; // Both have same sign
    }
    if (a_is_inf) return a;
    if (b_is_inf) return b;

    // Handle zeros - IEEE 754 specific rules
    bool a_is_zero = is_zero(a);
    bool b_is_zero = is_zero(b);
    if (a_is_zero && b_is_zero) {
        // -0 + -0 = -0, all other combinations = +0
        if (sign_a != 0u && sign_b != 0u) {
            return zero(F128_SIGN_MASK);
        }
        return zero(0u);
    }
    if (a_is_zero) return b;
    if (b_is_zero) return a;

    // Extract exponent and mantissa
    uint exp_a = (a.w & F128_EXP_MASK) >> 16;
    uint exp_b = (b.w & F128_EXP_MASK) >> 16;
    uvec4 mant_a = uvec4(a.x, a.y, a.z, a.w & F128_MANT_MASK);
    uvec4 mant_b = uvec4(b.x, b.y, b.z, b.w & F128_MANT_MASK);

    // Add implicit bit for normal numbers
    if (exp_a != 0u) mant_a.w |= F128_IMPLICIT_BIT;
    else if (exp_b != 0u) exp_a = 1u;

    if (exp_b != 0u) mant_b.w |= F128_IMPLICIT_BIT;
    else if (exp_a != 0u) exp_b = 1u;

    bool both_sub = (exp_a == 0u) && (exp_b == 0u); 
    
    // Handle exponent difference greater than precision (113 bits for binary128)
    uint exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    if (exp_diff > 114u) {
        // One operand is too small to affect the result
        return (exp_a > exp_b) ? a : b;
    }

    uint common_exp = max(exp_a, exp_b);
    uint shift_a = common_exp - exp_a;
    uint shift_b = common_exp - exp_b;
    
    uvec4 rem = uvec4(0u);
    // Align mantissas
    shift_right(mant_a, shift_a, rem);
    shift_right(mant_b, shift_b, rem);

    uint result_sign;
    uvec4 result_mant;
    uint result_exp = common_exp;
    uint grs = 0u;

    if (sign_a == sign_b) {
        // Same sign addition
        result_mant = add_mantissa(mant_a, mant_b);
        
        // Check for overflow and normalize
        if (result_mant.w >= (F128_IMPLICIT_BIT << 1)) {
            result_exp += 1u;
            shift_right(result_mant, 1u, rem);
            
        }else if (both_sub && (result_mant.w >= F128_IMPLICIT_BIT)) {       
            result_exp = 1u;
        }

        result_sign = sign_a;

    } else {
        // Different sign subtraction
        bool a_abs_is_bigger = is_mantissa_gte(mant_a, mant_b);
        uvec4 bigger_mant = a_abs_is_bigger ? mant_a : mant_b;
        uvec4 smaller_mant = a_abs_is_bigger ? mant_b : mant_a;

        result_mant = sub_mantissa(bigger_mant, smaller_mant, rem);
        
        // Check if result is zero
        if ((result_mant.x | result_mant.y | result_mant.z | result_mant.w) == 0u && (rem.x | rem.y | rem.z | rem.w) == 0u) {
            return zero(0u); // Result is +0 for subtraction yielding zero
        }
        // Normalize the result
        norm(result_mant, result_exp, rem);

        result_sign = a_abs_is_bigger ? sign_a : sign_b;
        
    }
    
    if (result_exp >= 0x7FFFu) return inf(result_sign);


    // Calculate rounding bits from remainder
    bool g = (rem.w & 0x80000000u) != 0u;
    bool r = (rem.w & 0x40000000u) != 0u;  
    bool s = ((rem.w & 0x3FFFFFFFu) != 0u) || (rem.z != 0u) || (rem.y != 0u) || (rem.x != 0u);
    grs = (g ? ROUND_G_BIT_MASK : 0u) | (r ? ROUND_R_BIT_MASK : 0u) | (s ? ROUND_S_BIT_MASK : 0u);
    bool lsb = (result_mant.x & 1u) != 0u; // ok if mantissa fully normalized

    // Apply rounding (round to nearest, ties to even)
    if (should_round_up_rne(lsb, grs)) {
        result_mant = add_mantissa(result_mant, uvec4(1u, 0u, 0u, 0u));
        
        // Check for mantissa overflow after rounding
        if (result_mant.w >= (F128_IMPLICIT_BIT << 1)) {
            
            result_exp += 1u;
            // Check for exponent overflow after rounding
            if (result_exp >= 0x7FFFu) return inf(result_sign);
        
            shift_right(result_mant, 1u, rem);
            
        }
    }

    result_mant.w &= F128_MANT_MASK;    
    // Construct final result
    uint w = result_sign | (result_exp << 16) | result_mant.w;
    return uvec4(result_mant.x, result_mant.y, result_mant.z, w);
}

uvec4 sub_f128(uvec4 a, uvec4 b){
    uint flip_sign_up = ((( b.w >> 31) ^ 1u ) << 31 ) | (b.w & 0x7FFFFFFFu);
    return add_f128(a, uvec4(b.x, b.y, b.z, flip_sign_up));
}


void multiply_mantissa_226(uvec4 a, uvec4 b, out uvec4 product_high, out uvec4 product_low) {

    uint prod[8];
    for (uint i = 0u; i < 8u; ++i) prod[i] = 0u;

    a.w &= 0x1FFFFu;
    b.w &= 0x1FFFFu;

    for (uint i = 0u; i < 4u; ++i) {
        uint ai = a[i];
        for (uint j = 0u; j < 4u; ++j) {
            uint bj = b[j];
            
            // Decompose input
            uint a_lo = ai & 0xFFFFu;
            uint a_hi = ai >> 16u;
            uint b_lo = bj & 0xFFFFu;
            uint b_hi = bj >> 16u;

            // Partial products
            uint p0 = a_lo * b_lo;
            uint p1 = a_lo * b_hi;
            uint p2 = a_hi * b_lo;
            uint p3 = a_hi * b_hi;

            uint carry = 0u, carry2 = 0u, carry3 = 0u;
            uint lo = 0u; 
            uint hi = 0u;

        
            sum32(p0, (p1 & 0xFFFFu) << 16u, carry, lo, carry); //can overflow once -> carry
            sum32(lo, (p2 & 0xFFFFu) << 16u, carry2, lo, carry2); // can overflow once -> carry2
            sum32(prod[i + j], lo, carry3, prod[i + j], carry3); // can overflow once -> carry 3
            uint carry_lo = carry + carry2 + carry3;
            
            uint no_carry = 0u;
            sum32(p3, (p1 >> 16u), carry_lo, hi, carry); 
            sum32(hi, (p2 >> 16u), no_carry , hi, carry2);
            uint carry_hi = carry + carry2;
            
            uint k = i + j + 1u;
            while (carry_hi > 0u || hi > 0u) {
                uint next = (hi > 0u) ? hi : 0u;
                sum32(prod[k], next, carry_hi, prod[k], carry_hi);
                hi = 0u;
                ++k;
            }
        }
    }

    product_low = uvec4(
        (prod[0] << 16u),
        (prod[1] << 16u) | (prod[0] >> 16u),
        (prod[2] << 16u) | (prod[1] >> 16u),
        (prod[3] << 16u) | (prod[2] >> 16u)
    );

    product_high = uvec4(
        (prod[3] >> 16u) | (prod[4] << 16u),
        (prod[4] >> 16u) | (prod[5] << 16u),
        (prod[5] >> 16u) | (prod[6] << 16u),
        (prod[6] >> 16u) | (prod[7] << 16u)
    );

}

void shift_right_226(inout uvec4 high, inout uvec4 low, uint shift, out uvec4 shifted_bits) {
    
    uint full[8];
    full[0] = low.x;
    full[1] = low.y;
    full[2] = low.z;
    full[3] = low.w;
    full[4] = high.x;
    full[5] = high.y;
    full[6] = high.z;
    full[7] = high.w;

    
    uint word_shift = shift / 32u;
    uint bit_shift = shift % 32u;
    uint inv_bit_shift = 32u - bit_shift;

    uint new_full[8] = uint[8](0u,0u,0u,0u,0u,0u,0u,0u);
    for (uint i = 0u; i < 8u; ++i) {
        if ((i + word_shift) < 8u) {
            new_full[i] = full[i + word_shift] >> bit_shift;
            if (bit_shift != 0u && (i + word_shift + 1u) < 8u) {
                new_full[i] |= full[i + word_shift + 1u] << inv_bit_shift;
            }
        }
    }

    // Write back shifted bits (for sticky/rounding)
    shifted_bits = uvec4(
        full[0] << (32u - bit_shift),
        full[1] << (32u - bit_shift),
        full[2] << (32u - bit_shift),
        full[3] << (32u - bit_shift)
    );

    // Write result back to high/low
    low.x = new_full[0];
    low.y = new_full[1];
    low.z = new_full[2];
    low.w = new_full[3];
    high.x = new_full[4];
    high.y = new_full[5];
    high.z = new_full[6];
    high.w = new_full[7];
}

void norm_226(inout uvec4 mantissa_hi, inout uvec4 mantissa_lo, inout int exponent, inout uvec4 rem) {
    // Handle subnormal case - don't normalize below exponent 1
    while ((mantissa_hi.w & F128_IMPLICIT_BIT) == 0u ) {
        exponent--;
        mantissa_hi.w = (mantissa_hi.w << 1) | (mantissa_hi.z >> 31);
        mantissa_hi.z = (mantissa_hi.z << 1) | (mantissa_hi.y >> 31);
        mantissa_hi.y = (mantissa_hi.y << 1) | (mantissa_hi.x >> 31);
        mantissa_hi.x = (mantissa_hi.x << 1) | (mantissa_lo.w >> 31);
        mantissa_lo.w = (mantissa_lo.w << 1) | (mantissa_lo.z >> 31);
        mantissa_lo.z = (mantissa_lo.z << 1) | (mantissa_lo.y >> 31);
        mantissa_lo.y = (mantissa_lo.y << 1) | (mantissa_lo.x >> 31);
        mantissa_lo.x = (mantissa_lo.x << 1) | (rem.w >> 31);
        rem.w = (rem.w << 1) | (rem.z >> 31);
        rem.z = (rem.z << 1) | (rem.y >> 31);
        rem.y = (rem.y << 1) | (rem.x >> 31);
        rem.x <<= 1;
    }
    
}


uvec4 mul_f128(uvec4 a, uvec4 b) {
    // Handle signaling NaNs first
    if (is_signaling_nan(a) || is_nan(a)) {
        return signed_quiet_nan(a);
    }
    if (is_signaling_nan(b) || is_nan(b)) {
        return signed_quiet_nan(b);
    }

    // Extract signs
    uint sign_a = a.w & F128_SIGN_MASK;
    uint sign_b = b.w & F128_SIGN_MASK;
    uint result_sign = sign_a ^ sign_b;

    // Handle infinities
    bool a_is_inf = is_inf(a);
    bool b_is_inf = is_inf(b);
    bool a_is_zero = is_zero(a);
    bool b_is_zero = is_zero(b);

    if (a_is_inf || b_is_inf) {
        // inf * 0 = NaN
        if (a_is_zero || b_is_zero) {
            return nan();
        }
        // inf * finite = inf (with appropriate sign)
        return inf(result_sign);
    }

    // Handle zeros
    if (a_is_zero || b_is_zero) {
        return zero(result_sign);
    }

    // Extract exponents and mantissas
    uint exp_a = (a.w & F128_EXP_MASK) >> 16;
    uint exp_b = (b.w & F128_EXP_MASK) >> 16;
    uvec4 mant_a = uvec4(a.x, a.y, a.z, a.w & F128_MANT_MASK);
    uvec4 mant_b = uvec4(b.x, b.y, b.z, b.w & F128_MANT_MASK);

    // Add implicit bit for normal numbers
    if (exp_a != 0u) mant_a.w |= F128_IMPLICIT_BIT;
    else if (exp_b != 0u) exp_a = 1u;

    if (exp_b != 0u) mant_b.w |= F128_IMPLICIT_BIT;
    else if (exp_a != 0u) exp_b = 1u;

    int result_exp_signed = int(exp_a) + int(exp_b) - int(F128_EXP_BIAS);

    uvec4 product_high, product_low, product_low_low = uvec4(0u);
    multiply_mantissa_226(mant_a, mant_b, product_high, product_low);

    if ((product_high.w & 0x00020000u) != 0u){
        result_exp_signed += 1;
        // Shift right by 1 bit, keeping track of the bit that gets shifted out
        bool lost_bit = (product_high.x & 1u) != 0u;
        // Shift the 226-bit result right by 1

        product_low_low.x = (product_low_low.x >> 1) | ((product_low_low.y & 1u) << 31);
        product_low_low.y = (product_low_low.y >> 1) | ((product_low_low.z & 1u) << 31);
        product_low_low.z = (product_low_low.z >> 1) | ((product_low_low.w & 1u) << 31);
        product_low_low.w = (product_low_low.w >> 1) | ((product_low.x & 1u) << 31);

        product_low.x = (product_low.x >> 1) | ((product_low.y & 1u) << 31);
        product_low.y = (product_low.y >> 1) | ((product_low.z & 1u) << 31);
        product_low.z = (product_low.z >> 1) | ((product_low.w & 1u) << 31);
        product_low.w = (product_low.w >> 1) | ((product_high.x & 1u) << 31);

        product_high.x = (product_high.x >> 1) | ((product_high.y & 1u) << 31);
        product_high.y = (product_high.y >> 1) | ((product_high.z & 1u) << 31);
        product_high.z = (product_high.z >> 1) | ((product_high.w & 1u) << 31);
        product_high.w = product_high.w >> 1;
        
    }else{
        norm_226(product_high, product_low, result_exp_signed, product_low_low);
    }

    // Check for underflow
    if (result_exp_signed <= 0) {
        // Result is subnormal or underflows to zero
        int shift_amount = 1 - result_exp_signed;
        
        if (shift_amount > 113) {
            // Complete underflow to zero
            return zero(result_sign);
        }
        
        // Shift mantissa right to make it subnormal
        shift_right_226(product_high, product_low, uint(shift_amount), product_low_low);
        
        result_exp_signed = 0;
    }

    // Check for overflow
    if (result_exp_signed >= 0x7FFF) {
        return inf(result_sign);
    }

    uint result_exp = uint(result_exp_signed);

    uvec4 result_mant = product_high;
    if (result_exp != 0u) {
        result_mant.w &= F128_MANT_MASK; // Remove implicit bit
    }

    bool g, r, s;    
    g = (product_low.w & 0x80000000u) != 0u;
    r = (product_low.w & 0x40008000u) != 0u;
    s = ((product_low.w & 0x3FFFFFFFu) != 0u) || 
            (product_low.z != 0u) || (product_low.y != 0u) || (product_low.x != 0u) || any(notEqual(product_low_low,uvec4(0u)));

    uint grs = (g ? ROUND_G_BIT_MASK : 0u) | 
               (r ? ROUND_R_BIT_MASK : 0u) | 
               (s ? ROUND_S_BIT_MASK : 0u);
    
    bool lsb = (result_mant.x & 1u) != 0u;

    // Apply rounding (round to nearest, ties to even)
    if (should_round_up_rne(lsb, grs)) {
        result_mant = add_mantissa(result_mant, uvec4(1u, 0u, 0u, 0u));
        
        // Check for mantissa overflow after rounding
        if (result_exp != 0u && (result_mant.w & F128_IMPLICIT_BIT) != 0u) {
            // Mantissa overflowed, increment exponent
            result_exp += 1u;
            
            // Check for exponent overflow
            if (result_exp >= 0x7FFFu) {
                return inf(result_sign);
            }
            
            // Reset mantissa to 0 (implicit bit will be added by format)
            result_mant = uvec4(0u, 0u, 0u, 0u);
        } else if (result_exp == 0u && (result_mant.w & F128_IMPLICIT_BIT) != 0u) {
            // Subnormal rounded up to normal
            result_exp = 1u;
            result_mant.w &= F128_MANT_MASK;
        }
    }

    // Ensure mantissa doesn't include implicit bit for final result
    if (result_exp != 0u) {
        result_mant.w &= F128_MANT_MASK;
    }

    // Construct final result
    uint w = result_sign | (result_exp << 16) | result_mant.w;
    return uvec4(result_mant.x, result_mant.y, result_mant.z, w);
}


void main() {
    // Each 4x4 block represents one f128 result.
    // The texture has dimensions (N*4) x (N*4) where N is INPUT_VALUES.len() (24).
    // So, texture_width = 96, texture_height = 96.

    uint i_index = uint(floor(gl_FragCoord.x / 4.0));
    uint j_index = uint(floor(gl_FragCoord.y / 4.0));

    uvec4 val_a = u_values[i_index];
    uvec4 val_b = u_values[j_index];

    uvec4 result_f128_a = mul_f128(val_a, val_b);
    uvec4 result_f128_s = mul_f128(val_a, val_b);

    uint col_in_block = uint(gl_FragCoord.x) % 4u; // Column within the current 4x4 block (0-3)
    uint row_in_block = uint(gl_FragCoord.y) % 4u; // Row within the current 4x4 block (0-3)

    uint byteIndex = row_in_block * 4u + col_in_block;

    uint wordIndex = byteIndex / 4u;   // Which u32 word (0-3)
    uint byteInWord = byteIndex % 4u;  // Which byte within that u32 word (0-3)

    uint word_a;
    uint word_s;
    if (wordIndex == 0u) {
        word_a = result_f128_a.x;
        word_s = result_f128_s.x;
    } else if (wordIndex == 1u) {
        word_a = result_f128_a.y;
        word_s = result_f128_s.y;
    } else if (wordIndex == 2u) {
        word_a = result_f128_a.z;
        word_s = result_f128_s.z;
    } else { // wordIndex == 3u
        word_a = result_f128_a.w;
        word_s = result_f128_s.w;
    }

    uint shift = byteInWord * 8u;
    uint byteVal_a = (word_a >> shift) & 0xFFu;
    uint byteVal_s = (word_s >> shift) & 0xFFu;

    float red = float(byteVal_a) / 255.0;
    float green = float(byteVal_s) / 255.0;
    fragColor = vec4(red, green, 0.0, 1.0);
}

    
"#;

pub struct ShaderApp {
    gl: Arc<glow::Context>,
    program: glow::Program,
    vao: glow::VertexArray,
    framebuffer: glow::Framebuffer,
    texture_width: i32,
    texture_height: i32,
}

impl ShaderApp {
    pub fn new(gl: Arc<glow::Context>) -> Self {
        unsafe {
            let vertex_shader = compile_shader(&gl, glow::VERTEX_SHADER, VERTEX_SHADER)
                .unwrap_or_else(|err| panic!("Vertex shader compilation failed: {}", err));
            let fragment_shader = compile_shader(&gl, glow::FRAGMENT_SHADER, FRAGMENT_SHADER)
                .unwrap_or_else(|err| panic!("Fragment shader compilation failed: {}", err));
            let program = link_program(&gl, vertex_shader, fragment_shader)
                .unwrap_or_else(|err| panic!("Program linking failed: {}", err));

            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));

            let texture_width = (INPUT_VALUES.len() * 4) as i32; // N values, each 4 pixels wide
            let texture_height = (INPUT_VALUES.len() * 4) as i32; // Each value is 4 pixels high

            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                texture_width,
                texture_height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                None,
            );

            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );

            let framebuffer = gl.create_framebuffer().unwrap();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );

            assert_eq!(
                gl.check_framebuffer_status(glow::FRAMEBUFFER),
                glow::FRAMEBUFFER_COMPLETE
            );

            Self {
                gl,
                program,
                vao,
                framebuffer,
                texture_width,
                texture_height,
            }
        }
    }

    fn render_values(&self) -> Vec<u8> {
        let gl = &self.gl;
        let num_pixels = (self.texture_width * self.texture_height * 4) as usize; // RGBA
        let mut pixels = vec![0u8; num_pixels];

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.framebuffer));
            gl.viewport(0, 0, self.texture_width, self.texture_height);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));

            let loc_values = gl.get_uniform_location(self.program, "u_values")
                .ok_or_else(|| "Could not find uniform location 'u_f128_values'".to_string()).unwrap();

            let flat_values: &[u32] = std::slice::from_raw_parts(INPUT_VALUES.as_ptr() as *const u32,INPUT_VALUES.len() * 4);

            gl.uniform_4_u32_slice(Some(&loc_values), flat_values);

            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            gl.read_pixels(
                0,
                0,
                self.texture_width,
                self.texture_height,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(&mut pixels[..]),
            );

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }

        pixels
    }
}

impl App for ShaderApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data = self.render_values();

            
            let input_converted: Vec<f128> = INPUT_VALUES.iter()
                                .map(|&x| to_f128(x))
                                .collect();
            let mut count = 0;
            // Iterate through each combination of INPUT_VALUES
            for i in 0..INPUT_VALUES.len() { // Corresponds to `value_idx` in GLSL (X-axis)
                for j in 0..INPUT_VALUES.len() { // Corresponds to `value_idy` in GLSL (Y-axis)
                    

                    let n_m = input_converted[i] * input_converted[j];
                    
                    let bytes: [u8; 16] = unsafe{
                        mem::transmute::<f128, [u8;16]>(n_m)
                    };

                    let f128_form_m: Vec<String> = bytes
                        .iter()
                        .map(|&b| format!("{:02X}", b))
                        .collect();

                    let n_d = input_converted[i] * input_converted[j];

                    let bytes: [u8; 16] = unsafe{
                        mem::transmute::<f128, [u8;16]>(n_d)
                    };

                    let f128_form_d: Vec<String> = bytes
                        .iter()
                        .map(|&b| format!("{:02X}", b))
                        .collect();


                    let abytes: [u8; 16] = unsafe{
                        mem::transmute::<f128, [u8;16]>(input_converted[i])
                    };

                    let af128_form: Vec<String> = abytes
                        .iter()
                        .map(|&b| format!("{:02X}", b))
                        .collect();

                    let bbytes: [u8; 16] = unsafe{
                        mem::transmute::<f128, [u8;16]>(input_converted[j])
                    };

                    let bf128_form: Vec<String> = bbytes
                        .iter()
                        .map(|&b| format!("{:02X}", b))
                        .collect();

                    let mut current_value_bytes_r = Vec::with_capacity(16);
                    let mut current_value_bytes_g = Vec::with_capacity(16);
                
                    // Calculate the starting pixel coordinates (top-left of the 4x4 block)
                    let start_pixel_x = (i * 4) as usize;
                    let start_pixel_y = (j * 4) as usize; // This is the crucial change!

                    for row_offset_in_block in 0..4 { // Iterate through rows within the 4x4 block
                        for col_offset_in_block in 0..4 { // Iterate through columns within the 4x4 block
                            let global_pixel_x = start_pixel_x + col_offset_in_block;
                            let global_pixel_y = start_pixel_y + row_offset_in_block;

                            // Calculate the index in the 'data' (raw pixel buffer)
                            // Each pixel is RGBA (4 bytes), so multiply by 4
                            let pixel_idx = (global_pixel_y * self.texture_width as usize + global_pixel_x) * 4;

                            if pixel_idx + 0 < data.len() {
                                current_value_bytes_r.push(data[pixel_idx + 0]); // Just the red channel
                                current_value_bytes_g.push(data[pixel_idx + 1]);

                            } else {
                                // This indicates an out-of-bounds access, which shouldn't happen
                                // if texture_width/height are correctly calculated.
                                eprintln!("Warning: Attempted to read pixel out of bounds. {} {}", global_pixel_x, global_pixel_y);
                                current_value_bytes_r.push(0xFF); // Push a sentinel value
                                current_value_bytes_g.push(0xFF); // Push a sentinel value
                            }
                        }
                    }

                    let byte_strings_m: Vec<String> = current_value_bytes_r
                        .iter()
                        .map(|&b| format!("{:02X}", b))
                        .collect();
                    
                    let byte_strings_d: Vec<String> = current_value_bytes_g
                        .iter()
                        .map(|&b| format!("{:02X}", b))
                        .collect();    

                    // Assert the expected (Rust calculated) bytes match the found (GPU calculated) bytes
                    
                    if true{
                        if f128_form_m != byte_strings_m {                       
                            println!("FAILED TEST [{}][{}] LITTLE ENDIAN: Mul of [{:?}*{:?}] Expected: {:?}, Found: {:?}",
                                i, j,
                                af128_form, bf128_form,
                                f128_form_m,
                                byte_strings_m
                            );
                            count += 1;
                    
                        }
                        if f128_form_d != byte_strings_d {                       
                            println!("FAILED TEST [{}][{}] LITTLE ENDIAN: Div of [{:?}/{:?}] Expected: {:?}, Found: {:?}",
                                i, j,
                                af128_form, bf128_form,
                                f128_form_d,
                                byte_strings_d
                            );
                            count += 1;
                            
                        }
                    }

                }
            }
            println!("{:} tests failed",count );
        });
    }
}

fn compile_shader(gl: &glow::Context, kind: u32, source: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(kind).unwrap();
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        let log = gl.get_shader_info_log(shader);
        if !log.is_empty() {
            eprintln!("Shader compile log: {}", log);
        }
        if !gl.get_shader_compile_status(shader) {
            Err(log)
        } else {
            Ok(shader)
        }
    }
}

fn link_program(
    gl: &glow::Context,
    vs: glow::Shader,
    fs: glow::Shader,
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program().unwrap();
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);
        let log = gl.get_program_info_log(program);
        if !log.is_empty() {
            eprintln!("Program link log: {}", log);
        }
        if !gl.get_program_link_status(program) {
            Err(log)
        } else {
            Ok(program)
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "f128 Viewer",
        options,
        Box::new(|cc| {
            let gl = cc.gl.as_ref().unwrap().clone();
            Box::new(ShaderApp::new(gl))
        }),
    )
}
