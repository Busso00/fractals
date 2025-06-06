use eframe::{egui, App, glow};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use f128::f128;
use num_traits::ToPrimitive;
use eframe::glow::HasContext;
use std::mem;


// Struct to hold the application's state
struct MandelbrotApp {
    center: (f128, f128), // Current center of the view in fractal space
    zoom: f128,           // Current zoom level
    max_iterations: usize, // Maximum iterations for Mandelbrot calculation
    pub t_render: Arc<Mutex<Duration>>, // Render time, shared safely across threads

    // OpenGL (glow) related fields
    gl_program: Option<glow::Program>,     // Compiled GLSL shader program
    gl_vao: Option<glow::VertexArray>,     // Vertex Array Object (for drawing a quad)
}

// Vertex shader remains unchanged
const VERTEX_SHADER: &str = r#"
    #version 300 es
    
    void main() {
        vec2 positions[4] = vec2[](
            vec2(-1.0, -1.0), // Bottom-left
            vec2( 1.0, -1.0), // Bottom-right
            vec2(-1.0,  1.0), // Top-left
            vec2( 1.0,  1.0)  // Top-right
        );
        gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
    }
"#;

// Fragment shader for the Mandelbrot set with quad-double precision emulation
const FRAGMENT_SHADER: &str = r#"
#version 300 es
precision highp float; // Crucial for precision in QD calculations
precision highp int;   // Needed for uint operations and integer types

// Uniforms passed from Rust
uniform vec2 u_resolution;  // Canvas resolution (width, height) in pixels
uniform uvec4 u_center_re;   // Center real part components (p1, p2, p3, p4) as uints
uniform uvec4 u_center_im;   // Center imag part components (p1, p2, p3, p4) as uints
uniform uvec4 u_scale;       // Zoom scale components (p1, p2, p3, p4) as uints
uniform int u_max_iter;     // Maximum iterations

out vec4 fragColor; // Output color for the fragment

const uint F32_SIGN_MASK = 0x80000000u;
const uint F32_EXP_MASK  = 0x7F800000u;
const uint F32_MANT_MASK = 0x007FFFFFu;

const uint F128_SIGN_MASK = 0x80000000u;
const uint F128_EXP_MASK  = 0x7FFF0000u;
const uint F128_MANT_MASK = 0x0000FFFFu;

const uint F32_EXP_BIAS  = 127u;
const uint F128_EXP_BIAS = 16383u;



// Helper: Add two 32-bit ints with carry
uvec2 two_sum(uint a, uint b) {
    uint sum = a + b;
    uint carry = (sum < a) ? 1u : 0u;
    return uvec2(sum, carry);
}

// Helper: Compare 112-bit mantissas
bool is_mantissa_gte(uvec3 a, uvec3 b) {
    if (a.z > b.z) return true;
    if (a.z < b.z) return false;
    if (a.y > b.y) return true;
    if (a.y < b.y) return false;
    return a.x >= b.x;
}

// Subtracts 112-bit mantissas: a - b (a ≥ b assumed)
uvec3 sub_mantissa(uvec3 a, uvec3 b) {
    uvec3 result;
    uint borrow = 0u;

    // Subtract lowest 32 bits
    if (a.x < b.x) {
        result.x = a.x + (0xFFFFFFFFu - b.x) + 1u;
        borrow = 1u;
    } else {
        result.x = a.x - b.x;
        borrow = 0u;
    }

    // Subtract middle 32 bits with borrow
    uint by = b.y + borrow;
    if (a.y < by) {
        result.y = a.y + (0xFFFFFFFFu - by) + 1u;
        borrow = 1u;
    } else {
        result.y = a.y - by;
        borrow = 0u;
    }

    // Subtract high 48 bits (z: upper part)
    uint bz = b.z + borrow;
    result.z = a.z - bz; // a ≥ b assumed, so no need to check for underflow here

    return result;
}

// norm: shift mantissa left until bit 111 is 1
void norm(inout uvec3 mantissa, inout uint mantissa_hi, inout uint exponent) {
    for (int i = 0; i < 112 && (mantissa_hi & 0x8000u) == 0u; i++) {
        // Shift left 1 bit
        mantissa_hi <<= 1;
        if ((mantissa.z & 0x80000000u) != 0u) mantissa_hi |= 1u;
        mantissa.z <<= 1;
        if ((mantissa.y & 0x80000000u) != 0u) mantissa.z |= 1u;
        mantissa.y <<= 1;
        if ((mantissa.x & 0x80000000u) != 0u) mantissa.y |= 1u;
        mantissa.x <<= 1;

        exponent -= 1u;
    }
}

uvec4 add_f128(uvec4 a, uvec4 b) {
    // === Step 1: Extract components ===
    uint sign_a = a.w & F128_SIGN_MASK;
    uint sign_b = b.w & F128_SIGN_MASK;

    uint exp_a = (a.w & F128_EXP_MASK) >> 16;
    uint exp_b = (b.w & F128_EXP_MASK) >> 16;

    uint mant_a_hi = a.w & F128_MANT_MASK;
    uint mant_b_hi = b.w & F128_MANT_MASK;

    uvec3 mant_a = uvec3(a.x, a.y, a.z);
    uvec3 mant_b = uvec3(b.x, b.y, b.z);

    // === Step 2: Align exponents ===
    if (exp_a > exp_b) {
        int shift = int(exp_a - exp_b);
        for (int i = 0; i < shift; ++i) {
            mant_b.x = (mant_b.x >> 1) | (mant_b.y << 31);
            mant_b.y = (mant_b.y >> 1) | (mant_b.z << 31);
            mant_b.z = (mant_b.z >> 1) | (mant_b_hi << 31);
            mant_b_hi >>= 1;
        }
        exp_b = exp_a;
    } else if (exp_b > exp_a) {
        int shift = int(exp_b - exp_a);
        for (int i = 0; i < shift; ++i) {
            mant_a.x = (mant_a.x >> 1) | (mant_a.y << 31);
            mant_a.y = (mant_a.y >> 1) | (mant_a.z << 31);
            mant_a.z = (mant_a.z >> 1) | (mant_a_hi << 31);
            mant_a_hi >>= 1;
        }
        exp_a = exp_b;
    }

    uint result_sign;
    uvec3 result_mant;
    uint result_mant_hi;

    // === Step 3: Add or subtract mantissas ===
    if (sign_a == sign_b) {
        // Add mantissas
        uvec2 s0 = two_sum(mant_a.x, mant_b.x);
        uvec2 s1 = two_sum(mant_a.y, mant_b.y);
        s1 = two_sum(s1.x, s0.y);
        uvec2 s2 = two_sum(mant_a.z, mant_b.z);
        s2 = two_sum(s2.x, s1.y);
        uvec2 s3 = two_sum(mant_a_hi, mant_b_hi);
        s3 = two_sum(s3.x, s2.y);

        result_mant = uvec3(s0.x, s1.x, s2.x);
        result_mant_hi = s3.x;
        if (s3.y == 1u) {
            // Carry overflow → shift mantissa and bump exponent
            result_mant.x = (result_mant.x >> 1) | (result_mant.y << 31);
            result_mant.y = (result_mant.y >> 1) | (result_mant.z << 31);
            result_mant.z = (result_mant.z >> 1) | (result_mant_hi << 31);
            result_mant_hi = (result_mant_hi >> 1) | (1u << 15);
            exp_a += 1u;
        }
        result_sign = sign_a;
    } else {
        // Subtract smaller from larger
        bool a_is_bigger = is_mantissa_gte(mant_a, mant_b) && mant_a_hi >= mant_b_hi;
        uvec3 bigger = a_is_bigger ? mant_a : mant_b;
        uvec3 smaller = a_is_bigger ? mant_b : mant_a;
        uint hi_bigger = a_is_bigger ? mant_a_hi : mant_b_hi;
        uint hi_smaller = a_is_bigger ? mant_b_hi : mant_a_hi;

        // borrow-based subtraction
        if (hi_bigger < hi_smaller) {
            hi_bigger += 0x10000u;
            bigger.z -= 1u;
        }
        uint hi_result = hi_bigger - hi_smaller;
        uvec3 result = sub_mantissa(bigger, smaller);

        result_mant = result;
        result_mant_hi = hi_result;
        result_sign = a_is_bigger ? sign_a : sign_b;

        // norm
        norm(result_mant, result_mant_hi, exp_a);
    }

    // === Step 4: Repack result ===
    uint w = result_sign | (exp_a << 16) | (result_mant_hi & 0xFFFFu);
    return uvec4(result_mant.x, result_mant.y, result_mant.z, w);
}

uvec4 sub_f128(uvec4 a, uvec4 b) {
    
    uvec4 neg_b = add_f128(uvec4(~b.x, ~b.y, ~b.z, ~b.w), uvec4(1u, 0u, 0u, 0u));
    return add_f128(a, neg_b);
}

bool is_zero(uvec4 f) {
    return (f.x | f.y | f.z | (f.w & F128_MANT_MASK)) == 0u && ((f.w & F128_EXP_MASK) >> 16) == 0u;
}

bool is_inf_or_nan(uvec4 f) {
    return ((f.w & F128_EXP_MASK) >> 16) == 0x7FFFu;
}

uvec4 zero(uint sign) {
    return uvec4(0u, 0u, 0u, sign); // exp = 0, mant = 0
}

uvec4 inf(uint sign) {
    return uvec4(0u, 0u, 0u, sign | 0x7FFF0000u);
}

uvec4 nan() {
    return uvec4(0u, 0u, 0u, 0x7FFF0001u); // Quiet NaN
}

// Multiplies two uints and returns result as [low, high]
uvec2 umul32(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16u;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16u;

    uint p0 = a_lo * b_lo;                   // 16b × 16b = 32b
    uint p1 = a_lo * b_hi;                   // 16b × 16b
    uint p2 = a_hi * b_lo;                   // 16b × 16b
    uint p3 = a_hi * b_hi;                   // 16b × 16b

    uint mid1 = p1 + (p0 >> 16u);            // carry into upper 32 bits
    uint carry1 = (mid1 < p1) ? 1u : 0u;

    uint mid2 = mid1 + p2;
    uint carry2 = (mid2 < mid1) ? 1u : 0u;

    uint high = p3 + (mid2 >> 16u) + carry1 + carry2;
    uint low = (mid2 << 16u) | (p0 & 0xFFFFu);

    return uvec2(low, high);
}

// Multiply two 32-bit unsigned integers to get a 64-bit result split into two 32-bit parts.
void umul32x32(uint a, uint b, out uint low, out uint high) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16;

    uint lo_lo = a_lo * b_lo;
    uint lo_hi = a_lo * b_hi;
    uint hi_lo = a_hi * b_lo;
    uint hi_hi = a_hi * b_hi;

    uint mid1 = lo_hi + ((lo_lo >> 16) & 0xFFFFu);
    uint mid2 = hi_lo + (mid1 & 0xFFFFu);

    high = hi_hi + (mid1 >> 16) + (mid2 >> 16);
    low = (mid2 << 16) | (lo_lo & 0xFFFFu);
}

// Performs 112-bit x 112-bit multiplication = 224-bit result.
// a, b are 4×uint (112-bit), result is 7×uint (224-bit)
void multiply_112(uvec4 a, uvec4 b, out uint result[7]) {
    for (int i = 0; i < 7; ++i) {
        result[i] = 0u;
    }

    for (int i = 0; i < 4; ++i) {
        uint carry = 0u;
        for (int j = 0; j < 4; ++j) {
            int k = i + j;
            if (k >= 7) continue;

            uint lo, hi;
            umul32x32(a[i], b[j], lo, hi);

            // Add low part
            uint temp = result[k];
            result[k] += lo;
            if (result[k] < temp) hi += 1u;

            // Add high part
            if (k + 1 < 7) {
                uint t = result[k + 1];
                result[k + 1] += hi;
                if (result[k + 1] < t) {
                    // propagate carry
                    for (int m = k + 2; m < 7; ++m) {
                        result[m] += 1u;
                        if (result[m] != 0u) break;
                    }
                }
            }
        }
    }
}


void norm_224(inout uint[7] val, out uvec3 mant, out uint hi16, out int shift) {
    shift = 0;
    while ((val[6] & 0x8000u) == 0u && shift < 112) {
        // Shift left 1 bit across all 7 words
        for (int i = 6; i > 0; --i) {
            val[i] = (val[i] << 1) | (val[i - 1] >> 31);
        }
        val[0] <<= 1;
        shift += 1;
    }

    mant = uvec3(val[0], val[1], val[2]);
    hi16 = val[3] & 0xFFFFu; // top 16 bits
}


uvec4 mul_f128(uvec4 a, uvec4 b) {
    // === Handle edge cases ===
    if (is_inf_or_nan(a) || is_inf_or_nan(b)) {
        if (is_zero(a) || is_zero(b)) return nan(); // 0 * inf = NaN
        return (is_inf_or_nan(a) ? a : b);
    }
    if (is_zero(a) || is_zero(b)) return zero((a.w ^ b.w) & F128_SIGN_MASK);

    // === Extract sign, exponent, mantissas ===
    uint sign = (a.w ^ b.w) & F128_SIGN_MASK;

    uint exp_a = (a.w & F128_EXP_MASK) >> 16;
    uint exp_b = (b.w & F128_EXP_MASK) >> 16;

    uint exp_result = exp_a + exp_b - F128_EXP_BIAS;

    uvec4 mant_a = uvec4(a.x, a.y, a.z, a.w & F128_MANT_MASK);
    uvec4 mant_b = uvec4(b.x, b.y, b.z, b.w & F128_MANT_MASK);

    // === Multiply 112-bit mantissas ===
    uint result[7];
    multiply_112(mant_a, mant_b, result);

    // === Normalize result ===
    uvec3 mantissa;
    uint hi16;
    int shift;
    norm_224(result, mantissa, hi16, shift);
    exp_result -= uint(shift);

    // === Handle underflow/overflow ===
    if (exp_result >= 0x7FFFu) return inf(sign); // overflow
    if (exp_result == 0u) return zero(sign);     // underflow

    // === Reassemble result ===
    uint w = sign | (exp_result << 16) | (hi16 & 0xFFFFu);
    return uvec4(mantissa.x, mantissa.y, mantissa.z, w);
}


// Converts float to custom f128 (uvec4)

uvec4 float_to_f128(float f) {
    uint raw = floatBitsToUint(f);

    uint sign     = raw & F32_SIGN_MASK;
    uint exponent = (raw & F32_EXP_MASK) >> 23;
    uint mantissa = raw & F32_MANT_MASK;

    // Special cases
    if (exponent == 0xFFu) {
        if (mantissa != 0u)
            return uvec4(0u, 0u, 0u, 0x7FFF0001u); // Quiet NaN
        return uvec4(0u, 0u, 0u, sign | 0x7FFF0000u); // Inf
    }

    if (exponent == 0u && mantissa == 0u) {
        return uvec4(0u, 0u, 0u, sign); // Zero
    }

    // norm exponent
    uint exp_f128;
    uint m24;

    if (exponent == 0u) {
        // Subnormal
        exp_f128 = F128_EXP_BIAS - (F32_EXP_BIAS - 1u);
        m24 = mantissa; // No implicit bit
    } else {
        // normd: Add implicit leading 1
        exp_f128 = exponent - F32_EXP_BIAS + F128_EXP_BIAS;
        m24 = mantissa | (1u << 23); // Add implicit 1
    }

    // Convert 24-bit mantissa to 112-bit (shift left by 88 bits)
    // Result goes into uvec4: [x, y, z, upper 16 bits in w]
    uvec4 mantissa_128 = uvec4(0u);

    // Shift 24-bit m24 by 88 bits: that is 2 full words (64 bits), +24
    // So:
    // - mantissa.x = 0
    // - mantissa.y = 0
    // - mantissa.z = m24 << 24
    // - mantissa.w upper 16 bits = m24 >> 8
    mantissa_128.x = 0u;
    mantissa_128.y = 0u;
    mantissa_128.z = m24 << 24;
    mantissa_128.w = (m24 >> 8) & 0xFFFFu;

    // Combine with sign and exponent into the upper 32 bits of w
    mantissa_128.w |= (exp_f128 << 16);
    mantissa_128.w |= sign;

    return mantissa_128;
}


void extract_f128(uvec4 a, out uint sign, out uint exp, out uint[4] mant) {
    sign = a.w & F128_SIGN_MASK;
    exp = (a.w & F128_EXP_MASK) >> 16;

    mant[0] = a.x;
    mant[1] = a.y;
    mant[2] = a.z;
    mant[3] = a.w & F128_MANT_MASK;

    if (exp != 0u) {
        mant[3] |= 0x00010000u; // Add implicit leading 1 to top 16 bits
    }
}

// Divide 112-bit a by 112-bit b, return 112-bit result in uvec3 + top16
void divide_112(uint[4] a, uint[4] b, out uint[4] quotient) {
    
    // Placeholder: fill with your integer division logic
    for (int i = 0; i < 4; ++i) quotient[i] = 0u;

    quotient[3] = 0x0001u; // force leading 1 for now (must norm properly)
}

void norm_112(inout uint[4] mant, inout uint exp) {
    // Shift mantissa left until highest bit is in bit 111
    while ((mant[3] & 0x00010000u) == 0u && exp > 0u) {
        // shift left 1 bit
        for (int i = 3; i > 0; --i) {
            mant[i] = (mant[i] << 1) | (mant[i - 1] >> 31);
        }
        mant[0] <<= 1;
        exp -= 1u;
    }
}

uvec4 compose_f128(uint sign, uint exp, uint[4] mant) {
    uint w = sign | (exp << 16) | (mant[3] & 0xFFFFu);
    return uvec4(mant[0], mant[1], mant[2], w);
}

uvec4 div_f128(uvec4 a, uvec4 b) {
    // Handle special cases
    if (is_zero(b)) {
        if (is_zero(a)) return nan();
        return inf((a.w ^ b.w) & F128_SIGN_MASK);
    }

    if (is_zero(a)) return zero((a.w ^ b.w) & F128_SIGN_MASK);
    if (is_inf_or_nan(a) || is_inf_or_nan(b)) {
        if (is_inf_or_nan(a) && is_inf_or_nan(b)) return nan();
        if (is_inf_or_nan(a)) return inf((a.w ^ b.w) & F128_SIGN_MASK);
        return zero((a.w ^ b.w) & F128_SIGN_MASK);
    }

    // Extract
    uint sign_a, exp_a, sign_b, exp_b;
    uint[4] mant_a, mant_b;

    extract_f128(a, sign_a, exp_a, mant_a);
    extract_f128(b, sign_b, exp_b, mant_b);

    uint sign = (sign_a ^ sign_b) & F128_SIGN_MASK;
    uint exp_result = exp_a - exp_b + F128_EXP_BIAS;

    // Divide mantissas
    uint[4] mant_result;
    divide_112(mant_a, mant_b, mant_result);

    // norm
    norm_112(mant_result, exp_result);

    // Handle overflow/underflow
    if (exp_result >= 0x7FFFu) return inf(sign);
    if (exp_result == 0u) return zero(sign);

    return compose_f128(sign, exp_result, mant_result);
}

bool f128_gt(uvec4 a, uvec4 b) {
    if (a.w != b.w) return a.w > b.w;
    if (a.z != b.z) return a.z > b.z;
    if (a.y != b.y) return a.y > b.y;
    return a.x > b.x;
}

int mandelbrot_qd(uvec4 cre, uvec4 cim, int max_iter) {
    uvec4 zr = zero(0u);
    uvec4 zi = zero(0u);
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        uvec4 zr2 = mul_f128(zr, zr);
        uvec4 zi2 = mul_f128(zi, zi);
        uvec4 mag2 = add_f128(zr2, zi2);

        if (f128_gt(mag2,float_to_f128(4.0))) { 
            break; 
        }

        uvec4 temp_zr = sub_f128(zr2, zi2);
        uvec4 zr_new = add_f128(temp_zr, cre);

        uvec4 zr_zi = mul_f128(zr, zi);
        uvec4 two_zr_zi = add_f128(zr_zi, zr_zi);
        uvec4 zi_new = add_f128(two_zr_zi, cim);

        zr = zr_new;
        zi = zi_new;
    }

    return iter;
}


void main() {
    vec2 ndc = gl_FragCoord.xy / u_resolution.xy; // normd Device Coordinates [0, 1]
    
    uvec4 scale = uvec4(u_scale.x, u_scale.y, u_scale.z, u_scale.w);
    uvec4 center_re = uvec4(u_center_re.x, u_center_re.y, u_center_re.z, u_center_re.w);
    uvec4 center_im = uvec4(u_center_im.x, u_center_im.y, u_center_im.z, u_center_im.w);


    uvec4 pixel_re_offset = mul_f128(float_to_f128(ndc.x * 2.0 - 1.0), div_f128(float_to_f128(2.0), scale));
    uvec4 pixel_im_offset = mul_f128(float_to_f128(ndc.y * 2.0 - 1.0), div_f128(float_to_f128(2.0), scale));
    
    // Adjust Y-axis for typical Mandelbrot mapping (positive Y goes up, so subtract)
    uvec4 cre = add_f128(center_re, pixel_re_offset);
    uvec4 cim = sub_f128(center_im, pixel_im_offset);

    int iter = mandelbrot_qd(cre, cim, u_max_iter);
        
    if (iter == u_max_iter) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0); // Inside set (black)
    } else {
        // Smooth coloring (based on continuous iteration count algorithm for a smoother look)
        float t = float(iter) / float(u_max_iter);
        
        float pi2 = 6.28318530718; // 2 * PI

        // Simple sinusoidal coloring
        float r = 0.5 + 0.5 * cos(t * pi2);
        float g = 0.5 + 0.5 * cos(t * pi2 + pi2 / 3.0);
        float b = 0.5 + 0.5 * cos(t * pi2 + 2.0 * pi2 / 3.0);
            
        fragColor = vec4(r, g, b, 1.0);
    }
}
"#;


impl MandelbrotApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("You need to enable GL with EGL_ENABLE_GL=true for egui native");
        let gl = Arc::clone(gl);

        let gl_program = unsafe { Self::compile_shader(&gl) }.unwrap();
        let gl_vao = unsafe { gl.create_vertex_array() }.unwrap();

        Self {
            center: (f128::from(-0.5), f128::from(0.0)),
            zoom: f128::from(1.0),
            max_iterations: 2048,
            t_render: Arc::new(Mutex::new(Duration::ZERO)),
            gl_program: Some(gl_program),
            gl_vao: Some(gl_vao),
        }
    }

    unsafe fn compile_shader(gl: &glow::Context) -> Result<glow::Program, String> {
        let program = gl.create_program()?;
        let shader_sources = [
            (glow::VERTEX_SHADER, VERTEX_SHADER),
            (glow::FRAGMENT_SHADER, FRAGMENT_SHADER),
        ];

        let mut shaders = Vec::with_capacity(shader_sources.len());
        for (shader_type, shader_source) in shader_sources.iter() {
            let shader = gl.create_shader(*shader_type)?;
            gl.shader_source(shader, shader_source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                return Err(gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(gl.get_program_info_log(program));
        }

        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        Ok(program)
    }

    /// Splits an f128 into four f32 components, representing decreasing levels of precision.
    ///
    /// This function attempts to capture the "remainder" of the f128 value
    /// after successive conversions to lower precision. This is a conceptual
    /// split and its utility depends heavily on how these four f32s would be
    /// interpreted and used in your GLSL shader.
    ///
    /// Be aware that the precision of f32 is limited, and a significant portion
    /// of the original f128 precision will be lost even with this method.
    fn split_f128_to_4_u32(value: f128) -> (u32, u32, u32, u32) {

        let bytes: [u8; 16] = unsafe { mem::transmute(value) };

        let p1 = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let p2 = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let p3 = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let p4 = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

    
        (p1, p2, p3, p4)
    }

    unsafe fn draw_mandelbrot_gl(
        painter: &egui_glow::Painter,
        program_id: glow::Program,
        vao_id: glow::VertexArray,
        clip_rect_width: f32,
        clip_rect_height: f32,
        center: (f128, f128),
        zoom: f128,
        max_iterations: usize,
    ) {
        let gl = painter.gl(); 

        gl.use_program(Some(program_id));
        gl.bind_vertex_array(Some(vao_id));

        gl.uniform_2_f32(
            gl.get_uniform_location(program_id, "u_resolution").as_ref(),
            clip_rect_width,
            clip_rect_height,
        );
        
        // Split center components into four f32s and pass them
        let (cx1, cx2, cx3, cx4) = Self::split_f128_to_4_u32(center.0);
        let (cy1, cy2, cy3, cy4) = Self::split_f128_to_4_u32(center.1);

        gl.uniform_4_u32(
            gl.get_uniform_location(program_id, "u_center_re").as_ref(),
            cx1, cx2, cx3, cx4, // ERROR: Using cy2 instead of cx2 here. Should be cx1, cx2, cx3, cx4
        );
        gl.uniform_4_u32(
            gl.get_uniform_location(program_id, "u_center_im").as_ref(),
            cy1, cy2, cy3, cy4,
        );

        // Split zoom into four f32s and pass them
        let (z1, z2, z3, z4) = Self::split_f128_to_4_u32(zoom);
        gl.uniform_4_u32(
            gl.get_uniform_location(program_id, "u_scale").as_ref(),
            z1, z2, z3, z4,
        );
        
        gl.uniform_1_i32(
            gl.get_uniform_location(program_id, "u_max_iter").as_ref(),
            max_iterations as i32,
        );

        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.clear(glow::COLOR_BUFFER_BIT);
        
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

        gl.bind_vertex_array(None);
        gl.use_program(None);
    }
}

impl App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        println!("DEBUG: update() is being called!");

        ctx.request_repaint();

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = ui.available_size().min(egui::vec2(512.0, 512.0));
            println!("DEBUG: Allocated available_size: {:?}", available_size);

            // --- UI CONTROLS ---
            ui.horizontal(|ui| {
                ui.label("Iterations:");
                let old_iter = self.max_iterations;
                ui.add(egui::Slider::new(&mut self.max_iterations, 128..=4096).logarithmic(true));
                if old_iter != self.max_iterations {
                    ctx.request_repaint();
                }

                ui.separator();

                // Display zoom and center with f128 formatting
                ui.label(format!("Zoom: {:.2}x", self.zoom.to_f64().unwrap_or(0.0)));
                ui.label(format!("Center: ({:.6}, {:.6})", self.center.0.to_f64().unwrap_or(0.0), self.center.1.to_f64().unwrap_or(0.0)));


                if ui.button("Reset View").clicked() {
                    self.center = (f128::from(-0.5), f128::from(0.0)); // Reset to f128
                    self.zoom = f128::from(1.0);            // Reset to f128
                    ctx.request_repaint();
                }
            });

            ui.label(format!("Render time: {:.1} ms", self.t_render.lock().unwrap().as_secs_f64() * 1000.0));

            // --- Interaction Logic (Pan/Zoom) ---
            let mut need_redraw = false;

            // Pan speed calculated with f128
            let pan_speed = f128::from(0.05) / self.zoom;
            ctx.input(|i| {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    self.center.0 -= pan_speed;
                    need_redraw = true;
                }
                if i.key_pressed(egui::Key::ArrowRight) {
                    self.center.0 += pan_speed;
                    need_redraw = true;
                }
                if i.key_pressed(egui::Key::ArrowUp) {
                    self.center.1 -= pan_speed;
                    need_redraw = true;
                }
                if i.key_pressed(egui::Key::ArrowDown) {
                    self.center.1 += pan_speed; // Inverted for consistency with previous request
                    need_redraw = true;
                }
            });

            let (response, painter) = ui.allocate_painter(available_size, egui::Sense::drag());

            let is_hovered = response.hovered();
            let scroll_delta = ctx.input(|i| i.raw_scroll_delta);
            let scroll_delta_y = scroll_delta.y;
            let drag = response.drag_delta();
            let drag_x = drag.x;
            let drag_y = drag.y;

            if scroll_delta_y != 0.0 {
                if let Some(pointer_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                    if is_hovered {
                        // Calculate position in fractal space before zoom with f128
                        let rel_x = f128::from((pointer_pos.x - response.rect.left()) / response.rect.width());
                        let rel_y = f128::from((pointer_pos.y - response.rect.top()) / response.rect.height());

                        let scale = f128::from(4.0) / self.zoom; // Current view width in fractal units (f128)

                        // Mouse position in fractal coordinates
                        let mouse_re = self.center.0 + (rel_x - f128::from(0.5)) * scale;
                        let mouse_im = self.center.1 + (rel_y - f128::from(0.5)) * scale;

                        // Apply zoom
                        let zoom_delta = f128::from(1.1f64.powf(-scroll_delta_y as f64 * 0.1)); // Invert scroll, use f128::from
                        self.zoom *= zoom_delta;

                        // Adjust center to keep mouse position fixed on the same fractal point
                        let new_scale = f128::from(4.0) / self.zoom;
                        self.center.0 = mouse_re - (rel_x - f128::from(0.5)) * new_scale;
                        self.center.1 = mouse_im - (rel_y - f128::from(0.5)) * new_scale;

                        need_redraw = true;
                    }
                }
            }
            else if response.dragged() {
                if is_hovered {
                    let scale_factor = f128::from(4.0) / self.zoom; // Total width in fractal units for current zoom (f128)
                    let dx = f128::from(drag_x) / f128::from(response.rect.width());
                    let dy = f128::from(drag_y) / f128::from(response.rect.height());

                    self.center.0 -= dx * scale_factor;
                    self.center.1 += dy * scale_factor; // Inverted Y drag
                    need_redraw = true;
                }
            }

            if need_redraw {
                ctx.request_repaint();
            }

            // --- GL Paint Callback Setup ---
            // These clones are fine as they are primitive types (f32)
            let center_clone = self.center;
            let zoom_clone = self.zoom;
            let max_iterations_clone = self.max_iterations;

            let t_render_arc_for_closure = Arc::clone(&self.t_render);

            let gl_program_id = self.gl_program.expect("GL program not initialized");
            let gl_vao_id = self.gl_vao.expect("GL VAO not initialized");

            println!("DEBUG: GL resources for callback: Program={:?}, VAO={:?}", gl_program_id, gl_vao_id);

            painter.add(egui::PaintCallback {
                rect: response.rect,
                callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter_gl| {
                    println!("DEBUG: PaintCallback is being executed!");

                    let start_render_time = Instant::now();

                    unsafe {
                        let x = info.clip_rect.left() as i32;
                        let y = (info.screen_size_px[1] as f32 - info.clip_rect.bottom()) as i32;
                        let width = info.clip_rect.width() as i32;
                        let height = info.clip_rect.height() as i32;

                        painter_gl.gl().viewport(x, y, width, height);

                        MandelbrotApp::draw_mandelbrot_gl(
                            painter_gl,
                            gl_program_id,
                            gl_vao_id,
                            info.clip_rect.width(),
                            info.clip_rect.height(),
                            center_clone,
                            zoom_clone,
                            max_iterations_clone,
                        );
                    }

                    *t_render_arc_for_closure.lock().unwrap() = start_render_time.elapsed();
                })),
            });
        });
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            unsafe {
                if let Some(program) = self.gl_program {
                    gl.delete_program(program);
                }
                if let Some(vao) = self.gl_vao {
                    gl.delete_vertex_array(vao);
                }
            }
        }
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(540.0, 600.0)),
        multisampling: 8,
        vsync: true,
        depth_buffer: 0,
        stencil_buffer: 0,
        #[cfg(target_arch = "wasm32")]
        renderer: eframe::Renderer::Glow,
        #[cfg(not(target_arch = "wasm32"))]
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "Mandelbrot Explorer (f128 via vec4)", // Updated title to reflect precision
        native_options,
        Box::new(|cc| {
            if cc.gl.is_none() {
                eprintln!("GL context not available! Make sure to enable it. E.g., for wgpu backend, set RUST_BACKBACKTRACE=1 for more details or try to enable glow backend explicitly.");
            }
            Box::new(MandelbrotApp::new(cc))
        }),
    )
}