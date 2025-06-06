use eframe::{egui, App, glow};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use eframe::glow::HasContext;
use f128::f128;
use std::mem;

// Struct to hold the application's state
struct MandelbrotApp {
    center: (f128, f128),           // Current center of the view in fractal space
    zoom: f128,                    // Current zoom level
    max_iterations: usize,        // Maximum iterations for Mandelbrot calculation
    pub t_render: Arc<Mutex<Duration>>, // Render time, shared safely across threads
    texture: Option<egui::TextureHandle>,
    texture_size: (usize, usize), // Current texture dimensions
    needs_update: bool,           // Flag to indicate texture needs updating
    
    // OpenGL resources
    gl: Arc<glow::Context>,
    gl_program: Option<glow::Program>,
    gl_vao: Option<glow::VertexArray>,
    gl_framebuffer: Option<glow::Framebuffer>,
    gl_texture: Option<glow::Texture>,
    initialized: bool,            // Track initialization state
}

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
precision highp float;
precision highp int;



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


uniform uvec4 u_offset_x[512];
uniform uvec4 u_offset_y[512];

uniform int u_max_iter;    // Maximum iterations
uniform uvec4 u_center_re;   // Center real part components (p1, p2, p3, p4) as uints
uniform uvec4 u_center_im;   // Center imag part components (p1, p2, p3, p4) as uints

out vec4 fragColor;


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
    sub32(a.y, add_f128
add_f128
add_f128
add_f128
add_f128b.y, borrow, result.y, borrow);
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


bool gt_127(uvec4 a, uvec4 b) {
    uint w_a = a.w & (~F128_SIGN_MASK);
    uint w_b = b.w & (~F128_SIGN_MASK);
    if (w_a > w_b) return true;
    if (w_b > w_a) return false;
    if (a.z > b.z) return true;
    if (a.z < b.z) return false;
    if (a.y > b.y) return true;
    if (a.y < b.y) return false;
    return a.x > b.x;
}


bool gt_f128(uvec4 a, uvec4 b){
    uint sign_a = a.w & F128_SIGN_MASK;
    uint sign_b = b.w & F128_SIGN_MASK;

    if (sign_a > sign_b){
        return false;
    }else if (sign_b > sign_a){
        return true;
    }else{
        if (all(equal(a,b))){
            return false;
        }
        bool res = gt_127(a, b);
        if (sign_a > 0u){
            return !res;
        }
        else{
            return res;
        }
    }
}


int mandelbrot_qd(uvec4 cre, uvec4 cim, int max_iter) {
    uvec4 zr = zero(0u);
    uvec4 zi = zero(0u);
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        uvec4 zr2 = mul_f128(zr, zr);
        uvec4 zi2 = mul_f128(zi, zi);
        uvec4 mag2 = add_f128(zr2, zi2);

        if (gt_f128(mag2,float_to_f128(4.0))) { 
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
    
    uvec4 pixel_re_offset = u_offset_x[uint(gl_FragCoord.xy.x)];
    uvec4 pixel_im_offset = u_offset_y[uint(gl_FragCoord.xy.y)];

    uvec4 cre = add_f128(u_center_re, pixel_re_offset);
    uvec4 cim = sub_f128(u_center_im, pixel_im_offset);

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
        let gl = cc.gl.as_ref().expect("You need to enable GL for this application");
        let gl = Arc::clone(gl);

        Self {
            center: (f128::from(-0.5), f128::from(0.0)),
            zoom: f128::from(1.0),
            max_iterations: 128,
            t_render: Arc::new(Mutex::new(Duration::ZERO)),
            texture: None,
            texture_size: (0, 0),
            needs_update: true,
            gl,
            gl_program: None,
            gl_vao: None,
            gl_framebuffer: None,
            gl_texture: None,
            initialized: false,
        }
    }

    // Initialize OpenGL resources safely
    fn initialize_gl(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        unsafe {
            // Compile shader program
            let program = Self::compile_shader(&self.gl)?;
            self.gl_program = Some(program);

            // Create VAO - check if extension is supported first
            let vao = self.gl.create_vertex_array()
                .map_err(|e| format!("Failed to create VAO: {}", e))?;
            self.gl_vao = Some(vao);

            self.initialized = true;
            Ok(())
        }
    }

    unsafe fn compile_shader(gl: &glow::Context) -> Result<glow::Program, String> {
        let program = gl.create_program()
            .map_err(|e| format!("Failed to create program: {}", e))?;
        
        let shader_sources = [
            (glow::VERTEX_SHADER, VERTEX_SHADER),
            (glow::FRAGMENT_SHADER, FRAGMENT_SHADER),
        ];

        let mut shaders = Vec::with_capacity(shader_sources.len());
        for (shader_type, shader_source) in shader_sources.iter() {
            let shader = gl.create_shader(*shader_type)
                .map_err(|e| format!("Failed to create shader: {}", e))?;
            
            gl.shader_source(shader, shader_source);
            gl.compile_shader(shader);
            
            if !gl.get_shader_compile_status(shader) {
                let error = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                return Err(format!("Shader compilation failed: {}", error));
            }
            
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let error = gl.get_program_info_log(program);
            // Clean up shaders
            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            gl.delete_program(program);
            return Err(format!("Program linking failed: {}", error));
        }

        // Clean up shaders (they're now part of the program)
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        
        Ok(program)
    }

    unsafe fn setup_framebuffer(&mut self, width: i32, height: i32) -> Result<(), String> {
        // Validate dimensions
        if width <= 0 || height <= 0 || width > 8192 || height > 8192 {
            return Err(format!("Invalid framebuffer dimensions: {}x{}", width, height));
        }

        // Clean up existing resources
        self.cleanup_framebuffer();

        // Create texture for rendering
        let texture = self.gl.create_texture()
            .map_err(|e| format!("Failed to create texture: {}", e))?;
        
        self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        
        // Check for OpenGL errors after texture creation
        let error = self.gl.get_error();
        if error != glow::NO_ERROR {
            self.gl.delete_texture(texture);
            return Err(format!("OpenGL error after texture creation: {}", error));
        }

        self.gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            width,
            height,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            eframe::glow::PixelUnpackData::Slice(None), // Use None instead of empty slice
        );

        // Check for errors after tex_image_2d
        let error = self.gl.get_error();
        if error != glow::NO_ERROR {
            self.gl.delete_texture(texture);
            return Err(format!("OpenGL error after tex_image_2d: {}", error));
        }

        self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

        // Create framebuffer
        let framebuffer = self.gl.create_framebuffer()
            .map_err(|e| format!("Failed to create framebuffer: {}", e))?;
        
        self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        self.gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );

        // Check framebuffer completeness
        let status = self.gl.check_framebuffer_status(glow::FRAMEBUFFER);
        if status != glow::FRAMEBUFFER_COMPLETE {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.delete_texture(texture);
            self.gl.delete_framebuffer(framebuffer);
            return Err(format!("Framebuffer not complete: {}", status));
        }

        // Restore default framebuffer
        self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        self.gl.bind_texture(glow::TEXTURE_2D, None);

        self.gl_framebuffer = Some(framebuffer);
        self.gl_texture = Some(texture);

        Ok(())
    }

    unsafe fn cleanup_framebuffer(&mut self) {
        if let Some(fb) = self.gl_framebuffer.take() {
            self.gl.delete_framebuffer(fb);
        }
        if let Some(tex) = self.gl_texture.take() {
            self.gl.delete_texture(tex);
        }
    }

    fn render_to_texture(&mut self, width: usize, height: usize) -> Result<egui::ColorImage, String> {
        // Ensure GL is initialized
        if !self.initialized {
            self.initialize_gl()?;
        }

        let start_time = Instant::now();

        unsafe {
            // Setup framebuffer if needed
            if self.gl_framebuffer.is_none() || self.texture_size != (width, height) {
                self.setup_framebuffer(width as i32, height as i32)?;
                self.texture_size = (width, height);
            }

            let program = self.gl_program.ok_or("Shader program not initialized")?;
            let vao = self.gl_vao.ok_or("VAO not initialized")?;
            let framebuffer = self.gl_framebuffer.ok_or("Framebuffer not initialized")?;

            // Save current GL state
            let mut prev_viewport = [0i32; 4];
            self.gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);

            // Bind framebuffer and setup viewport
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            self.gl.viewport(0, 0, width as i32, height as i32);

            // Use shader program
            self.gl.use_program(Some(program));
            self.gl.bind_vertex_array(Some(vao));
    

        
            // Split center components into four f32s and pass them
            let [cx1, cx2, cx3, cx4] = mem::transmute(self.center.0);
            let [cy1, cy2, cy3, cy4] = mem::transmute(self.center.1);
            
            self.gl.uniform_4_u32(
                self.gl.get_uniform_location(program, "u_center_re").as_ref(),
                cx1, cx2, cx3, cx4,
            );
            self.gl.uniform_4_u32(
                self.gl.get_uniform_location(program, "u_center_im").as_ref(),
                cy1, cy2, cy3, cy4,
            );

            self.gl.uniform_1_i32(
                self.gl.get_uniform_location(program, "u_max_iter").as_ref(),
                self.max_iterations as i32,
            );


            let mut off_x: [f128; 512] = [f128::from(0.0); 512];
            for x in 0..512{
                let eff_x = f128::from(x)/f128::from(512.0);
                off_x[x] = (eff_x * f128::from(2.0) - f128::from(1.0)) * f128::from(2.0) / self.zoom;
            }

            let loc_values_x = self.gl.get_uniform_location(program, "u_offset_x")
                .ok_or_else(|| "Could not find uniform location 'u_f128_values'".to_string()).unwrap();
            let flat_values_x: &[u32] = std::slice::from_raw_parts(off_x.as_ptr() as *const u32,off_x.len() * 4);
            self.gl.uniform_4_u32_slice(Some(&loc_values_x), flat_values_x);

            let mut off_y: [f128; 512] = [f128::from(0.0); 512];
            for y in 0..512{
                let eff_y = f128::from(y)/f128::from(512.0);
                off_y[y] = (eff_y * f128::from(2.0) - f128::from(1.0)) * f128::from(2.0) / self.zoom;
            }
            
            let loc_values_y = self.gl.get_uniform_location(program, "u_offset_y")
                .ok_or_else(|| "Could not find uniform location 'u_f128_values'".to_string()).unwrap();
            let flat_values_y: &[u32] = std::slice::from_raw_parts(off_y.as_ptr() as *const u32,off_y.len() * 4);
            self.gl.uniform_4_u32_slice(Some(&loc_values_y), flat_values_y);
            

            // Clear and render
            self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // Check for OpenGL errors
            let error = self.gl.get_error();
            if error != glow::NO_ERROR {
                eprintln!("OpenGL error during rendering: {}", error);
            }

            // Read pixels from framebuffer
            let mut pixels = vec![0u8; width * height * 4];
            self.gl.read_pixels(
                0, 0,
                width as i32, height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );

            // Restore previous GL state - simplified approach
            self.gl.bind_vertex_array(None);
            self.gl.use_program(None);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.viewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);

            // Convert to egui ColorImage (flip Y axis)
            let mut egui_pixels = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    // Flip Y coordinate
                    let flipped_y = height - 1 - y;
                    let idx = (flipped_y * width + x) * 4;
                    egui_pixels.push(egui::Color32::from_rgba_premultiplied(
                        pixels[idx],
                        pixels[idx + 1],
                        pixels[idx + 2],
                        pixels[idx + 3],
                    ));
                }
            }

            *self.t_render.lock().unwrap() = start_time.elapsed();

            Ok(egui::ColorImage {
                size: [width, height],
                pixels: egui_pixels,
            })
        }
    }
}

impl App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // --- UI CONTROLS ---
            ui.horizontal(|ui| {
                ui.label("Iterations:");
                let old_iter = self.max_iterations;
                ui.add(egui::Slider::new(&mut self.max_iterations, 128..=4096).logarithmic(true));
                if old_iter != self.max_iterations {
                    self.needs_update = true;
                }

                ui.separator();

                ui.label(format!("Zoom: {:.2}x", self.zoom));
                ui.label(format!("Center: ({:.6}, {:.6})", self.center.0, self.center.1));

                if ui.button("Reset View").clicked() {
                    self.center = (f128::from(-0.5), f128::from(0.0));
                    self.zoom = f128::from(1.0);
                    self.needs_update = true;
                }
            });

            ui.label(format!("Render time: {:.1} ms", 
                self.t_render.lock().unwrap().as_secs_f64() * 1000.0));

            ui.separator();

            // --- FRACTAL DISPLAY ---
            let size = egui::Vec2::new(512.0, 512.0);
            
            let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
            
            let width = size.x as usize;
            let height = size.y as usize;

            // Update texture if needed
            if self.needs_update || self.texture.is_none() || self.texture_size != (width, height) {
                match self.render_to_texture(width, height) {
                    Ok(image) => {
                        self.texture = Some(ctx.load_texture("mandelbrot", image, Default::default()));
                        self.needs_update = false;
                    }
                    Err(e) => {
                        eprintln!("Failed to render fractal: {}", e);
                        // Show error in UI instead of crashing
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(rect), |ui| {
                            ui.centered_and_justified(|ui| {
                                ui.label(format!("Render Error: {}", e));
                            });
                        });
                        return; // Early return to avoid further processing
                    }
                }
            }

            // Draw the texture
            if let Some(texture) = &self.texture {
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(rect), |ui| {
                    ui.image(texture);
                });
            }

            // --- INTERACTION HANDLING ---
            let mut need_redraw = false;

            // Keyboard controls
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
                    self.center.1 += pan_speed;
                    need_redraw = true;
                }
            });

            // Mouse zoom
            let scroll_delta = ctx.input(|i| i.raw_scroll_delta.y);
            if scroll_delta != 0.0 {
                if let Some(pointer_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                    let rel_x = (pointer_pos.x - rect.left()) / rect.width();
                    let rel_y = (pointer_pos.y - rect.top()) / rect.height();

                    let scale = f128::from(4.0) / self.zoom;
                    let mouse_re = self.center.0 + f128::from(rel_x - 0.5) * scale;
                    let mouse_im = self.center.1 + f128::from(rel_y - 0.5) * scale;

                    let zoom_delta = f128::from(1.1f32.powf(-scroll_delta * 0.1));
                    self.zoom /= zoom_delta;

                    
                    let new_scale = f128::from(4.0) / self.zoom;
                    self.center.0 = mouse_re - f128::from(rel_x - 0.5) * new_scale;
                    self.center.1 = mouse_im - f128::from(rel_y - 0.5) * new_scale;

                    need_redraw = true;
                }
            }

            // Mouse drag
            if response.dragged() {
                let drag = response.drag_delta();
                let scale_factor = f128::from(4.0) / self.zoom;
                let dx = f128::from(drag.x / rect.width());
                let dy = f128::from(drag.y / rect.height());

                self.center.0 -= dx * scale_factor;
                self.center.1 -= dy * scale_factor;
                need_redraw = true;
            }

            if need_redraw {
                self.needs_update = true;
                ctx.request_repaint();
            }
        });
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            unsafe {
                self.cleanup_framebuffer();
                
                if let Some(program) = self.gl_program.take() {
                    gl.delete_program(program);
                }
                if let Some(vao) = self.gl_vao.take() {
                    gl.delete_vertex_array(vao);
                }
            }
        }
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(539.0, 575.0))
            .with_title("Mandelbrot Explorer (GLSL + Texture)"),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "Mandelbrot Explorer",
        native_options,
        Box::new(|cc| {
            if cc.gl.is_none() {
                eprintln!("GL context not available! Make sure to enable it.");
                return Err("GL context not available".into());
            }
            Ok(Box::new(MandelbrotApp::new(cc)))
        }),
    )
}