use eframe::glow::HasContext;
use eframe::{egui, glow, App};
use std::sync::Arc;
use f128::f128;

static INPUT_VALUES: [f32; 125] = [
    1.17549435e-38,    // index 0 - smallest normal float32
    3.40282347e+38,    // index 1 - largest normal float32
    1.40129846e-45,    // index 2 - smallest subnormal float32
    -0.0,              // index 3 - negative zero
    1.0 / 0.0,         // index 4 - positive infinity
    -1.0 / 0.0,        // index 5 - negative infinity
    0.0 / 0.0,         // index 6 - NaN
    1.0000001,         // index 7 - just above 1.0
    0.9999999,         // index 8 - just below 1.0
    3.14159265,        // index 9 - pi
    -2.71828183,       // index 10 - -e
    1.41421356,        // index 11 - sqrt(2)
    0.57721566,        // index 12 - Euler-Mascheroni constant
    2.80259693e-45,    // index 13 - 2 * smallest subnormal
    4.20389539e-45,    // index 14 - 3 * smallest subnormal
    7.00649232e-45,    // index 15 - 5 * smallest subnormal
    1.40129846e-44,    // index 16 - 10 * smallest subnormal
    2.10194770e-44,    // index 17 - 15 * smallest subnormal
    1.17549421e-38,    // index 18 - just below smallest normal
    1.17549429e-38,    // index 19 - just below smallest normal (different mantissa)
    5.87747175e-39,    // index 20 - largest subnormal / 2
    2.93873588e-39,    // index 21 - largest subnormal / 4
    1.46936794e-39,    // index 22 - largest subnormal / 8
    1.54441805e-44,    // index 23 - subnormal with mantissa = 0x0000B (11 in binary)
    1.68485069e-44,    // index 24 - subnormal with mantissa = 0x0000C (12 in binary)
    0.0,                                    // index 25 - positive zero
    f32::from_bits(0x7F800001),            // index 26 - signaling NaN
    f32::from_bits(0x7FC12345),            // index 27 - NaN with payload
    f32::from_bits(0xFFC00000),            // index 28 - negative NaN
    -3.4028235e+38,                        // index 29 - maximum finite negative
    f32::from_bits(0x00000001),            // index 30 - smallest positive subnormal
    f32::from_bits(0x007FFFFF),            // index 31 - largest positive subnormal
    f32::from_bits(0x80000001),            // index 32 - smallest negative subnormal
    f32::from_bits(0x807FFFFF),            // index 33 - largest negative subnormal
    f32::from_bits(0x00400000),            // index 34 - mid-range subnormal
    f32::from_bits(0x00000002),            // index 35 - subnormal with single bit
    f32::from_bits(0x00200000),            // index 36 - subnormal with high bit
    f32::from_bits(0x00555555),            // index 37 - subnormal pattern 1
    f32::from_bits(0x00AAAAAA),            // index 38 - subnormal pattern 2
    f32::from_bits(0x0000000F),            // index 39 - subnormal near zero
    f32::from_bits(0x80123456),            // index 40 - negative subnormal pattern
    f32::from_bits(0x007FFFFE),            // index 41 - subnormal boundary
    f32::from_bits(0x00100000),            // index 42 - subnormal with trailing zeros
    f32::from_bits(0x00765432),            // index 43 - complex subnormal
    f32::from_bits(0x00000080),            // index 44 - subnormal edge case
    f32::from_bits(0x00800000),            // index 45 - smallest positive normal
    f32::from_bits(0x7F7FFFFF),            // index 46 - largest positive normal
    f32::from_bits(0x80800000),            // index 47 - smallest negative normal
    f32::from_bits(0xFF7FFFFF),            // index 48 - largest negative normal
    f32::from_bits(0x00800001),            // index 49 - exponent boundary +1
    f32::from_bits(0x3F800000),            // index 50 - mid-range exponent (1.0)
    f32::from_bits(0x7F000000),            // index 51 - high exponent test
    f32::from_bits(0x01000000),            // index 52 - low exponent test
    f32::from_bits(0x81000000),            // index 53 - negative exponent boundaries
    f32::from_bits(0x7F800000),            // index 54 - max exponent, zero mantissa (inf)
    f32::from_bits(0x55800000),            // index 55 - exponent bit pattern 1
    f32::from_bits(0x2A800000),            // index 56 - exponent bit pattern 2
    f32::from_bits(0x7F7FFFFE),            // index 57 - near-infinity normal
    f32::from_bits(0x01800000),            // index 58 - exponent transition +
    f32::from_bits(0x81800000),            // index 59 - exponent transition -
    f32::from_bits(0x3F7FFFFF),            // index 60 - boundary stress test
    f32::from_bits(0x3E800000),            // index 61 - bias boundary -
    f32::from_bits(0x7E800000),            // index 62 - extreme positive bias
    f32::from_bits(0x40800000),            // index 63 - bias boundary +
    f32::from_bits(0x3F000000),            // index 64 - 0.5f conversion
    f32::from_bits(0x3FFFFFFF),            // index 65 - all mantissa bits set
    f32::from_bits(0x3F800001),            // index 66 - single mantissa bit
    f32::from_bits(0x3F955555),            // index 67 - alternating mantissa bits
    f32::from_bits(0x3FAAAAAA),            // index 68 - inverse alternating bits
    f32::from_bits(0x3F800000 | 0x700000), // index 69 - high mantissa bits only
    f32::from_bits(0x3F800000 | 0x000007), // index 70 - low mantissa bits only
    f32::from_bits(0x3F800000 | 0x003800), // index 71 - middle mantissa bits
    f32::from_bits(0x3F800000 | 0x400000), // index 72 - mantissa boundary 1
    f32::from_bits(0x3F8F0F0F),            // index 73 - complex mantissa pattern
    f32::from_bits(0x3F810203),            // index 74 - mantissa shift test 1
    f32::from_bits(0x3F8FEDCB),            // index 75 - mantissa shift test 2
    f32::from_bits(0x3F800080),            // index 76 - precision boundary
    f32::from_bits(0x3F800040),            // index 77 - double precision edge
    f32::from_bits(0x3F8FFF00),            // index 78 - trailing zeros mantissa
    f32::from_bits(0x3FFFC000),            // index 79 - leading ones mantissa
    f32::from_bits(0x3F810101),            // index 80 - sparse mantissa bits
    f32::from_bits(0x3F8FEFEF),            // index 81 - dense mantissa bits
    f32::from_bits(0x3F8FFFFF),            // index 82 - mantissa overflow test
    f32::from_bits(0x3F823456),            // index 83 - bit rotation pattern 1
    f32::from_bits(0x3F8CDEF0),            // index 84 - bit rotation pattern 2
    f32::from_bits(0x3F8C3C30),            // index 85 - symmetrical mantissa
    f32::from_bits(0x3F812345),            // index 86 - asymmetrical mantissa
    f32::from_bits(0x40000000),            // index 87 - sign bit flip test 1 (+2.0)
    f32::from_bits(0xC0000000),            // index 88 - sign bit flip test 2 (-2.0)
    f32::from_bits(0xBF800000),            // index 89 - sign with normal (-1.0)
    f32::from_bits(0x80400000),            // index 90 - sign with subnormal
    f32::from_bits(0x7FFFFFFF),            // index 91 - sign boundary test
    -1.17549435e-38,                       // index 92 - negative tiny
    -3.4028235e+38,                        // index 93 - negative large
    f32::from_bits(0x80000001),            // index 94 - negative tiny subnormal
    f32::from_bits(0xFF7FFFFF),            // index 95 - negative maximum normal
    -1.0,                                  // index 96 - simple negative
    -2.0,                                  // index 97 - negative power of 2
    -0.5,                                  // index 98 - negative fraction
    f32::from_bits(0x80123456),            // index 99 - sign preservation complex
    f32::from_bits(0x55555555),            // index 100 - checkered pattern 1
    f32::from_bits(0xAAAAAAAA),            // index 101 - checkered pattern 2
    f32::from_bits(0x3F8FFFFE),            // index 102 - walking zeros
    f32::from_bits(0x3F811235),            // index 103 - fibonacci-like pattern
    f32::from_bits(0x3F82357B),            // index 104 - prime number pattern
    f32::from_bits(0x3F8B7E15),            // index 105 - chaos pattern
    f32::from_bits(0x00000004),            // index 106 - subnormal power of 2
    f32::from_bits(0x00000008),            // index 107 - subnormal power of 2
    f32::from_bits(0x00000010),            // index 108 - subnormal power of 2
    f32::from_bits(0x00000020),            // index 109 - subnormal power of 2
    f32::from_bits(0x00000040),            // index 110 - subnormal power of 2
    f32::from_bits(0x80000004),            // index 111 - negative subnormal power of 2
    f32::from_bits(0x80000008),            // index 112 - negative subnormal power of 2
    f32::from_bits(0x80000010),            // index 113 - negative subnormal power of 2
    f32::from_bits(0x3F800002),            // index 114 - just above 1.0 (precise)
    f32::from_bits(0x3F7FFFFE),            // index 115 - just below 1.0 (precise)
    f32::from_bits(0x40000001),            // index 116 - just above 2.0
    f32::from_bits(0x3FFFFFFE),            // index 117 - just below 2.0
    f32::from_bits(0x41000000),            // index 118 - 8.0
    f32::from_bits(0xC1000000),            // index 119 - -8.0
    f32::from_bits(0x42000000),            // index 120 - 32.0
    f32::from_bits(0xC2000000),            // index 121 - -32.0
    f32::from_bits(0x43000000),            // index 122 - 128.0
    f32::from_bits(0xC3000000),            // index 123 - -128.0
    f32::from_bits(0x7F7FFFFD),            // index 124 - near maximum finite
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

uniform float u_values[125];


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
    // Convert signaling NaN to quiet NaN by setting MSB of mantissa
    return uvec4(snan.x, snan.y, snan.z, (snan.w | 0x8000u) & 0x7FFFFFFFu );
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


void main() {
    
    // Each float's f128 representation occupies 4 columns.
    uint value_idx = uint(floor(gl_FragCoord.x / 4.0));
    
    // Get the float value for this column block
    float val = u_values[value_idx]; // Accessing the array directly
    uvec4 u_value128 = float_to_f128(val);

    // Determine which byte within this f128 value this fragment represents
    uint cx_local = uint(gl_FragCoord.x) % 4u; // Column within the 4-pixel block
    uint cy = uint(gl_FragCoord.y);            // Row within the 4-pixel block

    uint byteIndex = cy * 4u + cx_local; // Combined index from 0 to 15

    uint wordIndex = byteIndex / 4u;   // Which word (x, y, z, or w)
    uint byteInWord = byteIndex % 4u;  // Which byte within that word (0-3)

    uint word;
    if (wordIndex == 0u) {
        word = u_value128.x;
    } else if (wordIndex == 1u) {
        word = u_value128.y;
    } else if (wordIndex == 2u) {
        word = u_value128.z;
    } else { // wordIndex == 3u
        word = u_value128.w;
    }

    uint shift = byteInWord * 8u;
    uint byteVal = (word >> shift) & 0xFFu;

    float red = float(byteVal) / 255.0;

    fragColor = vec4(red, 0.0, 0.0, 1.0);
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
            let texture_height = 4;     // Each value is 4 pixels high

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
                .expect("Uniform 'u_values' not found. Check if it is declared and used in the shader.");
            
            gl.uniform_1_f32_slice(Some(&loc_values), &INPUT_VALUES);

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

            ui.heading("f128 Representations of Float Values");

            // Iterate through each of the 13 sets of bytes
            for i in 0..INPUT_VALUES.len() {
                ui.separator();
                let f128_val = unsafe { std::mem::transmute::<f128, [u8; 16]>(f128::from(INPUT_VALUES[i]))};
                
                let f128_form:Vec<String>= f128_val.to_vec().iter()
                    .map(|&b| format!("{:02X}", b))
                    .collect();

            
                let mut current_value_bytes = Vec::with_capacity(16);
                // Extract the 16 bytes for the current value from the read_pixels data
                for row in 0..4 {
                    for col_offset in 0..4 {
                        let global_col = (i * 4 + col_offset) as usize;
                        let pixel_idx = (row * self.texture_width as usize + global_col) * 4; // *4 for RGBA
                        if pixel_idx + 0 < data.len() {
                            current_value_bytes.push(data[pixel_idx + 0]); // Just the red channel
                        }
                    }
                }
                let byte_strings: Vec<String> = current_value_bytes
                    .iter()
                    .map(|&b| format!("{:02X}", b))
                    .collect();
                
                assert!( f128_form == byte_strings, "FAILED TEST {:} convert value {:}: {:?}: {:?}",i ,INPUT_VALUES[i] , f128_form, byte_strings)
            }
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

fn link_program(gl: &glow::Context, vs: glow::Shader, fs: glow::Shader) -> Result<glow::Program, String> {
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