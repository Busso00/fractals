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
 
#define OVERFLOW_THRESH uvec4(0u, 0u, 0u, 0x40000000u)
#define F128_EXP_BIAS   16383

uniform uvec4 u_offset_x[512];
uniform uvec4 u_offset_y[512];

uniform int u_max_iter;    // Maximum iterations
uniform uvec4 u_center_re;   // Center real part components (p1, p2, p3, p4) as uints
uniform uvec4 u_center_im;   // Center imag part components (p1, p2, p3, p4) as uints

out vec4 fragColor;

uvec4 shift128(uvec4 value, int shiftBits) {
    if (shiftBits == 0) return value;

    uvec4 result = uvec4(0u);

    if (shiftBits > 0) {
        
        if (shiftBits >= 128) return uvec4(0); // shifted out completely

        int wordShift = shiftBits / 32;
        int bitShift = shiftBits % 32;

        for (int i = 3; i >= 0; i -= 1) {
            if (i - wordShift < 0) continue;

            uint lower = value[i - wordShift] << bitShift;
            uint upper = 0u;

            if (bitShift != 0 && i - wordShift - 1 >= 0)
                upper = value[i - wordShift - 1] >> (32 - bitShift);

            result[i] = lower | upper;
        }

    } else {
        
        shiftBits = -shiftBits;
        if (shiftBits >= 128) return uvec4(0); // shifted out completely

        int wordShift = shiftBits / 32;
        int bitShift = shiftBits % 32;

        for (int i = 0; i < 4; ++i) {
            if (i + wordShift > 3) continue;

            uint upper = value[i + wordShift] >> bitShift;
            uint lower = 0u;

            if (bitShift != 0 && i + wordShift + 1 <= 3)
                lower = value[i + wordShift + 1] << (32 - bitShift);

            result[i] = upper | lower;
        }
    }

    return result;
}


uvec4 shift128rem(uvec4 value, uint rem, int shiftBits) {
    if (shiftBits == 0) return value;

    uvec4 result = uvec4(0u);

    if (shiftBits > 0) {
        
        if (shiftBits >= 128) return uvec4(0); // shifted out completely

        int wordShift = shiftBits / 32;
        int bitShift = shiftBits % 32;

        for (int i = 3; i >= 0; i -= 1) {
            if (i - wordShift < 0) continue;

            uint lower = value[i - wordShift] << bitShift;
            uint upper = 0u;

            if (bitShift != 0 && i - wordShift - 1 >= 0)
                upper = value[i - wordShift - 1] >> (32 - bitShift);
            else 
                upper = rem >> (32 - bitShift);

            result[i] = lower | upper;
        }

    } else {
        
        shiftBits = -shiftBits;
        if (shiftBits >= 128) return uvec4(0); // shifted out completely

        int wordShift = shiftBits / 32;
        int bitShift = shiftBits % 32;

        for (int i = 0; i < 4; ++i) {
            if (i + wordShift > 3) continue;

            uint upper = value[i + wordShift] >> bitShift;
            uint lower = 0u;

            if (bitShift != 0 && i + wordShift + 1 <= 3)
                lower = value[i + wordShift + 1] << (32 - bitShift);

            result[i] = upper | lower;
        }
    }

    return result;
}


int cmp128Unsigned(uvec4 a, uvec4 b) {
    // Compare from most significant to least significant word
    if (a.w > b.w) return 1;
    if (a.w < b.w) return -1;

    if (a.z > b.z) return 1;
    if (a.z < b.z) return -1;

    if (a.y > b.y) return 1;
    if (a.y < b.y) return -1;

    if (a.x > b.x) return 1;
    if (a.x < b.x) return -1;

    return 0; // a == b
}

void sum32(uint a, uint b, uint carry_in , out uint result, out uint carry_out) {
    uint temp_sum = a + b;
    uint temp_carry = (temp_sum < a) ? 1u : 0u;
    result = temp_sum + carry_in;
    uint carry2 = (result < temp_sum) ? 1u : 0u;
    carry_out = temp_carry + carry2;
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

void sub32(uint a, uint b, uint borrow_in, out uint result, out uint borrow_out) {
    uint temp = b + borrow_in;
    result = a - temp;
    borrow_out = (a < temp || (temp < b && borrow_in == 1u)) ? 1u : 0u;
}

uvec4 sub_mantissa(uvec4 a, uvec4 b) {
    uvec4 result;
    uint borrow = 0u;

    sub32(a.x, b.x, borrow, result.x, borrow);
    sub32(a.y, b.y, borrow, result.y, borrow);
    sub32(a.z, b.z, borrow, result.z, borrow);
    sub32(a.w, b.w, borrow, result.w, borrow);

    return result;
}

uvec4 mul_mantissa(uvec4 a, uvec4 b, out uint rem) {

    uint prod[8];
    for (uint i = 0u; i < 8u; ++i) prod[i] = 0u;

    for (uint i = 0u; i < 4u; ++i) {
        uint ai = a[i];
        for (uint j = 0u; j < 4u; ++j) {
            uint bj = b[j];
            
            if ((i+j) < 2u){
                continue;
            }

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

    rem = prod[3];

    return uvec4(
        prod[4],
        prod[5],
        prod[6],
        prod[7]
    );
}

uvec4 toCustom128(uvec4 ieee) {
    uvec4 result = uvec4(0);

    uint lo   = ieee.x;
    uint mid1 = ieee.y;
    uint mid2 = ieee.z;
    uint hi   = ieee.w;

    bool sign = (hi & 0x80000000u) != 0u;
    uint exponentBits = (hi >> 16) & 0x7FFFu;
    int exponent = int(exponentBits) - F128_EXP_BIAS + 14;

    uvec4 mantissa = uvec4(lo, mid1, mid2, hi & 0xFFFFu);

    bool isSubnormal = (exponentBits == 0u);
    if (!isSubnormal) {
        mantissa = add_mantissa(mantissa, uvec4(0u, 0u, 0u, 0x10000u));
    }

    int shift = exponent;
    uvec4 shifted = shift128(mantissa, shift);

    if (cmp128Unsigned(shifted, OVERFLOW_THRESH) >= 0) {
        return uvec4(0u, 0u, 0u, 0x40000000u);
    }

    shifted.w &= 0x3FFFFFFFu; // clear top 2 bits
    if (sign) shifted.w |= (1u << 31);

    return shifted;
}

uvec4 addCustom128(uvec4 a, uvec4 b, bool check){
    uint sign_a = a.w & 0x80000000u;
    uint sign_b = b.w & 0x80000000u;

    uvec4 mantissa_a = uvec4(a.x, a.y, a.z, a.w & 0x7FFFFFFFu);
    uvec4 mantissa_b = uvec4(b.x, b.y, b.z, b.w & 0x7FFFFFFFu);

    if (((cmp128Unsigned(mantissa_a, OVERFLOW_THRESH) >= 0 )||(cmp128Unsigned(mantissa_b, OVERFLOW_THRESH) >= 0 )) && check) {
        return uvec4(0u, 0u, 0u, 0x40000000u);
    }

    if (sign_a == sign_b){
        uvec4 res = add_mantissa(a, b);
    
        if ((cmp128Unsigned(res, OVERFLOW_THRESH) >= 0) && check) {
            return uvec4(0u, 0u, 0u, 0x40000000u);
        }

        return uvec4(res.x, res.y, res.z, res.w | sign_a);
    } else{
        if (cmp128Unsigned(mantissa_a, mantissa_b) > 0){
            uvec4 res = sub_mantissa(mantissa_a, mantissa_b);
            return uvec4(res.x, res.y, res.z, res.w | sign_a);
        } else {
            uvec4 res = sub_mantissa(mantissa_b, mantissa_a);
            return uvec4(res.x, res.y, res.z, res.w | sign_b);
        }
    }
}

uvec4 subCustom128(uvec4 a, uvec4 b, bool check){
    uint b_hi = ((( b.w >> 31) ^ 1u ) << 31 ) | (b.w & 0x7FFFFFFFu);
    return addCustom128(a, uvec4(b.x, b.y, b.z, b_hi ), check);
}

uvec4 mulCustom128(uvec4 a, uvec4 b, bool check){

    uvec4 mantissa_a = uvec4(a.x, a.y, a.z, a.w & 0x7FFFFFFFu);
    uvec4 mantissa_b = uvec4(b.x, b.y, b.z, b.w & 0x7FFFFFFFu);


    if (((cmp128Unsigned(mantissa_a, OVERFLOW_THRESH) >= 0 )||(cmp128Unsigned(mantissa_b, OVERFLOW_THRESH) >= 0 )) && check) {
        return uvec4(0u, 0u, 0u, 0x40000000u);
    }

    uint sign_a = a.w & 0x80000000u;
    uint sign_b = b.w & 0x80000000u;

    uint rem = 0u;
    uvec4 res = mul_mantissa(mantissa_a, mantissa_b, rem);

    res = shift128rem(res, rem, 2);

    if ((cmp128Unsigned(res, OVERFLOW_THRESH) >= 0) && check) {
        return uvec4(0u, 0u, 0u, 0x40000000u);
    }

    return uvec4(res.x, res.y, res.z, res.w | (sign_b^sign_a));
}

int mandelbrot_qd(uvec4 cre, uvec4 cim, int max_iter) {
    uvec4 zr = uvec4(0u);
    uvec4 zi = uvec4(0u);
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        uvec4 zr2 = mulCustom128(zr, zr, false);
        uvec4 zi2 = mulCustom128(zi, zi, false);
        uvec4 mag2 = addCustom128(zr2, zi2, true);


        uvec4 mag2_mant = uvec4(mag2.x, mag2.y, mag2.z, mag2.w & 0x7FFFFFFFu);
        if (cmp128Unsigned(mag2_mant, uvec4(0u, 0u, 0u, 0x40000000)) == 0) { 
            break; 
        }

        uvec4 temp_zr = subCustom128(zr2, zi2, false);
        uvec4 zr_new = addCustom128(temp_zr, cre, false);

        uvec4 zr_zi = mulCustom128(zr, zi, false);
        uvec4 two_zr_zi = addCustom128(zr_zi, zr_zi, false);
        uvec4 zi_new = addCustom128(two_zr_zi, cim, false);

        zr = zr_new;
        zi = zi_new;
    }

    return iter;
}


void main() {
    
    uvec4 pixel_re_offset = u_offset_x[int(gl_FragCoord.xy.x)];
    uvec4 pixel_im_offset = u_offset_y[int(gl_FragCoord.xy.y)];

    uvec4 cre = addCustom128(toCustom128(u_center_re), toCustom128(pixel_re_offset), false);
    uvec4 cim = subCustom128(toCustom128(u_center_im), toCustom128(pixel_im_offset), false);

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
            max_iterations: 2048,
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