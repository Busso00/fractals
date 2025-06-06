use eframe::{egui, App, glow};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use eframe::glow::HasContext;

// Struct to hold the application's state
struct MandelbrotApp {
    center: (f32, f32),           // Current center of the view in fractal space
    zoom: f32,                    // Current zoom level
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

// Vertex shader - creates a fullscreen quad
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

// Fragment shader for the Mandelbrot set
const FRAGMENT_SHADER: &str = r#"
    #version 300 es
    precision highp float;

    uniform vec2 u_resolution;
    uniform vec2 u_center;
    uniform float u_scale;
    uniform int u_max_iter;

    out vec4 fragColor;

    int mandelbrot_f32(float cre, float cim, int max_iter) {
        float zr = 0.0;
        float zi = 0.0;
        int iter = 0;
        
        while (iter < max_iter) {
            float zr2 = zr * zr;
            float zi2 = zi * zi;
            float mag2 = zr2 + zi2;
            
            if (mag2 > 4.0) break;
            
            float temp_zr = zr2 - zi2;
            float zr_new = temp_zr + cre;
            float zi_new = 2.0 * zr * zi + cim;
            
            zr = zr_new;
            zi = zi_new;
            iter++;
        }
        
        return iter;
    }
    
    void main() {
        vec2 ndc = gl_FragCoord.xy / u_resolution.xy;
        
        // Map normalized device coordinates from [0,1] to [-1,1]
        float fx = ndc.x * 2.0 - 1.0;
        float fy = ndc.y * 2.0 - 1.0;
        
        // Scale and translate relative to the center
        float current_view_width_half = 2.0 / u_scale;
        
        fx = fx * current_view_width_half;
        fy = fy * current_view_width_half;

        // Reconstruct complex number 'c' for Mandelbrot
        float cre = u_center.x + fx;
        float cim = u_center.y + fy;
        
        // Calculate iterations
        int iter = mandelbrot_f32(cre, cim, u_max_iter);
        
        // Color mapping
        if (iter == u_max_iter) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            float t = float(iter) / float(u_max_iter);
            t = sqrt(t);
            
            float pi2 = 6.28318530718;

            float r = 0.5 + 0.5 * cos(3.0 + t * pi2);
            float g = 0.5 + 0.5 * cos(3.0 + t * pi2 + 2.0 * pi2 / 3.0);
            float b = 0.5 + 0.5 * cos(3.0 + t * pi2 + 4.0 * pi2 / 3.0);
            
            fragColor = vec4(r, g, b, 1.0);
        }
    }
"#;

impl MandelbrotApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("You need to enable GL for this application");
        let gl = Arc::clone(gl);

        Self {
            center: (-0.5, 0.0),
            zoom: 1.0,
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
            let prev_program = self.gl.get_parameter_i32(glow::CURRENT_PROGRAM);
            let prev_vao = self.gl.get_parameter_i32(glow::VERTEX_ARRAY_BINDING);
            let prev_framebuffer = self.gl.get_parameter_i32(glow::FRAMEBUFFER_BINDING);
            let mut prev_viewport = [0i32; 4];
            self.gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);

            // Bind framebuffer and setup viewport
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            self.gl.viewport(0, 0, width as i32, height as i32);

            // Use shader program
            self.gl.use_program(Some(program));
            self.gl.bind_vertex_array(Some(vao));

            // Get uniform locations and check they exist
            let u_resolution = self.gl.get_uniform_location(program, "u_resolution");
            let u_center = self.gl.get_uniform_location(program, "u_center");
            let u_scale = self.gl.get_uniform_location(program, "u_scale");
            let u_max_iter = self.gl.get_uniform_location(program, "u_max_iter");

            // Set uniforms with error checking
            if let Some(loc) = u_resolution {
                self.gl.uniform_2_f32(Some(&loc), width as f32, height as f32);
            }
            if let Some(loc) = u_center {
                self.gl.uniform_2_f32(Some(&loc), self.center.0, self.center.1);
            }
            if let Some(loc) = u_scale {
                self.gl.uniform_1_f32(Some(&loc), self.zoom);
            }
            if let Some(loc) = u_max_iter {
                self.gl.uniform_1_i32(Some(&loc), self.max_iterations as i32);
            }

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
                    self.center = (-0.5, 0.0);
                    self.zoom = 1.0;
                    self.needs_update = true;
                }
            });

            ui.label(format!("Render time: {:.1} ms", 
                self.t_render.lock().unwrap().as_secs_f64() * 1000.0));

            ui.separator();

            // --- FRACTAL DISPLAY ---
            let available_size = ui.available_size();
            let size = egui::Vec2::new(
                available_size.x.min(available_size.y).max(256.0),
                available_size.x.min(available_size.y).max(256.0),
            );
            
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
            let pan_speed = 0.05 / self.zoom;
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
            if scroll_delta != 0.0 && response.hovered() {
                if let Some(pointer_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                    let rel_x = (pointer_pos.x - rect.left()) / rect.width();
                    let rel_y = (pointer_pos.y - rect.top()) / rect.height();

                    let scale = 4.0 / self.zoom;
                    let mouse_re = self.center.0 + (rel_x - 0.5) * scale;
                    let mouse_im = self.center.1 + (rel_y - 0.5) * scale;

                    let zoom_delta = 1.1f32.powf(-scroll_delta * 0.1);
                    self.zoom *= zoom_delta;

                    // Clamp zoom to prevent numerical issues
                    self.zoom = self.zoom.clamp(0.1, 1e10);

                    let new_scale = 4.0 / self.zoom;
                    self.center.0 = mouse_re - (rel_x - 0.5) * new_scale;
                    self.center.1 = mouse_im - (rel_y - 0.5) * new_scale;

                    need_redraw = true;
                }
            }

            // Mouse drag
            if response.dragged() {
                let drag = response.drag_delta();
                let scale_factor = 4.0 / self.zoom;
                let dx = drag.x / rect.width();
                let dy = drag.y / rect.height();

                self.center.0 -= dx * scale_factor;
                self.center.1 += dy * scale_factor;
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
            .with_inner_size(egui::vec2(600.0, 700.0))
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