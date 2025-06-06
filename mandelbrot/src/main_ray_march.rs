use ocl::{ProQue, Buffer};
use egui::ColorImage;
use eframe::{egui, App};
use std::sync::{Arc};
use std::time::{Instant, Duration};
use f128::f128;
use num_traits::ToPrimitive;

const MANDELBROT_KERNEL_DD: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// A double-double type
typedef struct { double hi, lo; } dd;

// Split a double into high/low for Dekker
static inline dd dd_from_double(double a) {
    return (dd){ a, 0.0 };
}

// Add two double-doubles
static inline dd dd_add(dd a, dd b) {
    double s = a.hi + b.hi;
    double v = s - a.hi;
    double t = ((b.hi - v) + (a.hi - (s - v))) + a.lo + b.lo;
    double z = s + t;
    return (dd){ z, t - (z - s) };
}

// Subtract two double-doubles
static inline dd dd_sub(dd a, dd b) {
    return dd_add(a, (dd){ -b.hi, -b.lo });
}

// Multiply two double-doubles using Dekker/FMA
static inline dd dd_mul(dd a, dd b) {
    double p = a.hi * b.hi;
    double err = fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
    double z = p + err;
    return (dd){ z, err - (z - p) };
}

static inline dd dd_div(dd a, dd b) {
    double q1 = a.hi / b.hi;

    // r = a - b*q1
    dd q1b = dd_mul(dd_from_double(q1), b);
    dd r = dd_add(a, (dd){ -q1b.hi, -q1b.lo });

    double q2 = r.hi / b.hi;

    double result_hi = q1 + q2;
    double result_lo = q2 - (result_hi - q1); // error correction

    return (dd){ result_hi, result_lo };
}

// Returns true if a < b
static inline int dd_lt(dd a, dd b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

// Returns true if a > b
static inline int dd_gt(dd a, dd b) {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
}

// Returns true if a == b (optional, if needed)
static inline int dd_eq(dd a, dd b) {
    return (a.hi == b.hi) && (a.lo == b.lo);
}

// Returns true if a <= b
static inline int dd_le(dd a, dd b) {
    return dd_lt(a, b) || dd_eq(a, b);
}

// Returns true if a >= b
static inline int dd_ge(dd a, dd b) {
    return dd_gt(a, b) || dd_eq(a, b);
}


typedef struct {
    int iter;
    dd zr;
    dd zi;
} mandelbrot_result;

// Modified function to return final z values
static inline mandelbrot_result mandelbrot_iter(dd cre, dd cim, int max_iter) {
    dd zr = dd_from_double(0.0);
    dd zi = dd_from_double(0.0);
    int iter = 0;
    
    while (iter < max_iter) {
        // zr2 = zr*zr, zi2 = zi*zi
        dd zr2 = dd_mul(zr, zr);
        dd zi2 = dd_mul(zi, zi);
        dd mag2 = dd_add(zr2, zi2);
        
        // test |z|^2 > 4 ⇒ zr2+zi2 > 4
        if (mag2.hi > 4.0) break;
        
        // zr_new = zr2 - zi2 + cre
        dd tmp = dd_add(zr2, (dd){-zi2.hi, -zi2.lo});
        dd zr_new = dd_add(tmp, cre);
        
        // zi_new = 2*zr*zi + cim
        dd prod = dd_mul(zr, zi);
        prod = dd_add(prod, prod); // *2
        dd zi_new = dd_add(prod, cim);
        
        zr = zr_new;
        zi = zi_new;
        iter++;
    }
    
    mandelbrot_result result;
    result.iter = iter;
    result.zr = zr;
    result.zi = zi;
    return result;
}



static inline float get_mandelbrot_height(
    dd world_x_dd, dd world_y_dd,
    dd center_x, dd center_y, dd mandelbrot_scale,
    int max_iter
) {
    // FIXED: Properly map world coordinates to Mandelbrot coordinates
    dd cre = dd_add(center_x, dd_mul(dd_sub(world_x_dd, center_x), mandelbrot_scale));
    dd cim = dd_add(center_y, dd_mul(dd_sub(world_y_dd, center_y), mandelbrot_scale));

    mandelbrot_result m_res = mandelbrot_iter(cre, cim, max_iter);
    
    if (m_res.iter == max_iter) {
        return 2.0f; // Points inside the set are at "sea level" (base of mountain)
    }
    
    float log_zn = log((float)(m_res.zr.hi*m_res.zr.hi + m_res.zi.hi*m_res.zi.hi));
    float nu = log(log_zn / log(2.0f)) / log(2.0f);
    float smooth_iter = (float)m_res.iter + 1.0f - nu;

    // FIXED: Scale height to reasonable range (0-2.0)
    return (smooth_iter / (float)max_iter) * 2.0f;
}

static inline float3 calculate_normal(
    dd world_x_dd, dd world_y_dd,
    dd center_x, dd center_y, dd mandelbrot_scale,
    int max_iter, float height_epsilon, float height
) {
    float h_xy = get_mandelbrot_height(world_x_dd, world_y_dd, center_x, center_y, mandelbrot_scale, max_iter);
    
    dd dx_dd = dd_add(world_x_dd, dd_from_double(height_epsilon));
    dd dy_dd = dd_add(world_y_dd, dd_from_double(height_epsilon));

    float h_xminus = get_mandelbrot_height(dd_sub(world_x_dd, dd_from_double(height_epsilon)), world_y_dd, center_x, center_y, mandelbrot_scale, max_iter);
    float h_yminus = get_mandelbrot_height(world_x_dd, dd_sub(world_y_dd, dd_from_double(height_epsilon)), center_x, center_y, mandelbrot_scale, max_iter);

    float nx = (h_xminus - height) / height_epsilon;
    float ny = (h_yminus - height) / height_epsilon;
    float nz = 1.0f; // Assuming height is along Z, and Z points to viewer

    float3 normal = normalize((float3)(nx, ny, nz));
    return normal;
}


static inline float3 reflect(float3 I, float3 N) {
    // Ensure N is normalized for correct reflection calculation
    N = normalize(N); 
    // The formula: R = I - 2 * dot(I, N) * N
    return I - 2.0f * dot(I, N) * N;
}

__kernel void mandelbrot_3d_surface_render(
    __global uchar* output,   // RGB pixel data (3 bytes per pixel)
    const int width,
    const int height,
    const double center_x_hi,
    const double center_y_hi,
    const double scale_hi,
    const double center_x_lo,
    const double center_y_lo,
    const double scale_lo,
    const int max_iter
) {

    float fov_degrees = 45.0f; // FIXED: Reduced FOV for better view

    int x_pixel = get_global_id(0);
    int y_pixel = get_global_id(1);
    int idx = y_pixel * width + x_pixel;

    dd center_x_dd = (dd){ center_x_hi, center_x_lo };
    dd center_y_dd = (dd){ center_y_hi, center_y_lo };
    dd mandelbrot_scale = (dd){ scale_hi, scale_lo };

    // --- FIXED CAMERA POSITION (higher up) ---
    float3 camera_pos = (float3)((float)center_x_hi, (float)center_y_hi, 5.0f); 
    
    // --- CAMERA DIRECTION (looking down at the surface) ---
    float3 look_at_point = (float3)((float)center_x_hi, (float)center_y_hi, 0.0f);
    float3 camera_dir = normalize(look_at_point - camera_pos);
    
    // --- CAMERA UP ---
    float3 cam_up = (float3)(0.0f, -1.0f, 0.0f);
    float3 camera_up_vec = normalize(cam_up);
    
    float3 camera_right_vec = normalize(cross(camera_dir, camera_up_vec));
    float3 actual_camera_up_vec = normalize(cross(camera_right_vec, camera_dir));

    // --- LIGHT POSITION ---
    float3 light_pos = (float3)((float)center_x_hi - 2.0f, (float)center_y_hi - 2.0f, 1.0f);

    // Screen setup for ray generation
    float aspect_ratio = (float)width / (float)height;
    float fov_rad = radians(fov_degrees);
    float tan_fov = tan(fov_rad / 2.0f);

    // Normalized pixel coordinates (from -1 to 1)
    float ndc_x = (2.0f * (float)x_pixel / (float)width - 1.0f);
    float ndc_y = (2.0f * (float)y_pixel / (float)height - 1.0f);

    // Ray direction in camera space
    float3 ray_target_camera_space = (float3)(ndc_x * aspect_ratio * tan_fov, ndc_y * tan_fov, -1.0f);
    
    // Convert ray direction to world space
    float3 ray_dir = normalize(
        ray_target_camera_space.x * camera_right_vec +
        ray_target_camera_space.y * actual_camera_up_vec +
        ray_target_camera_space.z * camera_dir
    );

    uchar3 final_color = (uchar3){0, 0, 0}; // FIXED: Dark blue background instead of red
    
    float3 intersection_point = camera_pos - ray_dir;

    dd hit_px_dd = dd_from_double(intersection_point.x);
    dd hit_py_dd = dd_from_double(intersection_point.y);

    float h = get_mandelbrot_height(hit_px_dd, hit_py_dd, center_x_dd, center_y_dd, mandelbrot_scale, max_iter);
    
    // Calculate Normal
    float normal_epsilon = 0.001f; // FIXED: Smaller epsilon
    float3 surface_normal = calculate_normal(
        hit_px_dd, hit_py_dd,
        center_x_dd, center_y_dd, mandelbrot_scale,
        max_iter, normal_epsilon, h
    );
        
    // --- Lighting Calculation ---
    float3 light_dir_to_surface = normalize(light_pos - intersection_point); 
        
    float diffuse = max(dot(surface_normal, light_dir_to_surface), 0.0f);
        
    float ambient = 0.4f;
    float lighting = ambient + diffuse * 0.6f;
        
    // Specular component
    float3 view_dir_to_camera = normalize(camera_pos - intersection_point); 
    float3 reflected_light_dir = reflect(-light_dir_to_surface, surface_normal);
        
    float shininess = 64.0f; 
    float specular_strength = 0.4f; 
    float spec = pow(max(dot(view_dir_to_camera, reflected_light_dir), 0.0f), shininess);
    lighting += spec * specular_strength; 

    lighting = clamp(lighting, 0.2f, 1.0f); 

    // --- FIXED: Better coloring based on height ---
    
    if (h < 2.0) {

        // CLASSIC MANDELBROT BASE COLORS (this is the key change!)
        float t = log2(h+1);
        float pi2 = 6.28318f;
        
        // Generate classic rainbow coloring
        float base_r = 1.0f;
        float base_g = 1.0f;
        float base_b = 1.0f;

        final_color.x = (uchar)(base_r * lighting * 255.0f);
        final_color.y = (uchar)(base_g * lighting * 255.0f);
        final_color.z = (uchar)(base_b * lighting * 255.0f);
    }
    
    output[idx * 3] = final_color.x;     
    output[idx * 3 + 1] = final_color.y; 
    output[idx * 3 + 2] = final_color.z; 
}
"#;

// Use u8 buffer for RGB data (3 bytes per pixel)

pub struct MandelbrotApp {
    center: (f128, f128),
    zoom: f128,
    texture: Option<egui::TextureHandle>,
    max_iterations: i32,
    pro_que: Arc<ProQue>,
    t_mt: Duration,
    t_gpu: Duration
}

fn split_f128_to_dd(val: f128) -> (f64, f64) {
    let hi = val.to_f64().unwrap_or(0.0); // Convert to f64, losing precision
    let lo = (val - f128::from(hi)).to_f64().unwrap_or(0.0); // Remainder as low part
    (hi, lo)
}

impl MandelbrotApp {
    pub fn new() -> Self {
        let pro_que = ProQue::builder()
            .src(MANDELBROT_KERNEL_DD)
            .build()
            .expect("OpenCL build failed");
         
        Self {
            center: (f128::from(0.0), f128::from(0.0)),
            zoom: f128::from(1.0),
            texture: None,
            max_iterations: 128,
            pro_que: Arc::new(pro_que),
            t_mt: Duration::ZERO,
            t_gpu: Duration::ZERO
        }
    }

    /// Renders the *entire* image in one OpenCL dispatch with GPU coloring.
    fn render_gpu(&mut self, width: usize, height: usize) -> ColorImage {
        let t0 = Instant::now();

        let total_pixels = width * height;
        let total_bytes = total_pixels * 3; // 3 bytes per pixel (RGB)
        
        // Create buffer for RGB pixel data (3 bytes per pixel)
        let buffer = Buffer::<u8>::builder()
            .queue(self.pro_que.queue().clone())
            .len(total_bytes)
            .build()
            .expect("Buffer build");

        let (center_x_hi, center_x_lo) = split_f128_to_dd(self.center.0);
        let (center_y_hi, center_y_lo) = split_f128_to_dd(self.center.1);
        let (scale_hi, scale_lo) = split_f128_to_dd(f128::from(4.0) / self.zoom);
        
        let kernel = self.pro_que.kernel_builder("mandelbrot_3d_surface_render")
            .arg(&buffer)
            .arg(width as i32)
            .arg(height as i32)
            .arg(center_x_hi)
            .arg(center_y_hi)
            .arg(scale_hi)
            .arg(center_x_lo)
            .arg(center_y_lo)
            .arg(scale_lo)
            .arg(self.max_iterations as i32)
            .build()
            .expect("Kernel build");

        // Execute kernel
        unsafe {
            kernel.cmd()
                .global_work_size([width, height])
                .local_work_size([16, 16])
                .enq()
                .expect("Kernel enqueue");
        }

        // Read back RGB pixel data
        let mut raw = vec![0u8; total_bytes];
        buffer.read(&mut raw).enq().expect("Read buffer");

        // Convert to egui::Color32 (read 3 bytes per pixel)
        let pixels = raw.chunks_exact(3)
            .map(|rgb| egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]))
            .collect();

        println!("GPU render time elapsed: {:?}", t0.elapsed());
        self.t_gpu = t0.elapsed();
        
        ColorImage { size: [width, height], pixels }
    }

    fn render(&mut self, w: usize, h: usize) -> ColorImage {
        // For now, always use GPU rendering since we removed CPU fallback
        self.render_gpu(w, h)
    }
}

impl App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = egui::Vec2::new(512.0, 512.0);
            let size = [available_size.x as usize, available_size.y as usize];
            
            // UI Controls
            ui.horizontal(|ui| {
                ui.label("Iterations:");
                let old_iter = self.max_iterations;
                ui.add(egui::Slider::new(&mut self.max_iterations, 128..=4096).logarithmic(true));
                if old_iter != self.max_iterations {
                    self.texture = None; // Force redraw if iterations changed
                }
                
                ui.separator();
                
                ui.label(format!("Zoom: {:.2}x", self.zoom.to_f64().unwrap()));
                ui.label(format!("Center: ({:.6}, {:.6})", self.center.0.to_f64().unwrap(), self.center.1.to_f64().unwrap()));
                
                if ui.button("Reset View").clicked() {
                    self.center = (f128::from(0.0), f128::from(0.0));
                    self.zoom = f128::from(1.0);
                    self.texture = None;
                }
            });
            
            // Mandelbrot rendering
            if self.texture.is_none() || self.texture.as_ref().unwrap().size() != size {
                let img = self.render(size[0], size[1]);
                self.texture = Some(ui.ctx().load_texture("mandelbrot", img, Default::default()));
            }

            if let Some(tex) = &self.texture {
                let response = ui.image(tex);              
                let mut need_redraw = false;

                // Pan with arrow keys
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
                
                // Zoom controls
                let is_hovered = response.hovered();
                let scroll_delta = ctx.input(|i| i.raw_scroll_delta);
                let scroll_delta_y = scroll_delta.y;     
                let drag = ctx.input(|i| i.pointer.delta());
                let drag_x = drag.x;
                let drag_y = drag.y;

                if scroll_delta_y != 0.0 {
                    if let Some(pointer_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                        if is_hovered {
                            // Calculate position in fractal space before zoom
                            let rel_x = f128::from((pointer_pos.x - ui.min_rect().left()) / available_size.x);
                            let rel_y = f128::from((pointer_pos.y - ui.min_rect().top()) / available_size.y);
                            let scale = f128::from(4.0) / self.zoom;
                            let mouse_re = self.center.0 + (rel_x - f128::from(0.5)) * scale;
                            let mouse_im = self.center.1 + (rel_y - f128::from(0.5)) * scale;
                            
                            // Apply zoom
                            let zoom_delta = 1.1f64.powf(scroll_delta_y as f64 * 0.1);
                            self.zoom *= f128::from(zoom_delta);
                            
                            // Adjust center to keep mouse position fixed
                            let new_scale = f128::from(4.0) / self.zoom;
                            self.center.0 = mouse_re - (rel_x - f128::from(0.5)) * new_scale;
                            self.center.1 = mouse_im - (rel_y - f128::from(0.5)) * new_scale;
                            
                            need_redraw = true;
                        }
                    }
                } else {
                    // Handle dragging
                    if (drag_x != 0.0 || drag_y != 0.0) && ctx.input(|i| i.pointer.is_decidedly_dragging()) {
                        let scale = f128::from(4.0) / self.zoom;
                        self.center.0 -= f128::from(drag_x) / f128::from(size[0]) * scale;
                        self.center.1 -= f128::from(drag_y) / f128::from(size[1]) * scale;
                        need_redraw = true;
                    }
                }

                if need_redraw {
                    self.texture = None;
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions {
        ..Default::default()
    };

    native_options.viewport.inner_size = Some((512.0+16.0, 512.0+40.0).into());

    eframe::run_native(
        "Mandelbrot",
        native_options,
        Box::new(|_cc| Ok(Box::new(MandelbrotApp::new()))),
    )
}