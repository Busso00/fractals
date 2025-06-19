use ocl::{ProQue, Buffer};
use egui::ColorImage;
use eframe::{egui, App};
use std::sync::{Arc};
use std::time::{Instant, Duration};

const TETHRAEDRON_KERNEL_F32: &str = r#"
// Menger sponge 3D texture kernel


static inline float tetrahedron_sdf(float3 p, float size) {
    // Define the 4 face normals of a regular tetrahedron
    float3 v1 = (float3)( 0.0f, 0.0f, 0.433f);
    float3 v2 = (float3)(-0.5f, 0.0f, -0.433f);
    float3 v3 = (float3)( 0.5f, 0.0f, -0.433f);
    float3 v4 = (float3)( 0.0f, -0.866f, 0.0f);

    float3 mid1 = (v2 + v3 + v4)/ 3.0f;  
    float3 n1 = v1 - mid1;

    float3 mid2 = (v1 + v3 + v4) / 3.0f;
    float3 n2 = v2 - mid2;

    float3 mid3 = (v1 + v2 + v4) / 3.0f;
    float3 n3 = v3 - mid3;

    float3 mid4 = (v1 + v2 + v3) / 3.0f;
    float3 n4 = v4 - mid4;
    
    // Distance to each face plane
    float d1 = dot(p, n1) - size;
    float d2 = dot(p, n2) - size;
    float d3 = dot(p, n3) - size;
    float d4 = dot(p, n4) - size;
    
    // Return the maximum distance (intersection of half-spaces)
    return max(max(d1, d2), max(d3, d4));
}

// 3D Sierpinski Triangle (Tetrahedron) SDF
static inline float sierpinski_3d_sdf(float3 p, int iterations) {
    float d = tetrahedron_sdf(p, 0.5f);  // Start with unit tetrahedron
    //wrapper
    
    return d;
}


static inline float3 reflect(float3 I, float3 N) {
    // Ensure N is normalized for correct reflection calculation
    N = normalize(N); 
    // The formula: R = I - 2 * dot(I, N) * N
    return I - 2.0f * dot(I, N) * N;
}

// Raymarching function to find intersection with the Menger sponge
static inline bool ray_menger_intersection(
    float3 ray_origin,
    float3 ray_dir,
    float center_x,
    float center_y,
    int iter,
    float* t_out,
    float3* hit_point_out
) {
    float t = 0.01f;             // Start a bit away from the ray origin
    const float max_t = 100.0f;  // Max ray length
    const float epsilon = 0.001f; // Surface hit threshold

    for (int i = 0; i < 512; i++) {
        float3 p = ray_origin + ray_dir * t;

        // Transform world coordinates into sponge's local space
        float3 sponge_p = (float3)(
            (p.x - center_x),
            (p.y - center_y),
            p.z
        );

        float dist = sierpinski_3d_sdf(sponge_p, iter);

        if (dist < epsilon) {
            *t_out = t;
            *hit_point_out = p;
            return true;
        }

        t += dist;
        if (t > max_t) break;
    }

    return false;
}


// Calculate normal using gradient of SDF
static inline float3 calculate_normal(
    float3 p, float center_x, float center_y, int max_iter
) {
    const float epsilon = 0.001f;
    
    float3 sponge_p = (float3)(
        (p.x - center_x),
        (p.y - center_y),
        p.z
    );
    
    float d = sierpinski_3d_sdf(sponge_p, max_iter); 
    
    float3 n = (float3)(
        sierpinski_3d_sdf(sponge_p + (float3)(epsilon, 0, 0), max_iter) - d,
        sierpinski_3d_sdf(sponge_p + (float3)(0, epsilon, 0), max_iter) - d,
        sierpinski_3d_sdf(sponge_p + (float3)(0, 0, epsilon), max_iter) - d
    );
    
    return normalize(n);
}

__kernel void menger_3d_surface_render(
    __global uchar* output,
    int width,
    int height,
    float center_x,
    float center_y,
    float zoom,
    float cam_x, float cam_y, float cam_z,
    float look_x, float look_y, float look_z,
    int max_iter
) {

    float fov_degrees = 45.0f;

    int x_pixel = get_global_id(0);
    int y_pixel = get_global_id(1);
    int idx = y_pixel * width + x_pixel;

    float3 camera_pos = (float3)(cam_x, cam_y, cam_z);
    float3 look_at = (float3)(look_x, look_y, look_z);

    float3 world_up = (float3)(0.0f, -1.0f, 0.0f);
    float3 g_forward = normalize(look_at - camera_pos);
    float3 g_right = normalize(cross(g_forward, world_up));
    float3 g_up = normalize(cross(g_right, g_forward));


    float fov_rad = radians(fov_degrees);
    float tan_fov = tan(fov_rad / 2.0f);

    float ndc_x = (2.0f * (float)x_pixel / (float)width - 1.0f);
    float ndc_y = (2.0f * (float)y_pixel / (float)height - 1.0f);

    float3 ray_target = (float3)(ndc_x * tan_fov, ndc_y * tan_fov, 1.0f);
    float3 ray_dir = normalize(
        ray_target.x * g_right +
        ray_target.y * g_up +
        ray_target.z * g_forward
    );

    float3 hit_point;
    float t;
    uchar3 final_color = (uchar3)(255,0,0); // background

    bool hit = ray_menger_intersection(
        camera_pos, ray_dir,
        center_x, center_y,
        max_iter,
        &t, &hit_point
    );

    if (hit) {
        float3 normal = calculate_normal(hit_point, center_x, center_y, max_iter);

        float3 light_pos = (float3)(3.0f, 3.0f, 3.0f);
        float3 light_dir = normalize(light_pos - hit_point);
        float diffuse = max(dot(normal, light_dir), 0.0f);

        float3 view_dir = normalize(camera_pos - hit_point);
        float3 reflect_dir = normalize(reflect(-light_dir, normal));
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0f), 32.0f);

        float ambient = 0.3f;
        float shadow = 1.0f;

        // Soft shadows (optional)
        float3 shadow_origin = hit_point + normal * 0.01f;
        for (float st = 0.0f; st < length(light_pos - shadow_origin); st += 0.01f) {
            float3 sp = shadow_origin + light_dir * st;

            float3 sponge_sp = (float3)(
                (sp.x - center_x),
                (sp.y - center_y),
                sp.z
            );
            float d = sierpinski_3d_sdf(sponge_sp, max_iter);
            if (d < 0.01f) {
                shadow = 0.2f;
                break;
            }
        }

        float lighting = ambient + diffuse * 0.7f * shadow + spec * 0.5f * shadow;
        lighting = clamp(lighting, 0.1f, 1.0f);

        float3 base_color;
        base_color.x = 0.2f;
        base_color.y = 0.4f;
        base_color.z = 0.8f;

        base_color *= lighting;
        final_color = convert_uchar3_sat_rte(base_color * 255.0f);
    }

    output[idx * 3 + 0] = final_color.x;
    output[idx * 3 + 1] = final_color.y;
    output[idx * 3 + 2] = final_color.z;
}
"#;

// Use u8 buffer for RGB data (3 bytes per pixel)
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
}

pub struct SpongeApp {
    center: (f32, f32),
    zoom: f32,
    texture: Option<egui::TextureHandle>,
    max_iterations: i32,
    pro_que: Arc<ProQue>,
    camera: Camera,
    t_gpu: Duration,
    auto_rotate: bool,
    rotation_angle: f32,
}

impl SpongeApp {
    pub fn new() -> Self {
        let pro_que = ProQue::builder()
            .src(TETHRAEDRON_KERNEL_F32)
            .build()
            .expect("OpenCL build failed");
         
        Self {
            center: (0.0, 0.0),
            zoom: 1.0,
            texture: None,
            max_iterations: 2, // Adjusted default iterations to be within slider range
            pro_que: Arc::new(pro_que),
            camera: Camera {
                position: [3.0, 3.0, 3.0],
                target: [0.0, 0.0, 0.0],
            },
            t_gpu: Duration::ZERO,
            auto_rotate: false,
            rotation_angle: 0.0,
        }
    }

    fn update_camera_rotation(&mut self) {
        if self.auto_rotate {
            self.rotation_angle += 0.01;
            let radius = 4.0;
            self.camera.position[0] = radius * self.rotation_angle.cos();
            self.camera.position[2] = radius * self.rotation_angle.sin();
            self.camera.position[1] = 2.0; // Keep some height
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

        
        let kernel = self.pro_que.kernel_builder("menger_3d_surface_render")
            .arg(&buffer)
            .arg(width as i32)
            .arg(height as i32)
            .arg(self.center.0)
            .arg(self.center.1)
            .arg(self.zoom)
            .arg(self.camera.position[0])
            .arg(self.camera.position[1])
            .arg(self.camera.position[2])
            .arg(self.camera.target[0])
            .arg(self.camera.target[1])
            .arg(self.camera.target[2])
            .arg(self.max_iterations)
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
        self.render_gpu(w, h)
    }
}

impl App for SpongeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update camera rotation if auto-rotate is enabled
        self.update_camera_rotation();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = egui::Vec2::new(512.0, 512.0);
            let size = [available_size.x as usize, available_size.y as usize];
            
            // UI Controls
            ui.horizontal(|ui| {
                ui.label("Iterations:");
                let old_iter = self.max_iterations;
                // Changed max range from 1024 to 30 for better performance and visibility
                ui.add(egui::Slider::new(&mut self.max_iterations, 1..=10).logarithmic(true)); 
                if old_iter != self.max_iterations {
                    self.texture = None; // Force redraw if iterations changed
                }
                
                ui.separator();
                
                if ui.checkbox(&mut self.auto_rotate, "Auto Rotate").changed() {
                    self.texture = None;
                }
                
                ui.separator();
                
                ui.label(format!("Zoom: {:.2}x", self.zoom));
                ui.label(format!("Center: ({:.3}, {:.3})", self.center.0, self.center.1));
                
                if ui.button("Reset View").clicked() {
                    self.center = (0.0, 0.0);
                    self.zoom = 1.0;
                    self.camera.position = [3.0, 3.0, 3.0];
                    self.camera.target = [0.0, 0.0, 0.0];
                    self.rotation_angle = 0.0;
                    self.texture = None;
                }
            });
            
            // Camera controls
            ui.horizontal(|ui| {
                ui.label("Camera:");
                if ui.button("Front").clicked() {
                    self.camera.position = [0.0, 0.0, 4.0];
                    self.camera.target = [0.0, 0.0, 0.0];
                    self.texture = None;
                }
                if ui.button("Side").clicked() {
                    self.camera.position = [4.0, 0.0, 0.0];
                    self.camera.target = [0.0, 0.0, 0.0];
                    self.texture = None;
                }
                if ui.button("Top").clicked() {
                    self.camera.position = [0.0, 4.0, 0.001];//correction factor along z to avoid parallel world up vector underflow
                    self.camera.target = [0.0, 0.0, 0.0];
                    self.texture = None;
                }
                if ui.button("Diagonal").clicked() {
                    self.camera.position = [3.0, 3.0, 3.0];
                    self.camera.target = [0.0, 0.0, 0.0];
                    self.texture = None;
                }
            });
            
            // Menger sponge rendering
            if self.texture.is_none() || self.texture.as_ref().unwrap().size() != size || self.auto_rotate {
                let img = self.render(size[0], size[1]);
                self.texture = Some(ui.ctx().load_texture("menger_sponge", img, Default::default()));
            }

            if let Some(tex) = &self.texture {
                let response = ui.image(tex);              
                let mut need_redraw = false;

                // Pan with arrow keys
                let pan_speed = 0.1 / self.zoom;
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
                            // Apply zoom
                            let zoom_delta = 1.1f32.powf(scroll_delta_y * 0.1);
                            self.zoom *= zoom_delta;
                            
                            need_redraw = true;
                        }
                    }
                } else {
                    // Handle dragging for camera rotation
                    if (drag_x != 0.0 || drag_y != 0.0) && ctx.input(|i| i.pointer.is_decidedly_dragging()) {
                        // Simple camera rotation around target
                        let sensitivity = 0.01;
                        self.rotation_angle += drag_x * sensitivity;
                        
                        let radius = (self.camera.position[0].powi(2) + self.camera.position[2].powi(2)).sqrt();
                        self.camera.position[0] = radius * self.rotation_angle.cos();
                        self.camera.position[2] = radius * self.rotation_angle.sin();
                        
                        // Vertical rotation
                        self.camera.position[1] = (self.camera.position[1] - drag_y * sensitivity * 2.0).clamp(-5.0, 5.0);
                        
                        need_redraw = true;
                    }
                }

                if need_redraw {
                    self.texture = None;
                }
            }
            
            ui.label(format!("GPU time: {:?}", self.t_gpu));
        });
        
        // Request continuous repaints if auto-rotating
        if self.auto_rotate {
            ctx.request_repaint();
        }
    }
}

fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions {
        ..Default::default()
    };

    native_options.viewport.inner_size = Some((512.0+16.0, 512.0+80.0).into());

    eframe::run_native(
        "Menger Sponge 3D",
        native_options,
        Box::new(|_cc| Ok(Box::new(SpongeApp::new()))),
    )
}
