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


__kernel void mandelbrot_dd(
    __global int* output,   // iteration count
    const int width_s,
    const int height_s,
    const double center_x_hi,
    const double center_y_hi,
    const double scale_hi,
    const double center_x_lo,
    const double center_y_lo,
    const double scale_lo,
    const int max_iter
) {
    int x_s = get_global_id(0);
    int y_s = get_global_id(1);
    int idx = y_s * width_s + x_s;

    dd center_x = (dd){ center_x_hi, center_x_lo };
    dd center_y = (dd){ center_y_hi, center_y_lo };
    dd scale = (dd){ scale_hi, scale_lo };
    dd x = dd_from_double((double) x_s);
    dd y = dd_from_double((double) y_s);
    dd width = dd_from_double((double) width_s);
    dd height = dd_from_double((double) height_s);

    if (dd_ge(x,width) || dd_ge(y,height)) return;


    // compute c = center + (pixel/size - 0.5)*scale
    dd fx = dd_add(dd_div(x, width), (dd) {-0.5,0.0});
    fx = dd_mul(fx, scale);
    dd fy = dd_add(dd_div(y, height), (dd) {-0.5,0.0});
    fy = dd_mul(fy, scale);

    dd cre = dd_add(center_x, fx);
    dd cim = dd_add(center_y, fy);

    dd zr = dd_from_double(0.0);
    dd zi = dd_from_double(0.0);

    int iter = 0;
    while (iter < max_iter) {
        // zr2 = zr*zr, zi2 = zi*zi
        dd zr2 = dd_mul(zr, zr);
        dd zi2 = dd_mul(zi, zi);

        dd mag2 = dd_add(zr2, zi2);

        // test |z|^2 > 4  ⇒ zr2+zi2 > 4
       
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

    output[idx] = iter;
}
"#;



fn get_color(iter: i32, max_iter: i32) -> [u8; 3] {
    if iter == max_iter { return [0, 0, 0]; }
    let t = (iter % 256) as f32 / 256.0;

    let r = (0.5 + 0.5 * (6.28318 * t).cos()) * 255.0;
    let g = (0.5 + 0.5 * (6.28318 * t + 2.09439).cos()) * 255.0;
    let b = (0.5 + 0.5 * (6.28318 * t + 4.18879).cos()) * 255.0;
    [r as u8, g as u8, b as u8]
}

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
            center: (f128::from(-0.5), f128::from(0.0)),
            zoom: f128::from(1.0),
            texture: None,
            max_iterations: 2048,
            pro_que: Arc::new(pro_que),
            t_mt: Duration::ZERO,
            t_gpu: Duration::ZERO
        }
    }

       
    /// Renders the *entire* image in one OpenCL dispatch.
    fn render_gpu(&mut self, width: usize, height: usize) -> ColorImage {
        let t0 = Instant::now();

        let total_pixels = width * height;
        // 1) Create one big output buffer

        let buffer = Buffer::<i32>::builder()
            .queue(self.pro_que.queue().clone())
            .len(total_pixels)
            .build()
            .expect("Buffer build");
        // 2) Build kernel with all args

        
        
        let (center_x_hi, center_x_lo) = split_f128_to_dd(self.center.0);
        let (center_y_hi, center_y_lo) = split_f128_to_dd(self.center.1);
        let (scale_hi, scale_lo)       = split_f128_to_dd(f128::from(4.0) / self.zoom);
        let kernel = self.pro_que.kernel_builder("mandelbrot_dd")
            .arg(&buffer)
            .arg(width as i32)
            .arg(height as i32)
            .arg(center_x_hi)
            .arg(center_y_hi)
            .arg(scale_hi)
            .arg(center_x_lo)
            .arg(center_y_lo)
            .arg(scale_lo)           // scale
            .arg(self.max_iterations as i32)
            .build()
            .expect("Kernel build");
        // 3) Enqueue it over the full 2D range
        unsafe {
            kernel.cmd()
                .global_work_size([width, height])
                .local_work_size([16, 16])   // 16×16 = 256 threads per group
                .enq()
                .expect("Kernel enqueue");
        }
        // 4) Read back all pixels
        let mut raw = vec![0i32; total_pixels];
        buffer.read(&mut raw).enq().expect("Read buffer");
        // 5) Convert to egui::Color32
        let pixels = raw.into_iter()
            .map(|v| {
                let it = v as i32;
                let [r,g,b] = get_color(it, self.max_iterations);
                egui::Color32::from_rgb(r,g,b)
            })
            .collect();

        println!("time elapsed:{:?}",t0.elapsed());
        
        self.t_gpu = t0.elapsed();
        
        ColorImage { size: [width, height], pixels }
        
    }

    fn render(&mut self, w: usize, h:usize ) -> ColorImage{
        //avoid overloading
        if self.t_gpu > self.t_mt {
            //render with multi-thread CPU
            return self.render_gpu(w, h);
        }else{
            //render with GPU
            return self.render_gpu(w, h);
        }
    }
}



impl App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = egui::Vec2::new(512.0, 512.0);
            let size = [available_size.x as usize, available_size.y as usize];
            // Pan with drag (improved)
              
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
                    self.center = (f128::from(-0.5), f128::from(0.0));
                    self.zoom = f128::from(1.0);
                    self.texture = None;
                }
            });
            
            // Mandelbrot rendering
            // Redraw image if window size changes or parameters change
            if self.texture.is_none() || self.texture.as_ref().unwrap().size() != size {
                let img = self.render(size[0], size[1]);
                self.texture = Some(ui.ctx().load_texture("mandelbrot", img, Default::default()));
            }

            if let Some(tex) = &self.texture {
                let response = ui.image(tex);              
                let mut need_redraw = false;

                // Pan with arrow keys (always available)
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
                
                // Zoom with scroll wheel (anywhere on image)
                let scroll_delta = ctx.input(|i| i.raw_scroll_delta);
                //let scroll_delta_x = scroll_delta.x; //scroll x is not detected by OS
                let scroll_delta_y = scroll_delta.y;     
                let drag = ctx.input(|i| i.pointer.delta());
                let drag_x = drag.x;//dreg x is not detected by OS  
                let drag_y = drag.y;//dreg y is not detected by OS 

                if scroll_delta_y != 0.0 {
                    // Get cursor position for zooming toward cursor
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
                                
                                // Adjust center to keep mouse position fixed on the same fractal point
                                let new_scale = f128::from(4.0) / self.zoom;
                                self.center.0 = mouse_re - (rel_x - f128::from(0.5)) * new_scale;
                                self.center.1 = mouse_im - (rel_y - f128::from(0.5)) * new_scale;
                                
                                need_redraw = true;
                            }
                        }
                } else {
                    // Not scrolling

                    if (drag_x != 0.0 || drag_y != 0.0) &&  ctx.input(|i| i.pointer.is_decidedly_dragging()) {
                        let scale = f128::from(4.0) / self.zoom;
                        // Convert f32 drag values to f64 for calculation
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
    // Just use the default options since the exact window size field names have changed
    // between eframe versions
    let mut native_options = eframe::NativeOptions {
        ..Default::default()
    };

    native_options.viewport.inner_size = Some((512.0+16.0, 512.0+40.0).into());

    eframe::run_native(
        "Mandelbrot",
        native_options,
        Box::new(|_cc| Box::new(MandelbrotApp::new())),
    )
}
