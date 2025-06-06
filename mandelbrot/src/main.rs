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

#define ZERO (dd) {0.0, 0.0}
#define ONE (dd){1.0, 0.0}
#define BAILOUT (dd){1e20, 0.0}
#define STEP (dd){6.0, 0.0}


// A double-double type for high precision calculations
typedef struct { double hi, lo; } dd;

// New: A struct to represent a complex number using double-double components
typedef struct { dd x, y; } dd2;


// Split a double into high/low for Dekker's algorithm
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

// Multiply two double-doubles using Dekker's algorithm and FMA (fused multiply-add)
static inline dd dd_mul(dd a, dd b) {
    double p = a.hi * b.hi;
    // Calculate error term for high-precision product
    double err = fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
    double z = p + err;
    return (dd){ z, err - (z - p) };
}

// Divide two double-doubles
static inline dd dd_div(dd a, dd b) {
    double q1 = a.hi / b.hi;

    // r = a - b*q1 (high precision remainder)
    dd q1b = dd_mul(dd_from_double(q1), b);
    dd r = dd_add(a, (dd){ -q1b.hi, -q1b.lo });

    double q2 = r.hi / b.hi; // Second term of the quotient

    double result_hi = q1 + q2;
    double result_lo = q2 - (result_hi - q1); // Error correction for the low part

    return (dd){ result_hi, result_lo };
}

// Complex number multiplication (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
static inline dd2 dd_complex_mul(dd2 a, dd2 b) {
    dd real_part = dd_sub(dd_mul(a.x, b.x), dd_mul(a.y, b.y));
    dd imag_part = dd_add(dd_mul(a.x, b.y), dd_mul(a.y, b.x));
    return (dd2){real_part, imag_part};
}

// Complex number addition (a + bi) + (c + di) = (a+c) + i(b+d)
static inline dd2 dd_complex_add(dd2 a, dd2 b) {
    dd real_part = dd_add(a.x, b.x);
    dd imag_part = dd_add(a.y, b.y);
    return (dd2){real_part, imag_part};
}

// Complex number division (a + bi) / (c + di) = (ac + bd)/(c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
static inline dd2 dd_complex_div(dd2 num, dd2 den) {
    dd den_sq_re = dd_mul(den.x, den.x);
    dd den_sq_im = dd_mul(den.y, den.y);
    dd denominator = dd_add(den_sq_re, den_sq_im); // c^2 + d^2

    // Numerator real part: ac + bd
    dd num_real_term1 = dd_mul(num.x, den.x);
    dd num_real_term2 = dd_mul(num.y, den.y);
    dd num_real_part = dd_add(num_real_term1, num_real_term2);

    // Numerator imag part: bc - ad
    dd num_imag_term1 = dd_mul(num.y, den.x);
    dd num_imag_term2 = dd_mul(num.x, den.y);
    dd num_imag_part = dd_sub(num_imag_term1, num_imag_term2);
    
    dd result_real = dd_div(num_real_part, denominator);
    dd result_imag = dd_div(num_imag_part, denominator);

    return (dd2){result_real, result_imag};
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

static inline dd dd_sqrt(dd a) {
    if (a.hi == 0.0) return (dd){0.0, 0.0};

    double x = sqrt(a.hi);  // Initial approximation
    dd dx = {x, 0.0};
    dd x2 = dd_mul(dx, dx);
    dd r = dd_sub(a, x2);
    dd correction = dd_div(r, dd_add(dx, dx));  // r / (2x)
    return dd_add(dx, correction);
}

static inline dd dd_sin(dd a) {
    double s = sin(a.hi);
    double c = cos(a.hi);
    double err = a.lo * c;
    return (dd){s + err, err - (s + err - s)};
}

static inline dd dd_cos(dd a) {
    double c = cos(a.hi);
    double s = sin(a.hi);
    double err = -a.lo * s;
    return (dd){c + err, err - (c + err - c)};
}

static inline dd dd_ln(dd a) {
    if (a.hi <= 0.0) return (dd){NAN, NAN};  // log domain error

    double x = log(a.hi);
    dd dx = {x, 0.0};
    dd exp_dx = {exp(x), 0.0};
    dd r = dd_div(a, exp_dx);      // a / exp(dx)
    dd correction = dd_sub(r, (dd){1.0, 0.0});
    return dd_add(dx, correction);
}

static inline dd dd_floor(dd a) {
    double flo = floor(a.hi);  // Get floor of the high part

    dd result = {flo, 0.0};
    
    if (flo > a.hi || (flo == a.hi && a.lo < 0.0)) {
        result.hi -= 1.0;
    }

    return result;
}

static inline dd dd_exp(dd a) {
    // Constants
    const double LOG2E = 1.4426950408889634;     // 1/ln(2)
    const double LN2_HI = 0.6931471805599453;    // hi part of ln(2)
    const double LN2_LO = 2.3190468138462996e-17; // lo part of ln(2)

    // Convert to base-2 exponent: a * log2(e)
    dd k_real = dd_mul(a, (dd){LOG2E, 0.0});
    int k = (int)(k_real.hi + (k_real.hi >= 0.0 ? 0.5 : -0.5));  // round to nearest int

    // r = a - k * ln(2)
    dd k_dd = (dd){(double)k, 0.0};
    dd ln2 = (dd){LN2_HI, LN2_LO};
    dd t = dd_mul(k_dd, ln2);
    dd r = dd_sub(a, t);  // reduced argument

    // Use a Taylor expansion to approximate exp(r)
    dd term = (dd){1.0, 0.0};
    dd sum = term;
    for (int i = 1; i <= 10; ++i) {
        term = dd_mul(term, r);
        term = dd_div(term, (dd){(double)i, 0.0});
        sum = dd_add(sum, term);
    }

    // Recompose: exp(a) ≈ exp(r) * 2^k
    sum.hi = ldexp(sum.hi, k);
    sum.lo = ldexp(sum.lo, k);
    return sum;
}

// Structure to hold Mandelbrot iteration results, including derivatives
typedef struct {
    int iter;
    dd zr;  // Final real part of Z
    dd zi;  // Final imaginary part of Z
    dd dzr; // Real part of dz/dc (derivative of Z with respect to C)
    dd dzi; // Imaginary part of dz/dc
} mandelbrot_result;


// Modified mandelbrot_iter function to return final Z values AND its derivative
static inline mandelbrot_result mandelbrot_iter(dd cre, dd cim, int max_iter) {
    dd zr = dd_from_double(0.0);
    dd zi = dd_from_double(0.0);
    
    dd dzr = dd_from_double(1.0); // Real part of dz/dc
    dd dzi = dd_from_double(0.0); // Imaginary part of dz/dc
    int iter = 0;
    
    while (iter < max_iter) {
        dd zr2 = dd_mul(zr, zr);
        dd zi2 = dd_mul(zi, zi);
        dd mag2 = dd_add(zr2, zi2);
        
        if (mag2.hi > BAILOUT.hi) break;
        
        dd term_real = dd_sub(dd_mul(zr, dzr), dd_mul(zi, dzi));
        dd term_imag = dd_add(dd_mul(zr, dzi), dd_mul(zi, dzr));

        dd new_dzr = dd_add(dd_mul(dd_from_double(2.0), term_real), dd_from_double(1.0));
        dd new_dzi = dd_mul(dd_from_double(2.0), term_imag);
        
        dzr = new_dzr;
        dzi = new_dzi;

        dd tmp = dd_add(zr2, (dd){-zi2.hi, -zi2.lo});
        dd zr_new = dd_add(tmp, cre);
        
        
        dd prod = dd_mul(zr, zi);
        prod = dd_add(prod, prod);
        dd zi_new = dd_add(prod, cim);
        
        zr = zr_new;
        zi = zi_new;
        iter++;
    }
    
    mandelbrot_result result;
    result.iter = iter;
    result.zr = zr;
    result.zi = zi;
    result.dzr = dzr; // Store the final dzr
    result.dzi = dzi; // Store the final dzi
    return result;
}


static inline uchar3 get_potential_differential_shadow(mandelbrot_result m_res, dd cre, dd cim, int max_iter, dd scale) {
    if (m_res.iter == max_iter) {
        return (uchar3){0, 0, 0};
    }

    // Use dd constants
    dd h2 = (dd){1.5, 0.0};
    dd angle = (dd){45.0, 0.0};
    dd pi = (dd){M_PI, 0.0};
    dd deg_to_rad = dd_div(pi, (dd){180.0, 0.0});
    dd angle_rad = dd_mul(angle, deg_to_rad);

    dd v_re = dd_cos(angle_rad);
    dd v_im = dd_sin(angle_rad);

    dd den_sq_re = dd_mul(m_res.dzr, m_res.dzr);
    dd den_sq_im = dd_mul(m_res.dzi, m_res.dzi);
    dd denominator = dd_add(den_sq_re, den_sq_im);

    if (denominator.hi == 0.0 && denominator.lo == 0.0) {
        return (uchar3){255, 255, 255};
    }

    dd u_re_num_term1 = dd_mul(m_res.zr, m_res.dzr);
    dd u_re_num_term2 = dd_mul(m_res.zi, m_res.dzi);
    dd u_re_numerator = dd_add(u_re_num_term1, u_re_num_term2);
    dd u_re = dd_div(u_re_numerator, denominator);

    dd u_im_num_term1 = dd_mul(m_res.zi, m_res.dzr);
    dd u_im_num_term2 = dd_mul(m_res.zr, m_res.dzi);
    dd u_im_numerator = dd_sub(u_im_num_term1, u_im_num_term2);
    dd u_im = dd_div(u_im_numerator, denominator);

    // Compute abs(u)^2 = u_re^2 + u_im^2
    dd u_re_sq = dd_mul(u_re, u_re);
    dd u_im_sq = dd_mul(u_im, u_im);
    dd abs_u_sq = dd_add(u_re_sq, u_im_sq);
    dd abs_u = dd_sqrt(abs_u_sq);

    // Normalize u: u / |u|
    dd u_re_norm = dd_div(u_re, abs_u);
    dd u_im_norm = dd_div(u_im, abs_u);

    // t = dot(u_norm, v) + h2
    dd t_dot_re = dd_mul(u_re_norm, v_re);
    dd t_dot_im = dd_mul(u_im_norm, v_im);
    dd t_sum = dd_add(t_dot_re, t_dot_im);
    dd t = dd_add(t_sum, h2);

    // Normalize t: t = t / (1 + h2)
    dd one = (dd){1.0, 0.0};
    dd denom = dd_add(one, h2);
    t = dd_div(t, denom);

    // Clamp t to [0, 1]
    if (t.hi < 0.0) {
        t = ZERO;
    }

    // Convert to 0–255 grayscale
    dd scale255 = (dd){255.0, 0.0};
    dd color_val = dd_mul(t, scale255);
    uchar val = (uchar)(color_val.hi);  // truncate to uchar

    return (uchar3){val, val, val};
}

static inline uchar3 get_smooth_shadow(mandelbrot_result m_res, dd cre, dd cim, int max_iter, dd scale) {
    if (m_res.iter == max_iter) {
        return (uchar3){0, 0, 0};
    }

    
    dd smooth_iter = (dd){(double)m_res.iter, 0.0};
    smooth_iter = dd_add(smooth_iter, ONE);
    dd l2 = dd_ln((dd){2.0, 0.0});
    dd l_bailout = dd_ln(BAILOUT);
    dd sqrt_z = dd_add(dd_mul(m_res.zr, m_res.zr), dd_mul(m_res.zi,m_res.zi));
    smooth_iter = dd_add(smooth_iter, dd_div(dd_ln(dd_div(dd_ln(sqrt_z),l_bailout)),l2));

    dd periodic_smoothing = dd_sub(smooth_iter, dd_mul(dd_floor(dd_div(smooth_iter, STEP)), STEP));
    dd t = dd_gt(periodic_smoothing , ONE) ? ONE: periodic_smoothing;

    dd scale255 = (dd){255.0, 0.0};
    dd color_val = dd_mul(t, scale255);
    uchar val = (uchar)(color_val.hi); 
    return (uchar3){val, val, val};
}

static inline uchar3 get_color(mandelbrot_result m_res, dd cre, dd cim, int max_iter, dd scale) {
    if (m_res.iter == max_iter) {
        return (uchar3){0, 0, 0};
    }
    
    float t = log2((float)(m_res.iter/(int)STEP.hi + 1));
    float pi2 = 6.28318f;
    
    float r = (0.5f + 0.5f * cos(pi2 * t)) * 255.0f;
    float g = (0.5f + 0.5f * cos(pi2 * t + 2.09439f)) * 255.0f;
    float b = (0.5f + 0.5f * cos(pi2 * t + 4.18879f)) * 255.0f;
    
    return (uchar3){(uchar)r, (uchar)g, (uchar)b};
}


static inline uchar3 blend(uchar3 c1, uchar3 c2, float t) {
    float3 fc1 = convert_float3(c1);
    float3 fc2 = convert_float3(c2);

    float3 blended = (1.0f - t) * fc1 + t * fc2;

    return convert_uchar3_sat_rte(blended); // Round to nearest, clamp to [0,255]
}

static inline uchar3 blend_aggressive(uchar3 c1, uchar3 c2){
    float3 fc1 = convert_float3(c1);
    float3 fc2 = convert_float3(c2);

    float3 blended = fc1*fc2/(float3)(255.0f, 255.0f, 255.0f);

    return convert_uchar3_sat_rte(blended);
}



// Modified kernel call to pass scale parameter
__kernel void mandelbrot_dd_improved(
    __global uchar* output,
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

    dd fx = dd_add(dd_div(x, width), (dd) {-0.5,0.0});
    fx = dd_mul(fx, scale);
    dd fy = dd_add(dd_div(y, height), (dd) {-0.5,0.0});
    fy = dd_mul(fy, scale);

    dd cre = dd_add(center_x, fx);
    dd cim = dd_add(center_y, fy);

    mandelbrot_result m_res = mandelbrot_iter(cre, cim, max_iter);
    uchar3 color1 = get_potential_differential_shadow(m_res, cre, cim, max_iter, scale);
    uchar3 color2 = get_smooth_shadow(m_res, cre, cim, max_iter, scale);
    uchar3 color3 = get_color(m_res, cre, cim, max_iter, scale);

    uchar3 color12 = blend_aggressive(color1, color2);
    uchar3 color = blend_aggressive(color12, color3);

    output[idx * 3] = color.x;
    output[idx * 3 + 1] = color.y;
    output[idx * 3 + 2] = color.z;
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
        
        let kernel = self.pro_que.kernel_builder("mandelbrot_dd_improved")
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