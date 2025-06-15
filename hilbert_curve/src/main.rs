use eframe::egui;
use egui::{Color32, Pos2, Stroke, Rect, Vec2, Slider, CentralPanel, TopBottomPanel, Shape};

struct HilbertCurveApp {
    order: u32,
    points: Vec<Pos2>,
}

impl Default for HilbertCurveApp {
    fn default() -> Self {
        Self {
            order: 3,
            points: Vec::new(),
        }
    }
}

impl HilbertCurveApp {
    fn new() -> Self {
        Self::default()
    }

    fn rot(n: u32, mut x: u32, mut y: u32, rx: u32, ry: u32) -> (u32, u32) {
        if ry == 0 {
            if rx == 1 {
                x = (1 << n) - 1 - x;
                y = (1 << n) - 1 - y;
            }
            let t = x;
            x = y;
            y = t;
        }
        (x, y)
    }

    fn d2xy(n: u32, d: u32) -> (u32, u32) {
        let mut x = 0;
        let mut y = 0;
        let mut t = d;

        for i in 0..n {
            let rx = 1 & (t / 2);
            let ry = 1 & (t ^ rx);

            let (new_x, new_y) = Self::rot(i, x, y, rx, ry);
            x = new_x;
            y = new_y;

            x += rx * (1 << i);
            y += ry * (1 << i);

            t /= 4;
        }
        (x, y)
    }

    fn generate_hilbert_points(&mut self, rect: Rect) {
        let n = self.order;
        let num_points = 1 << (2 * n);

        self.points.clear();
        if n == 0 {
            self.points.push(rect.center());
            return;
        }
        self.points.reserve(num_points as usize);

        let mut min_logical_x = f32::MAX;
        let mut max_logical_x = f32::MIN;
        let mut min_logical_y = f32::MAX;
        let mut max_logical_y = f32::MIN;

        let mut temp_logical_points = Vec::with_capacity(num_points as usize);

        for i in 0..num_points {
            let (x, y) = Self::d2xy(n, i);
            let logical_x = x as f32;
            let logical_y = y as f32;

            temp_logical_points.push(Pos2::new(logical_x, logical_y));

            min_logical_x = min_logical_x.min(logical_x);
            max_logical_x = max_logical_x.max(logical_x);
            min_logical_y = min_logical_y.min(logical_y);
            max_logical_y = max_logical_y.max(logical_y);
        }

        let curve_width = (max_logical_x - min_logical_x).max(1.0);
        let curve_height = (max_logical_y - min_logical_y).max(1.0);

        let scale_x = rect.width() / curve_width;
        let scale_y = rect.height() / curve_height;
        let final_scale = scale_x.min(scale_y) * 0.95;

        let center_logical_x = (min_logical_x + max_logical_x) / 2.0;
        let center_logical_y = (min_logical_y + max_logical_y) / 2.0;

        let offset_x = rect.center().x - center_logical_x * final_scale;
        let offset_y = rect.center().y - center_logical_y * final_scale;

        for p in temp_logical_points {
            self.points.push(Pos2::new(
                offset_x + p.x * final_scale,
                offset_y + p.y * final_scale,
            ));
        }
    }

    fn draw(&self, painter: &egui::Painter) {
        if self.points.len() > 1 {
            painter.add(Shape::line(
                self.points.clone(),
                Stroke::new(1.5, Color32::WHITE),
            ));
        } else if let Some(p) = self.points.first() {
            painter.circle(*p, 2.0, Color32::WHITE, Stroke::new(0.0, Color32::TRANSPARENT));
        }
    }
}

impl eframe::App for HilbertCurveApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label("Hilbert Curve Order (n): ");
                ui.add(Slider::new(&mut self.order, 1..=8).text("Order"));
                ui.add_space(20.0);
                ui.label(format!("Current Order: {}", self.order));
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            let rect = ui.max_rect();
            let painter = ui.painter();

            painter.rect_filled(rect, 0.0, Color32::from_rgb(10, 10, 30));
            self.generate_hilbert_points(rect);
            self.draw(painter);
        });

        ctx.request_repaint();
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(Vec2::new(1000.0, 800.0))
            .with_title("Hilbert Curve Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "Hilbert Curve Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(HilbertCurveApp::new()))),
    )
}