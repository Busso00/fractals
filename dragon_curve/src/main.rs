use eframe::egui;
use egui::{Color32, Pos2, Stroke};

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

struct DragonCurve {
    points: Vec<Point>,
    current_iteration: usize,
}

impl Default for DragonCurve {
    fn default() -> Self {
        Self {
            points: vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)], // Initial segment
            current_iteration: 0,
        }
    }
}

impl DragonCurve {
    fn new() -> Self {
        Self::default()
    }

    fn generate_iteration(&mut self) {
        

        let mut new_points = self.points.clone();
        let len = self.points.len();

        let last_point = *self.points.last().unwrap();

        for i in (0..(len - 1)).rev() {
            let p = self.points[i];
            // Translate point to origin relative to last_point
            let translated_x = p.x - last_point.x;
            let translated_y = p.y - last_point.y;

            // Rotate 90 degrees clockwise: (x, y) -> (y, -x)
            let rotated_x = translated_y;
            let rotated_y = -translated_x;

            // Translate back
            new_points.push(Point::new(rotated_x + last_point.x, rotated_y + last_point.y));
        }

        self.points = new_points;
        self.current_iteration += 1;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn draw(&self, painter: &egui::Painter, rect: egui::Rect) {
        let max_dim = (rect.width().min(rect.height()) * 0.8) as f64; // Use 80% of min dimension
        let (min_x, max_x, min_y, max_y) = self.points.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
            |(mx, M_x, my, M_y), p| {
                (mx.min(p.x), M_x.max(p.x), my.min(p.y), M_y.max(p.y))
            },
        );

        let range_x = max_x - min_x;
        let range_y = max_y - min_y;

        let scale = if range_x == 0.0 && range_y == 0.0 {
            1.0 // Avoid division by zero for initial single point
        } else if range_x == 0.0 {
            max_dim / range_y
        } else if range_y == 0.0 {
            max_dim / range_x
        } else {
            max_dim / range_x.max(range_y)
        };


        let offset_x = rect.center().x as f64 - ((min_x + max_x) / 2.0) * scale;
        let offset_y = rect.center().y as f64 - ((min_y + max_y) / 2.0) * scale;


        for i in 0..(self.points.len() - 1) {
            let p1 = self.points[i];
            let p2 = self.points[i + 1];

            let screen_p1 = Pos2::new(
                (offset_x + p1.x * scale) as f32,
                (offset_y + p1.y * scale) as f32,
            );
            let screen_p2 = Pos2::new(
                (offset_x + p2.x * scale) as f32,
                (offset_y + p2.y * scale) as f32,
            );
            painter.line_segment([screen_p1, screen_p2], Stroke::new(1.0, Color32::YELLOW));
        }
    }
}

impl eframe::App for DragonCurve {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Increase Iteration").clicked() {
                        self.current_iteration += 1;
                        self.generate_iteration();
                        
                    }
                    if ui.button("Decrease Iteration").clicked() {
                        if self.current_iteration > 0 {
                            self.current_iteration -= 1;
                            self.generate_iteration();
                        }
                    }
                    
                    if ui.button("Reset").clicked() {
                        self.reset();
                    }
                    ui.label(format!("Current Iteration: {}", self.current_iteration));
                   
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {

            let rect = ui.max_rect();
            let painter = ui.painter();
            painter.rect_filled(rect, egui::Rounding::ZERO, Color32::from_rgb(10, 10, 30));
            self.draw(painter, rect);
        });



        ctx.request_repaint(); // continually repaint for smooth updates
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 800.0]) // Adjusted size for better view
            .with_title("Dragon Curve Fractal"),
        ..Default::default()
    };

    eframe::run_native(
        "Dragon Curve Fractal",
        options,
        Box::new(|_cc| Ok(Box::new(DragonCurve::new()))),
    )
}