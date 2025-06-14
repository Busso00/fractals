use eframe::egui;
use egui::{Color32, Pos2};
use rand::Rng;

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

struct BarnsleyFern {
    points: Vec<Point>,
    max_points: usize,
    show_controls: bool,
}

impl Default for BarnsleyFern {
    fn default() -> Self {
        Self {
            points: vec![Point::new(0.0, 0.0)],
            max_points: 100_000,
            show_controls: true,
        }
    }
}

impl BarnsleyFern {
    fn new() -> Self {
        Self::default()
    }

    fn iterate(&mut self, n: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..n {
            let last = *self.points.last().unwrap();
            let r: f64 = rng.r#gen();

            let next = if r < 0.01 {
                Point::new(0.0, 0.16 * last.y)
            } else if r < 0.86 {
                Point::new(
                    0.85 * last.x + 0.04 * last.y,
                    -0.04 * last.x + 0.85 * last.y + 1.6,
                )
            } else if r < 0.93 {
                Point::new(
                    0.20 * last.x - 0.26 * last.y,
                    0.23 * last.x + 0.22 * last.y + 1.6,
                )
            } else {
                Point::new(
                    -0.15 * last.x + 0.28 * last.y,
                    0.26 * last.x + 0.24 * last.y + 0.44,
                )
            };

            self.points.push(next);
            if self.points.len() > self.max_points {
                self.points.remove(0);
            }
        }
    }

    fn draw(&self, painter: &egui::Painter, rect: egui::Rect) {
        let scale = 60.0;
        let offset_x = rect.center().x;
        let offset_y = rect.bottom();

        for p in &self.points {
            let x = offset_x + (p.x as f32 * scale);
            let y = offset_y - (p.y as f32 * scale);
            painter.circle_filled(Pos2::new(x, y), 0.5, Color32::GREEN);
        }
    }
}

impl eframe::App for BarnsleyFern {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.show_controls {
            egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Add 10,000 Points").clicked() {
                        self.iterate(10_000);
                    }
                    if ui.button("Reset").clicked() {
                        self.points = vec![Point::new(0.0, 0.0)];
                    }
                    if ui.button("Hide Controls").clicked() {
                        self.show_controls = false;
                    }
                    ui.label(format!("Points: {}", self.points.len()));
                });
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if !self.show_controls {
                ui.horizontal(|ui| {
                    if ui.button("Show Controls").clicked() {
                        self.show_controls = true;
                    }
                });
            }

            let rect = ui.max_rect();
            let response = ui.allocate_rect(rect, egui::Sense::click());

            if response.clicked() {
                self.iterate(10_000);
            }

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
            .with_inner_size([800.0, 1000.0])
            .with_title("Barnsley Fern"),
        ..Default::default()
    };

    eframe::run_native(
        "Barnsley Fern",
        options,
        Box::new(|_cc| Ok(Box::new(BarnsleyFern::new()))),
    )
}
