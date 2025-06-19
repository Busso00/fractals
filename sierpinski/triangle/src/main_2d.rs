use eframe::egui;
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }

    fn to_pos2(self) -> Pos2 {
        Pos2::new(self.x, self.y)
    }

    fn midpoint(self, other: Point) -> Point {
        Point::new((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct Triangle {
    p1: Point,
    p2: Point,
    p3: Point,
    is_filled: bool,
}

impl Triangle {
    fn new(p1: Point, p2: Point, p3: Point, is_filled: bool) -> Self {
        Triangle { p1, p2, p3, is_filled }
    }

    fn get_center_triangle(&self) -> Triangle {
        // Create the center triangle by connecting midpoints
        let mid12 = self.p1.midpoint(self.p2);
        let mid23 = self.p2.midpoint(self.p3);
        let mid31 = self.p3.midpoint(self.p1);
        
        Triangle::new(mid12, mid23, mid31, false) // Center triangle is always empty
    }

    fn get_corner_triangles(&self) -> [Triangle; 3] {
        let mid12 = self.p1.midpoint(self.p2);
        let mid23 = self.p2.midpoint(self.p3);
        let mid31 = self.p3.midpoint(self.p1);
        
        [
            Triangle::new(self.p1, mid12, mid31, true), // Top triangle
            Triangle::new(mid12, self.p2, mid23, true), // Bottom left triangle
            Triangle::new(mid31, mid23, self.p3, true), // Bottom right triangle
        ]
    }
}

struct SierpinskiTriangle {
    triangles: Vec<Triangle>,
    center: Point,
    initial_size: f32,
    show_controls: bool,
    iteration: u32,
}

impl Default for SierpinskiTriangle {
    fn default() -> Self {
        Self {
            triangles: Vec::new(),
            center: Point::new(512.0, 512.0),
            initial_size: 512.0,
            show_controls: true,
            iteration: 0,
        }
    }
}

impl SierpinskiTriangle {
    fn new() -> Self {
        let mut triangle = Self::default();
        triangle.initialize_triangle();
        triangle
    }

    fn initialize_triangle(&mut self) {
        // Create initial equilateral triangle (pointing up)
        let size = self.initial_size;
        let height = size * (3.0_f32.sqrt() / 2.0);
        
        let p1 = Point::new(self.center.x, self.center.y - height / 2.0); // Top
        let p2 = Point::new(self.center.x - size / 2.0, self.center.y + height / 2.0); // Bottom left
        let p3 = Point::new(self.center.x + size / 2.0, self.center.y + height / 2.0); // Bottom right
        
        self.triangles = vec![Triangle::new(p1, p2, p3, true)];
        self.iteration = 0;
    }

    fn add_sierpinski_triangles(&mut self) {
        let mut new_triangles = Vec::new();
        
        for triangle in &self.triangles {
            if triangle.is_filled {
                // For filled triangles, subdivide into 3 corner triangles + 1 empty center triangle
                let corner_triangles = triangle.get_corner_triangles();
                new_triangles.extend(corner_triangles);
                
                let center_triangle = triangle.get_center_triangle();
                new_triangles.push(center_triangle);
            } else {
                // Keep empty triangles as they are
                new_triangles.push(*triangle);
            }
        }
        
        self.triangles = new_triangles;
        self.iteration += 1;
    }

    fn reset_triangle(&mut self) {
        self.initialize_triangle();
    }

    fn draw_triangle(&self, painter: &Painter) {
        let filled_stroke = Stroke::new(1.5, Color32::LIGHT_BLUE);
        let empty_stroke = Stroke::new(1.0, Color32::from_rgb(100, 100, 100));
        
        for triangle in &self.triangles {
            let points = [
                triangle.p1.to_pos2(),
                triangle.p2.to_pos2(),
                triangle.p3.to_pos2(),
            ];
            
            if triangle.is_filled {
                // Draw filled triangle with light blue outline
                painter.add(egui::Shape::convex_polygon(
                    points.to_vec(),
                    Color32::from_rgb(0, 120, 200), // Semi-transparent blue fill
                    filled_stroke,
                ));
            } else {
                // Draw empty triangle with gray outline only
                painter.add(egui::Shape::convex_polygon(
                    points.to_vec(),
                    Color32::TRANSPARENT, // No fill
                    empty_stroke,
                ));
            }
        }

    }
}

impl eframe::App for SierpinskiTriangle {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel for controls
        if self.show_controls {
            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Sierpinski Triangle - Iteration: {}", self.iteration));
                    
                    ui.separator();
                    
                    if ui.button("Add Empty Triangles (Next Iteration)").clicked() {
                        self.add_sierpinski_triangles();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Reset").clicked() {
                        self.reset_triangle();
                    }
                    
                    ui.separator();
                    
                    ui.label("Initial Size:");
                    if ui.add(egui::Slider::new(&mut self.initial_size, 512.0..=768.0)).changed() {
                        self.reset_triangle();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Hide Controls").clicked() {
                        self.show_controls = false;
                    }
                });
                
                ui.label("Click 'Add Empty Triangles' or click anywhere on the canvas to grow the Sierpinski triangle!");
            });
        }

        // Main central panel for drawing
        egui::CentralPanel::default().show(ctx, |ui| {
            // Show controls toggle when hidden
            if !self.show_controls {
                ui.horizontal(|ui| {
                    if ui.button("Show Controls").clicked() {
                        self.show_controls = true;
                    }
                });
            }

            // Get the available space for drawing
            let available_rect = ui.available_rect_before_wrap();
            
            // Handle mouse clicks - add triangles on any click
            let response = ui.allocate_rect(available_rect, egui::Sense::click());
            
            if response.clicked() {
                self.add_sierpinski_triangles();
            }

            // Get painter for custom drawing
            let painter = ui.painter();
            
            // Draw background
            painter.rect_filled(
                available_rect,
                egui::Rounding::ZERO,
                Color32::from_rgb(15, 15, 25), // Dark background
            );

            // Draw the Sierpinski triangle
            self.draw_triangle(&painter);

            // Show information
            let filled_count = self.triangles.iter().filter(|t| t.is_filled).count();
            let empty_count = self.triangles.iter().filter(|t| !t.is_filled).count();
            
            let info_text = format!(
                "Iteration: {} | Filled: {} | Empty: {} | Total: {} | Click to subdivide",
                self.iteration, filled_count, empty_count, self.triangles.len()
            );
            let text_pos = Pos2::new(available_rect.left() + 10.0, available_rect.bottom() - 30.0);
            painter.text(
                text_pos,
                egui::Align2::LEFT_BOTTOM,
                info_text,
                egui::FontId::default(),
                Color32::WHITE,
            );

            // Instructions
            if self.iteration == 0 {
                let instruction_pos = Pos2::new(available_rect.center().x, available_rect.top() + 80.0);
                painter.text(
                    instruction_pos,
                    egui::Align2::CENTER_CENTER,
                    "Click anywhere to start creating the Sierpinski triangle fractal!",
                    egui::FontId::proportional(16.0),
                    Color32::YELLOW,
                );
            }
        });

        // Request repaint for smooth interaction
        ctx.request_repaint();
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 1024.0])
            .with_title("Sierpinski Triangle - Click to add empty triangles"),
        ..Default::default()
    };

    eframe::run_native(
        "Sierpinski Triangle",
        options,
        Box::new(|cc| {
            Ok(Box::new(SierpinskiTriangle::new()))
        }),
    )
}
