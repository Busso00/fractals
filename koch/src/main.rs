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
}

struct KochSnowflake {
    edges: Vec<(Point, Point)>,
    center: Point,
    initial_size: f32,
    show_controls: bool,
    iteration: u32,
}

impl Default for KochSnowflake {
    fn default() -> Self {
        Self {
            edges: Vec::new(),
            center: Point::new(512.0, 512.0), // Center of 800x600 window
            initial_size: 512.0,
            show_controls: true,
            iteration: 0,
        }
    }
}

impl KochSnowflake {
    fn new() -> Self {
        let mut snowflake = Self::default();
        snowflake.initialize_triangle();
        snowflake
    }

    fn initialize_triangle(&mut self) {
        // Create initial equilateral triangle (pointing up)
        let size = self.initial_size;
        let height = size * (3.0_f32.sqrt() / 2.0);
        
        let p1 = Point::new(self.center.x, self.center.y - height * 2.0 / 3.0); // Top
        let p2 = Point::new(self.center.x - size / 2.0, self.center.y + height / 3.0); // Bottom left
        let p3 = Point::new(self.center.x + size / 2.0, self.center.y + height / 3.0); // Bottom right
        
        self.edges = vec![
            (p1, p2), // Top to bottom-left
            (p2, p3), // Bottom-left to bottom-right
            (p3, p1), // Bottom-right to top
        ];
        self.iteration = 0;
    }

    fn add_koch_triangles(&mut self) {
        let mut new_edges = Vec::new();
        
        for &(p1, p2) in &self.edges {
            // Apply Koch curve transformation to this edge
            let koch_edges = self.koch_curve_edges(p1, p2);
            new_edges.extend(koch_edges);
        }
        
        self.edges = new_edges;
        self.iteration += 1;
    }

    fn koch_curve_edges(&self, p1: Point, p2: Point) -> Vec<(Point, Point)> {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;

        // Divide the line into three equal parts
        let a = Point::new(p1.x + dx / 3.0, p1.y + dy / 3.0);
        let b = Point::new(p1.x + 2.0 * dx / 3.0, p1.y + 2.0 * dy / 3.0);

        // Create the peak of the equilateral triangle (pointing outward)
        let angle = dy.atan2(dx) + PI / 3.0; // Changed from - to + to point outward
        let length = ((dx * dx + dy * dy).sqrt()) / 3.0;
        let c = Point::new(
            a.x + length * angle.cos(),
            a.y + length * angle.sin(),
        );

        // Return the four edges that replace the original edge
        vec![
            (p1, a), // First third
            (a, c),  // Left side of triangle
            (c, b),  // Right side of triangle
            (b, p2), // Last third
        ]
    }

    fn reset_snowflake(&mut self) {
        self.initialize_triangle();
    }

    fn draw_snowflake(&self, painter: &Painter) {
        let stroke = Stroke::new(2.0, Color32::LIGHT_BLUE);
        
        // Draw all edges
        for &(p1, p2) in &self.edges {
            painter.line_segment([p1.to_pos2(), p2.to_pos2()], stroke);
        }

        
    }
}

impl eframe::App for KochSnowflake {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel for controls
        if self.show_controls {
            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Koch Snowflake - Iteration: {}", self.iteration));
                    
                    ui.separator();
                    
                    if ui.button("Add Triangles (Next Iteration)").clicked() {
                        self.add_koch_triangles();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Reset").clicked() {
                        self.reset_snowflake();
                    }
                    
                    ui.separator();
                    
                    ui.label("Initial Size:");
                    if ui.add(egui::Slider::new(&mut self.initial_size, 512.0..=768.0)).changed() {
                        self.reset_snowflake();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Hide Controls").clicked() {
                        self.show_controls = false;
                    }
                });
                
                ui.label("Click 'Add Triangles' or click anywhere on the canvas to grow the Koch snowflake!");
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
                self.add_koch_triangles();
            }

            // Get painter for custom drawing
            let painter = ui.painter();
            
            // Draw background
            painter.rect_filled(
                available_rect,
                egui::Rounding::ZERO,
                Color32::from_rgb(10, 10, 30), // Dark blue background
            );

            // Draw the Koch snowflake
            self.draw_snowflake(&painter);

            // Show information
            let info_text = format!(
                "Iteration: {} | Edges: {} | Click anywhere to add triangles",
                self.iteration,
                self.edges.len()
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
                let instruction_pos = Pos2::new(available_rect.center().x, available_rect.top() + 50.0);
                painter.text(
                    instruction_pos,
                    egui::Align2::CENTER_CENTER,
                    "Click anywhere to start growing the Koch snowflake!",
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
            .with_title("Koch Snowflake - Click to grow the fractal"),
        ..Default::default()
    };

    eframe::run_native(
        "Koch Snowflake",
        options,
        Box::new(|cc| {
            Ok(Box::new(KochSnowflake::new()))
        }),
    )
}
