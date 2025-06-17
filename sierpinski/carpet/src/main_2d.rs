use eframe::egui;
use egui::{Color32, Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};

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

#[derive(Debug, Clone, Copy)]
struct Square {
    top_left: Point,
    size: f32,
    is_filled: bool,
}

impl Square {
    fn new(top_left: Point, size: f32, is_filled: bool) -> Self {
        Square { top_left, size, is_filled }
    }

    fn get_rect(&self) -> Rect {
        Rect::from_min_size(
            self.top_left.to_pos2(),
            Vec2::new(self.size, self.size)
        )
    }

    fn subdivide(&self) -> [Square; 9] {
        let third_size = self.size / 3.0;
        let mut squares = [Square::new(Point::new(0.0, 0.0), 0.0, false); 9];
        
        for i in 0..3 {
            for j in 0..3 {
                let x = self.top_left.x + (i as f32) * third_size;
                let y = self.top_left.y + (j as f32) * third_size;
                let index = i * 3 + j;
                
                // The center square (index 4) should be empty, others filled
                let is_filled = index != 4;
                squares[index] = Square::new(Point::new(x, y), third_size, is_filled);
            }
        }
        
        squares
    }
}

struct SierpinskiCarpet {
    squares: Vec<Square>,
    center: Point,
    initial_size: f32,
    show_controls: bool,
    iteration: u32,
}

impl Default for SierpinskiCarpet {
    fn default() -> Self {
        Self {
            squares: Vec::new(),
            center: Point::new(512.0, 512.0),
            initial_size: 512.0,
            show_controls: true,
            iteration: 0,
        }
    }
}

impl SierpinskiCarpet {
    fn new() -> Self {
        let mut carpet = Self::default();
        carpet.initialize_carpet();
        carpet
    }

    fn initialize_carpet(&mut self) {
        // Create initial square
        let half_size = self.initial_size / 2.0;
        let top_left = Point::new(
            self.center.x - half_size,
            self.center.y - half_size
        );
        
        self.squares = vec![Square::new(top_left, self.initial_size, true)];
        self.iteration = 0;
    }

    fn add_sierpinski_squares(&mut self) {
        let mut new_squares = Vec::new();
        
        for square in &self.squares {
            if square.is_filled {
                // For filled squares, subdivide into 9 squares (8 filled + 1 empty center)
                let subdivided = square.subdivide();
                new_squares.extend(subdivided);
            } else {
                // Keep empty squares as they are
                new_squares.push(*square);
            }
        }
        
        self.squares = new_squares;
        self.iteration += 1;
    }

    fn reset_carpet(&mut self) {
        self.initialize_carpet();
    }

    fn draw_carpet(&self, painter: &Painter) {
        for square in &self.squares {
            let rect = square.get_rect();
            
            if square.is_filled {
                // Draw filled square with solid blue fill, no outline
                painter.rect_filled(
                    rect,
                    egui::Rounding::ZERO,
                    Color32::from_rgb(0, 120, 200), // Solid blue fill
                );
            } else {
                // Draw empty square with dark background, no outline
                painter.rect_filled(
                    rect,
                    egui::Rounding::ZERO,
                    Color32::from_rgb(15, 15, 25), // Dark fill matching background
                );
            }
        }

        
    }
}

impl eframe::App for SierpinskiCarpet {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel for controls
        if self.show_controls {
            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Sierpinski Carpet - Iteration: {}", self.iteration));
                    
                    ui.separator();
                    
                    if ui.button("Add Empty Squares (Next Iteration)").clicked() {
                        self.add_sierpinski_squares();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Reset").clicked() {
                        self.reset_carpet();
                    }
                    
                    ui.separator();
                    
                    ui.label("Initial Size:");
                    if ui.add(egui::Slider::new(&mut self.initial_size, 256.0..=768.0)).changed() {
                        self.reset_carpet();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Hide Controls").clicked() {
                        self.show_controls = false;
                    }
                });
                
                ui.label("Click 'Add Empty Squares' or click anywhere on the canvas to grow the Sierpinski carpet!");
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
            
            // Handle mouse clicks - add squares on any click
            let response = ui.allocate_rect(available_rect, egui::Sense::click());
            
            if response.clicked() {
                self.add_sierpinski_squares();
            }

            // Get painter for custom drawing
            let painter = ui.painter();
            
            // Draw background
            painter.rect_filled(
                available_rect,
                egui::Rounding::ZERO,
                Color32::from_rgb(15, 15, 25), // Dark background
            );

            // Draw the Sierpinski carpet
            self.draw_carpet(&painter);

            // Show information
            let filled_count = self.squares.iter().filter(|s| s.is_filled).count();
            let empty_count = self.squares.iter().filter(|s| !s.is_filled).count();
            
            let info_text = format!(
                "Iteration: {} | Filled: {} | Empty: {} | Total: {} | Click to subdivide",
                self.iteration, filled_count, empty_count, self.squares.len()
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
                    "Click anywhere to start creating the Sierpinski carpet fractal!",
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
            .with_title("Sierpinski Carpet - Click to add empty squares"),
        ..Default::default()
    };

    eframe::run_native(
        "Sierpinski Carpet",
        options,
        Box::new(|_cc| {
            Ok(Box::new(SierpinskiCarpet::new()))
        }),
    )
}