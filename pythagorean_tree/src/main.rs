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

struct PythagorasTree {
    order: u32,
    max_order: u32,
    base_length: f32,
    show_controls: bool,
    mouse_x: f32,
    ground_level: f32
}

impl Default for PythagorasTree {
    fn default() -> Self {
        Self {
            order: 1,
            max_order: 12,
            base_length: 100.0,
            show_controls: true,
            mouse_x: 512.0,
            ground_level: 1024.0 - 180.0
        }
    }
}

impl PythagorasTree {
    fn get_color_for_depth(&self, depth: u32) -> Color32 {
        // Create a smooth transition from brown (trunk) to green (leaves)
        let max_depth = self.max_order;
        let progress = (max_depth - depth) as f32 / max_depth as f32;
        
        // Brown color (trunk): RGB(101, 67, 33)
        let brown_r = 101.0;
        let brown_g = 67.0;
        let brown_b = 33.0;
        
        // Green color (leaves): RGB(34, 139, 34)
        let green_r = 34.0;
        let green_g = 139.0;
        let green_b = 34.0;
        
        // Interpolate between brown and green
        let r = (brown_r + (green_r - brown_r) * progress) as u8;
        let g = (brown_g + (green_g - brown_g) * progress) as u8;
        let b = (brown_b + (green_b - brown_b) * progress) as u8;
        
        Color32::from_rgb(r, g, b)
    }

    fn draw_tree(
        &self,
        p: Point,
        size: f32,
        angle: f32,
        painter: &Painter,
        depth: u32,
        a: f32,
        b: f32,
        left: bool,
    ) {
        if depth == 0 {
            return;
        }
        
        let mut p1 = p; // Bottom-left point
        let mut p2 = Point::new(p.x + size * angle.cos(), p.y - size * angle.sin());
        let mut p3 = Point::new(p2.x - size * angle.sin(), p2.y - size * angle.cos());
        let mut p4 = Point::new(p1.x - size * angle.sin(), p1.y - size * angle.cos());
        
        if !left {
            p1 = p1;
            p2 = Point::new(p.x - size * angle.cos(), p.y + size * angle.sin());
            p3 = Point::new(p2.x - size * angle.sin(), p2.y - size * angle.cos());
            p4 = Point::new(p1.x - size * angle.sin(), p1.y - size * angle.cos());
        }

        let color = self.get_color_for_depth(depth);

        painter.add(egui::Shape::convex_polygon(
            vec![p1.to_pos2(), p2.to_pos2(), p3.to_pos2(), p4.to_pos2()],
            color,
            Stroke::new(1.0, Color32::from_rgb(40, 25, 15)), // Dark brown outline
        ));

        let hyp = (a * a + b * b).sqrt();
        if hyp == 0.0 {
            return;
        }

        let left_size = size * b / hyp;
        let right_size = size * a / hyp;

        let alpha = a.atan2(b);
        let beta = b.atan2(a);

        if left {
            self.draw_tree(p4, left_size, angle + alpha, painter, depth - 1, a, b, true);
            self.draw_tree(p3, right_size, angle - beta, painter, depth - 1, a, b, false); 
        } else {
            self.draw_tree(p3, left_size, angle + alpha, painter, depth - 1, a, b, true);
            self.draw_tree(p4, right_size, angle - beta, painter, depth - 1, a, b, false); 
        }
    }

    fn draw(&self, painter: &Painter, screen_width: f32, screen_height: f32, ctx: &egui::Context) {
        // Update mouse-based ratio for skew
        let mouse_x = ctx
            .input(|i| i.pointer.hover_pos().map(|p| p.x))
            .unwrap_or(self.mouse_x);
        let ratio = ((mouse_x - screen_width / 2.0) / (screen_width / 2.0)).clamp(-1.0, 1.0);

        // Position root at the bottom of the screen
        // Adjusted ground_level to make the terrain taller (start higher on screen)
        let root_size = self.base_length;
        
        // Offset for the tree and scene to move it slightly to the right
        let scene_offset_x = 50.0; 
        let root_x = screen_width / 2.0 - root_size / 2.0 + scene_offset_x;
        let root_y = self.ground_level;

        let root_point = Point::new(root_x, root_y);

        let skew_angle_for_legs = PI / 4.0 + ratio * PI / 4.0;

        let leg_a = skew_angle_for_legs.cos();
        let leg_b = skew_angle_for_legs.sin();

        // Draw tree
        self.draw_tree(
            root_point, root_size, 0.0, painter, self.order, leg_a, leg_b, false,
        );
    }

    fn increment_order(&mut self) {
        if self.order < self.max_order {
            self.order += 1;
        }
    }

    fn reset(&mut self) {
        self.order = 1;
    }
}

impl eframe::App for PythagorasTree {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.show_controls {
            egui::TopBottomPanel::top("controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Pythagoras Tree - Order {}", self.order));
                    if ui.button("Next Order").clicked() {
                        self.increment_order();
                    }
                    if ui.button("Reset").clicked() {
                        self.reset();
                    }
                    if ui.button("Hide Controls").clicked() {
                        self.show_controls = false;
                    }
                });
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if !self.show_controls {
                if ui.button("Show Controls").clicked() {
                    self.show_controls = true;
                }
            }

            let available_rect = ui.available_rect_before_wrap();
            self.mouse_x = ctx.input(|i| i.pointer.hover_pos().map(|p| p.x).unwrap_or(512.0));

            let painter = ui.painter();
            
            // Draw gradient background (sky)
            let gradient_steps = 50;
            let step_height = available_rect.height() / gradient_steps as f32;
            
            for i in 0..gradient_steps {
                let progress = i as f32 / (gradient_steps - 1) as f32;
                let r = (240.0 + (135.0 - 240.0) * progress) as u8;
                let g = (248.0 + (206.0 - 248.0) * progress) as u8;
                let b = (255.0 + (235.0 - 255.0) * progress) as u8;
                
                let color = Color32::from_rgb(r, g, b);
                let y_start = available_rect.top() + i as f32 * step_height;
                let y_end = y_start + step_height + 1.0; // Small overlap to avoid gaps
                
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        Pos2::new(available_rect.left(), y_start),
                        Pos2::new(available_rect.right(), y_end)
                    ),
                    egui::Rounding::ZERO,
                    color,
                );

                if y_start + step_height >= self.ground_level{
                    painter.rect_filled(
                    egui::Rect::from_min_max(
                        Pos2::new(available_rect.left(), y_start),
                        Pos2::new(available_rect.right(), y_end)
                    ),
                    egui::Rounding::ZERO,
                    Color32::from_rgb(139, 119, 85), // Sandy brown terrain color
                    );
                }
                
    
            }

            self.draw(painter, available_rect.width(), available_rect.height(), ctx);
            

            let info = format!(
                "Order: {} | Mouse X: {:.1} (Tree skew)",
                self.order, self.mouse_x
            );
            painter.text(
                Pos2::new(available_rect.left() + 10.0, available_rect.bottom() - 20.0),
                egui::Align2::LEFT_BOTTOM,
                info,
                egui::FontId::default(),
                Color32::from_rgb(40, 25, 15), // Dark brown text
            );

            if ui
                .allocate_rect(available_rect, egui::Sense::click())
                .clicked()
            {
                self.increment_order();
            }
        });

        ctx.request_repaint();
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 1024.0])
            .with_title("Pythagoras Tree - Natural Scene"),
        ..Default::default()
    };

    eframe::run_native(
        "Pythagoras Tree",
        options,
        Box::new(|_cc| Ok(Box::new(PythagorasTree::default()))),
    )
}
