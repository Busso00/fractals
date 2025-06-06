#include <emscripten.h>

// High precision Mandelbrot calculation using long double
int mandelbrot_dd(long double re, long double im, int max_iter) {
    long double zr = 0.0L;
    long double zi = 0.0L;
    long double zr_sq, zi_sq;
    int iter = 0;
    
    // Optimization: precompute squares to avoid redundant calculations
    while (iter < max_iter) {
        zr_sq = zr * zr;
        zi_sq = zi * zi;
        
        // Check if point has escaped (|z|² > 4)
        if (zr_sq + zi_sq > 4.0L) {
            break;
        }
        
        // z = z² + c
        // (a + bi)² = a² - b² + 2abi
        long double new_zr = zr_sq - zi_sq + re;
        long double new_zi = 2.0L * zr * zi + im;
        
        zr = new_zr;
        zi = new_zi;
        iter++;
    }
    
    return iter;
}

int mandelbrot_double(double re, double im, int max_iter) {
    double zr = 0.0;
    double zi = 0.0;
    double zr_sq, zi_sq;
    int iter = 0;

    while (iter < max_iter) {
        zr_sq = zr * zr;
        zi_sq = zi * zi;

        if (zr_sq + zi_sq > 4.0) {
            break;
        }

        double new_zr = zr_sq - zi_sq + re;
        double new_zi = 2.0 * zr * zi + im;

        zr = new_zr;
        zi = new_zi;
        iter++;
    }

    return iter;
}

// Exported function to JavaScript (wasm)
EMSCRIPTEN_KEEPALIVE
int mandelbrot(double re, double im, int max_iter) {
    return mandelbrot_double(re, im, max_iter);
}