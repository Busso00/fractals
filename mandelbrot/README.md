# Mandelbrot rendering
- main OpenCL uses OpenCL on Linux/MinGW relying respectively on f128::f128/std::f128
- sol_double_linux is a bug fix to compare f128::f128 variables
- implementation on Rayon is similar in performances with GPU version only when using f64
- easiest version to run is main_rayon (openCL require to manually install libraries .a)
- quadmath is not supported in MSVC and can be used only with Linux/MinGW
- mingw must run with nightly toolchain

## Additional features
### GSLS f128 implementation with uvec4 
- from f32
- add
- sub
- mul
- div left to the developer

# Mandelbrot rendering with 3d-effect in main

![Alt text](img/0.png)
![Alt text](img/1.png)
![Alt text](img/2.png)
![Alt text](img/3.png)
![Alt text](img/4.png)
![Alt text](img/5.png)
![Alt text](img/6.png)
![Alt text](img/7.png)
![Alt text](img/8.png)
![Alt text](img/9.png)
![Alt text](img/10.png)