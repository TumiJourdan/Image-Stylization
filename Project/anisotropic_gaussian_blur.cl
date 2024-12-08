float gaussian(float x, float sigma) {
    return (1.0 / sqrt(2.0 * M_PI * sigma * sigma)) * exp(- (x * x) / (2.0 * sigma * sigma));
}

#define MAX_KERNEL_SIZE 1201 
__kernel void anisotropic_gaussian_blur(
    __global const float *image,
    __global const float *etf_map,
    int height,
    int width,
    int kernel_size,
    float sigma,
    __global float *output)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= height || j >= width)
        return;

    int idx = i * width + j;
    float angle = etf_map[idx] + M_PI / 2.0;  // Rotate by 90 degrees

    float kernel1[MAX_KERNEL_SIZE];
    for (int k = 0; k < kernel_size; k++) {
        //kernel1[k] = exp(-0.5 * pow( (float)((k - kernel_size / 2) / sigma), (float)(2.0) ));
        kernel1[k] = gaussian(k - kernel_size / 2,sigma);
    }

    float sum = 0.0f;
    float kernel_sum = 0.0f;
    float sin_angle = sin(angle);
    float cos_angle = cos(angle);

    for (int k = 0; k < kernel_size; k++) {
        int x = j + (k - kernel_size / 2) * cos_angle;
        int y = i + (k - kernel_size / 2) * sin_angle;

        if (x >= 0 && x < width && y >= 0 && y < height) {
            sum += image[y * width + x] * kernel1[k];
            kernel_sum += kernel1[k];
        }
    }

    output[idx] = sum / kernel_sum;
}