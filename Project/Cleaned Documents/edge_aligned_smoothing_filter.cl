__kernel void edge_aligned_smoothing_filter(
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
    float angle = etf_map[idx];

    float sum = 0.0f;
    float weight_sum = 0.0f;
    int half_size = kernel_size / 2;
    
    // Step size for integration (you may need to adjust this)
    float step_size = 2;

    // Trace in the positive direction of the ETF
    for (float step = 0; step <= half_size; step += step_size) {
        int x = j + step * cos(angle);
        int y = i + step * sin(angle);

        if (x >= 0 && x < width && y >= 0 && y < height) {
            float weight = exp(-0.5 * (step * step) / (sigma * sigma));
            sum += image[y * width + x] * weight;
            weight_sum += weight;
        } else {
            break;
        }
    }

    // Trace in the negative direction of the ETF
    for (float step = -step_size; step >= -half_size; step -= step_size) {
        int x = j + step * cos(angle);
        int y = i + step * sin(angle);

        if (x >= 0 && x < width && y >= 0 && y < height) {
            float weight = exp(-0.5 * (step * step) / (sigma * sigma));
            sum += image[y * width + x] * weight;
            weight_sum += weight;
        } else {
            break;
        }
    }

    output[idx] = sum / weight_sum;
}
