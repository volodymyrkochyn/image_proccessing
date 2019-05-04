#include "image_loader.h"
#include <time.h>

int sum(const unsigned char* image, size_t size)
{
    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_ms;
    
    start_t = clock();
    int sum = 0;
    for (size_t i = 0; i < size; i+=3)
    {
        sum += image[i];
    }
    end_t = clock();

    clock_delta = end_t - start_t;
    clock_delta_ms = ((double)clock_delta/CLOCKS_PER_SEC)*1000;
    printf("Sum time, ms\t %.4f \t\n", clock_delta_ms);
    return sum;
}

int main()
{
    const char *file = "111.bmp";
    size_t size = 0;
    unsigned char* image = loadImage(file, size);
    
    printf("Image size: %zu\n", size);
    printf("Pixel sum: %d\n", sum(image, size));
    return 0;
}
