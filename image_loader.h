#pragma once

#include "bmp.h"
#include <stdio.h>
#include <stdlib.h>

unsigned char* loadImage(const char *fileNama, size_t &size)
{
    FILE *inptr = fopen(fileNama, "r");
    if (inptr == NULL)
    {
        fprintf(stderr, "Could not open %s.\n", fileNama);
        return NULL;
    }

    // read infile's BITMAPFILEHEADER
    BITMAPFILEHEADER bf;
    fread(&bf, sizeof(BITMAPFILEHEADER), 1, inptr);

    // read infile's BITMAPINFOHEADER
    BITMAPINFOHEADER bi;
    fread(&bi, sizeof(BITMAPINFOHEADER), 1, inptr);

    // ensure infile is (likely) a 24-bit uncompressed BMP 4.0
    if (bf.bfType != 0x4d42 || bf.bfOffBits != 54 || bi.biSize != 40 ||
        bi.biBitCount != 24 || bi.biCompression != 0)
    {
        fclose(inptr);
        fprintf(stderr, "Unsupported file format.\n");
        return NULL;
    }

    // determine padding for scanlines
    int padding = (4 - (bi.biWidth * sizeof(RGBTRIPLE)) % 4) % 4;
    
    size = abs(bi.biHeight) * bi.biWidth;
    unsigned char* data = (unsigned char*)malloc(size);

    // iterate over infile's scanlines
    for (int i = 0, biHeight = abs(bi.biHeight); i < biHeight; i++)
    {
        fread(data + i, sizeof(unsigned char), bi.biWidth, inptr);
        // skip over padding, if any
        fseek(inptr, padding, SEEK_CUR);
    }
    return data;
}
