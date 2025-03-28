#include <emmintrin.h>
#include <unordered_set>
#include <chrono>

#include "PlaneExtractor.h"
#include "triangle.h"

PlaneExtractor::PlaneExtractor(const ConfigData& config)
        : m_config(config)
{

}

PlaneExtractor::~PlaneExtractor() {
    _mm_free(sobelDescLeft);
    sobelDescLeft = nullptr;

    _mm_free(sobelDescRight);
    sobelDescRight = nullptr;
}

void PlaneExtractor::resetMembers() {
    if (sobelDescLeft) {
        _mm_free(sobelDescLeft);
        sobelDescLeft = nullptr;
    }
    if (sobelDescRight) {
        _mm_free(sobelDescRight);
        sobelDescRight = nullptr;
    }

    supportPoints.clear();
    triangles.clear();
    clusters.clear();
    mainClusters.clear();
    planes.clear();
}

void PlaneExtractor::runPipeline(cv::Mat& imLeft, cv::Mat& imRight) {
    correctDistortion(imLeft, imRight);

    performSobelFiltering(imLeft, sobelDescLeft);
    performSobelFiltering(imRight, sobelDescRight);

    computeMatches();

    applyBilateralFilter();

    generateMesh();

    clusterTriangles();

    mergeClusters();

    computePlaneParameters();
}

void PlaneExtractor::correctDistortion(cv::Mat& imLeft, cv::Mat& imRight) {
    cv::remap(imLeft, imLeft, m_config.M1l, m_config.M2l, cv::INTER_LINEAR);
    cv::remap(imRight, imRight, m_config.M1r, m_config.M2r, cv::INTER_LINEAR);

    cv::cvtColor(imLeft, canvas, cv::COLOR_GRAY2RGB);
}

void PlaneExtractor::performSobelFiltering(cv::Mat& im, uint8_t*& sobelDesc) {
    int32_t bpl = m_config.width + 15 - (m_config.width - 1) % 16;
    uint8_t* I = (uint8_t*)_mm_malloc(bpl * m_config.height * sizeof(uint8_t), 16);
    memset(I, 0, bpl * m_config.height * sizeof(uint8_t));

    if (bpl == m_config.width) {
        memcpy(I, im.data, bpl * m_config.height * sizeof(uint8_t));
    } else {
        for (int32_t v = 0; v < m_config.height; v++) {
            memcpy(I + v * bpl, im.data + v * m_config.width, m_config.width * sizeof(uint8_t));
        }
    }

    sobelDesc = (uint8_t*)_mm_malloc(16 * m_config.width * m_config.height * sizeof(uint8_t), 16);
    uint8_t* I_du = (uint8_t*)_mm_malloc(bpl * m_config.height * sizeof(uint8_t), 16);
    uint8_t* I_dv = (uint8_t*)_mm_malloc(bpl * m_config.height * sizeof(uint8_t), 16);
    sobel3x3(I, I_du, I_dv, bpl, m_config.height);
    createDescriptor(sobelDesc, I_du, I_dv, m_config.width, m_config.height, bpl);

    _mm_free(I_du);
    _mm_free(I_dv);
    _mm_free(I);
}

void PlaneExtractor::sobel3x3(const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h) {
    int16_t* temp_h = (int16_t*)(_mm_malloc(w * h * sizeof(int16_t), 16));
    int16_t* temp_v = (int16_t*)(_mm_malloc(w * h * sizeof(int16_t), 16));

    convolve_cols_3x3(in, temp_v, temp_h, w, h);
    convolve_101_row_3x3_16bit(temp_v, out_v, w, h);
    convolve_121_row_3x3_16bit(temp_h, out_h, w, h);

    _mm_free(temp_h);
    _mm_free(temp_v);
}

void PlaneExtractor::createDescriptor(uint8_t* sobelDesc, uint8_t* I_du, uint8_t* I_dv, int32_t width, int32_t height, int32_t bpl) {
    uint8_t* I_desc_curr;
    uint32_t addr_v0, addr_v1, addr_v2, addr_v3, addr_v4;

    for (int32_t v = 3; v < height - 3; v++) {
        addr_v2 = v * bpl;
        addr_v0 = addr_v2 - 2 * bpl;
        addr_v1 = addr_v2 - 1 * bpl;
        addr_v3 = addr_v2 + 1 * bpl;
        addr_v4 = addr_v2 + 2 * bpl;

        for (int32_t u = 3; u < width - 3; u++) {
            I_desc_curr = sobelDesc + (v * width + u) * 16;
            *(I_desc_curr++) = *(I_du + addr_v0 + u + 0);
            *(I_desc_curr++) = *(I_du + addr_v1 + u - 2);
            *(I_desc_curr++) = *(I_du + addr_v1 + u + 0);
            *(I_desc_curr++) = *(I_du + addr_v1 + u + 2);
            *(I_desc_curr++) = *(I_du + addr_v2 + u - 1);
            *(I_desc_curr++) = *(I_du + addr_v2 + u + 0);
            *(I_desc_curr++) = *(I_du + addr_v2 + u + 0);
            *(I_desc_curr++) = *(I_du + addr_v2 + u + 1);
            *(I_desc_curr++) = *(I_du + addr_v3 + u - 2);
            *(I_desc_curr++) = *(I_du + addr_v3 + u + 0);
            *(I_desc_curr++) = *(I_du + addr_v3 + u + 2);
            *(I_desc_curr++) = *(I_du + addr_v4 + u + 0);
            *(I_desc_curr++) = *(I_dv + addr_v1 + u + 0);
            *(I_desc_curr++) = *(I_dv + addr_v2 + u - 1);
            *(I_desc_curr++) = *(I_dv + addr_v2 + u + 1);
            *(I_desc_curr++) = *(I_dv + addr_v3 + u + 0);
        }
    }
}

void PlaneExtractor::convolve_cols_3x3(const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h) {
    const int w_chunk = w / 16;
    __m128i* i0 = (__m128i*)(in);
    __m128i* i1 = (__m128i*)(in) + w_chunk * 1;
    __m128i* i2 = (__m128i*)(in) + w_chunk * 2;
    __m128i* result_h = (__m128i*)(out_h) + 2 * w_chunk;
    __m128i* result_v = (__m128i*)(out_v) + 2 * w_chunk;
    __m128i* end_input = (__m128i*)(in) + w_chunk * h;

    for (; i2 != end_input; i0++, i1++, i2++, result_v += 2, result_h += 2 ) {
        *result_h = _mm_setzero_si128();
        *(result_h + 1) = _mm_setzero_si128();
        *result_v = _mm_setzero_si128();
        *(result_v + 1) = _mm_setzero_si128();
        __m128i ilo, ihi;
        unpack_8bit_to_16bit(*i0, ihi, ilo);
        unpack_8bit_to_16bit(*i0, ihi, ilo);
        *result_h = _mm_add_epi16(ihi, *result_h);
        *(result_h + 1) = _mm_add_epi16(ilo, *(result_h + 1));
        *result_v = _mm_add_epi16(*result_v, ihi);
        *(result_v + 1) = _mm_add_epi16(*(result_v + 1), ilo);
        unpack_8bit_to_16bit(*i1, ihi, ilo);
        *result_v = _mm_add_epi16(*result_v, ihi);
        *(result_v + 1) = _mm_add_epi16(*(result_v + 1), ilo);
        *result_v = _mm_add_epi16(*result_v, ihi);
        *(result_v + 1) = _mm_add_epi16(*(result_v + 1), ilo);
        unpack_8bit_to_16bit(*i2, ihi, ilo);
        *result_h = _mm_sub_epi16(*result_h, ihi);
        *(result_h + 1) = _mm_sub_epi16(*(result_h + 1), ilo);
        *result_v = _mm_add_epi16(*result_v, ihi);
        *(result_v + 1) = _mm_add_epi16(*(result_v + 1), ilo);
    }
}

void PlaneExtractor::convolve_101_row_3x3_16bit(const int16_t* in, uint8_t* out, int w, int h) {
    const __m128i* i0 = (const __m128i*)(in);
    const int16_t* i2 = in + 2;
    uint8_t* result = out + 1;
    const int16_t* const end_input = in + w * h;
    const size_t blocked_loops = (w * h - 2) / 16;
    __m128i offs = _mm_set1_epi16(128);
    for (size_t i = 0; i != blocked_loops; i++) {
        __m128i result_register_lo;
        __m128i result_register_hi;
        __m128i i2_register;

        i2_register = _mm_loadu_si128((__m128i*)(i2));
        result_register_lo = *i0;
        result_register_lo = _mm_sub_epi16(result_register_lo, i2_register);
        result_register_lo = _mm_srai_epi16(result_register_lo, 2);
        result_register_lo = _mm_add_epi16(result_register_lo, offs);

        i0 += 1;
        i2 += 8;

        i2_register = _mm_loadu_si128((__m128i*)(i2));
        result_register_hi = *i0;
        result_register_hi = _mm_sub_epi16(result_register_hi, i2_register);
        result_register_hi = _mm_srai_epi16(result_register_hi, 2);
        result_register_hi = _mm_add_epi16(result_register_hi, offs);

        i0 += 1;
        i2 += 8;

        pack_16bit_to_8bit_saturate(result_register_lo, result_register_hi, result_register_lo);
        _mm_storeu_si128(((__m128i*)(result)), result_register_lo);

        result += 16;
    }

    for (; i2 < end_input; i2++, result++) {
        *result = ((*(i2 - 2) - *i2) >> 2) + 128;
    }
}

void PlaneExtractor::convolve_121_row_3x3_16bit(const int16_t* in, uint8_t* out, int w, int h) {
    const __m128i* i0 = (const __m128i*)(in);
    const int16_t* i1 = in + 1;
    const int16_t* i2 = in + 2;
    uint8_t* result = out + 1;
    const int16_t* const end_input = in + w * h;
    const size_t blocked_loops = (w * h - 2) / 16;
    __m128i offs = _mm_set1_epi16(128);
    for (size_t i = 0; i != blocked_loops; i++) {
        __m128i result_register_lo;
        __m128i result_register_hi;
        __m128i i1_register;
        __m128i i2_register;

        i1_register = _mm_loadu_si128((__m128i*)(i1));
        i2_register = _mm_loadu_si128((__m128i*)(i2));
        result_register_lo = *i0;
        i1_register = _mm_add_epi16(i1_register, i1_register);
        result_register_lo = _mm_add_epi16(i1_register, result_register_lo);
        result_register_lo = _mm_add_epi16(i2_register, result_register_lo);
        result_register_lo = _mm_srai_epi16(result_register_lo, 2);
        result_register_lo = _mm_add_epi16(result_register_lo, offs);

        i0++;
        i1 += 8;
        i2 += 8;

        i1_register = _mm_loadu_si128((__m128i*)(i1));
        i2_register = _mm_loadu_si128((__m128i*)(i2));
        result_register_hi = *i0;
        i1_register = _mm_add_epi16(i1_register, i1_register);
        result_register_hi = _mm_add_epi16(i1_register, result_register_hi);
        result_register_hi = _mm_add_epi16(i2_register, result_register_hi);
        result_register_hi = _mm_srai_epi16(result_register_hi, 2);
        result_register_hi = _mm_add_epi16(result_register_hi, offs);

        i0++;
        i1 += 8;
        i2 += 8;

        pack_16bit_to_8bit_saturate(result_register_lo, result_register_hi, result_register_lo);
        _mm_storeu_si128(((__m128i*)(result)), result_register_lo);

        result += 16;
    }
}

void PlaneExtractor::unpack_8bit_to_16bit(const __m128i a, __m128i& b0, __m128i& b1) {
    __m128i zero = _mm_setzero_si128();
    b0 = _mm_unpacklo_epi8(a, zero);
    b1 = _mm_unpackhi_epi8(a, zero);
}

void PlaneExtractor::pack_16bit_to_8bit_saturate(const __m128i a0, const __m128i a1, __m128i& b) {
    b = _mm_packus_epi16(a0, a1);
}

void PlaneExtractor::computeMatches() {
    int32_t D_candidate_stepsize = m_config.candidateStepsize;

    int32_t D_can_width = 0;
    int32_t D_can_height = 0;
    for (int32_t u = 0; u < m_config.width; u += D_candidate_stepsize) {
        D_can_width++;
    }
    for (int32_t v = 0; v < m_config.height; v += D_candidate_stepsize) {
        D_can_height++;
    }

    float* D_can = (float*)calloc(D_can_width * D_can_height, sizeof(float));

    int32_t u, v;
    float d, d2;

    for (int32_t u_can = 1; u_can < D_can_width; u_can++) {
        u = u_can * D_candidate_stepsize;
        for (int32_t v_can = 1; v_can < D_can_height; v_can++) {
            v = v_can * D_candidate_stepsize;

            *(D_can + getAddressOffsetImage(u_can, v_can, D_can_width)) = -1;

            d = computeDisparity(u, v, false);
            if (d >= 0) {
                d2 = computeDisparity(u - d, v, true);
                if (d2 >= 0 && abs(d - d2) <= m_config.lrThreshold) {
                    *(D_can + getAddressOffsetImage(u_can, v_can, D_can_width)) = d;
                }
            }
        }
    }

    removeInconsistentSupportPoints(D_can, D_can_width, D_can_height);
    removeRedundantSupportPoints(D_can, D_can_width, D_can_height, 5, 1, true);
    removeRedundantSupportPoints(D_can, D_can_width, D_can_height, 5, 1, false);

    for (int32_t u_can = 1; u_can < D_can_width; u_can++) {
        for (int32_t v_can = 1; v_can < D_can_height; v_can++) {
            if (*(D_can + getAddressOffsetImage(u_can, v_can, D_can_width)) >= 0) {
                SupportPoint sp(u_can * D_candidate_stepsize,
                                v_can * D_candidate_stepsize,
                                *(D_can + getAddressOffsetImage(u_can, v_can, D_can_width)));

                double z = m_config.bf / sp.d;
                double x = (sp.u - m_config.cx) / m_config.fx * z;
                double y = (sp.v - m_config.cy) / m_config.fy * z;
                sp.point3D << x, y, z;
                supportPoints.push_back(sp);
            }
        }
    }

    free(D_can);
}

inline float PlaneExtractor::computeDisparity(const int32_t& u, const int32_t& v, const bool& right_image) {
    const int32_t u_step = 2;
    const int32_t v_step = 2;
    const int32_t window_size = 3;

    int32_t desc_offset_1 = -16 * u_step - 16 * m_config.width * v_step;
    int32_t desc_offset_2 = +16 * u_step - 16 * m_config.width * v_step;
    int32_t desc_offset_3 = -16 * u_step + 16 * m_config.width * v_step;
    int32_t desc_offset_4 = +16 * u_step + 16 * m_config.width * v_step;

    __m128i xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

    if (u >= window_size + u_step && u <= m_config.width - window_size - 1 - u_step &&
        v >= window_size + v_step && v <= m_config.height - window_size - 1 - v_step)
    {
        int32_t line_offset = 16 * m_config.width * v;
        uint8_t *I1_line_addr, *I2_line_addr;

        if (!right_image) {
            I1_line_addr = sobelDescLeft + line_offset;
            I2_line_addr = sobelDescRight + line_offset;
        } else {
            I1_line_addr = sobelDescRight + line_offset;
            I2_line_addr = sobelDescLeft + line_offset;
        }

        uint8_t* I1_block_addr = I1_line_addr + 16 * u;
        uint8_t* I2_block_addr;

        int32_t sum = 0;
        for (int32_t i = 0; i < 16; i++)
            sum += abs((int32_t)(*(I1_block_addr + i)) - 128);
        if (sum < m_config.supportTexture) {
            return -1;
        }

        xmm1 = _mm_load_si128((__m128i*)(I1_block_addr + desc_offset_1));
        xmm2 = _mm_load_si128((__m128i*)(I1_block_addr + desc_offset_2));
        xmm3 = _mm_load_si128((__m128i*)(I1_block_addr + desc_offset_3));
        xmm4 = _mm_load_si128((__m128i*)(I1_block_addr + desc_offset_4));

        int32_t u_warp;

        int16_t min_1_E = 32767;
        int16_t min_1_d = -1;
        int16_t min_2_E = 32767;
        int16_t min_2_d = -1;

        int32_t disp_min_valid = std::max(m_config.dispMin, 0);
        int32_t disp_max_valid = m_config.dispMax;
        if (!right_image) {
            disp_max_valid = std::min(m_config.dispMax, u - window_size - u_step);
        } else {
            disp_max_valid = std::min(m_config.dispMax, m_config.width - u - window_size - u_step);
        }

        if (disp_max_valid - disp_min_valid < 10) {
            return -1;
        }

        for (int16_t d = disp_min_valid; d <= disp_max_valid; d++) {
            if (!right_image) {
                u_warp = u - d;
            } else {
                u_warp = u + d;
            }

            I2_block_addr = I2_line_addr + 16 * u_warp;

            xmm6 = _mm_load_si128((__m128i*)(I2_block_addr + desc_offset_1));
            xmm6 = _mm_sad_epu8(xmm1, xmm6);
            xmm5 = _mm_load_si128((__m128i*)(I2_block_addr + desc_offset_2));
            xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm2, xmm5), xmm6);
            xmm5 = _mm_load_si128((__m128i*)(I2_block_addr + desc_offset_3));
            xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm3, xmm5), xmm6);
            xmm5 = _mm_load_si128((__m128i*)(I2_block_addr + desc_offset_4));
            xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm4, xmm5), xmm6);
            sum  = _mm_extract_epi16(xmm6, 0) + _mm_extract_epi16(xmm6, 4);

            if (sum < min_1_E) {
                min_2_E = min_1_E;
                min_2_d = min_1_d;
                min_1_E = sum;
                min_1_d = d;
            } else if (sum < min_2_E) {
                min_2_E = sum;
                min_2_d = d;
            }
        }

        float ratio_test = static_cast<float>(min_1_E) / static_cast<float>(min_2_E);
        if (ratio_test >= m_config.supportThreshold) {
            return -1;
        }

        if (min_1_d == disp_min_valid || min_1_d == disp_max_valid) {
            return min_1_d;
        }

        auto computeCost = [&](int16_t d) {
            int32_t u_warp = (!right_image) ? (u - d) : (u + d);
            uint8_t* I2_addr = I2_line_addr + 16 * u_warp;

            __m128i x6 = _mm_load_si128((__m128i*)(I2_addr + desc_offset_1));
            x6 = _mm_sad_epu8(xmm1, x6);

            __m128i xtmp = _mm_load_si128((__m128i*)(I2_addr + desc_offset_2));
            x6 = _mm_add_epi16(_mm_sad_epu8(xmm2, xtmp), x6);

            xtmp = _mm_load_si128((__m128i*)(I2_addr + desc_offset_3));
            x6 = _mm_add_epi16(_mm_sad_epu8(xmm3, xtmp), x6);

            xtmp = _mm_load_si128((__m128i*)(I2_addr + desc_offset_4));
            x6 = _mm_add_epi16(_mm_sad_epu8(xmm4, xtmp), x6);

            int32_t cost = _mm_extract_epi16(x6, 0) + _mm_extract_epi16(x6, 4);
            return static_cast<int16_t>(cost);
        };

        int16_t Edm1 = computeCost(min_1_d - 1);
        int16_t Ed = min_1_E;
        int16_t Edp1 = computeCost(min_1_d + 1);

        float denom = static_cast<float>(Edm1 + Edp1 - 2 * Ed);
        if (std::fabs(denom) < 1e-5f) {
            return min_1_d;
        }

        float delta = 0.5f * static_cast<float>(Edm1 - Edp1) / denom;
        if (delta < -1.0f || delta > 1.0f) {
            return min_1_d;
        }

        float final_disp = static_cast<float>(min_1_d) + delta;

        return final_disp;
    } else {
        return -1;
    }
}

void PlaneExtractor::removeInconsistentSupportPoints(float* D_can, int32_t D_can_width, int32_t D_can_height) {
    for (int32_t u_can = 0; u_can < D_can_width; u_can++) {
        for (int32_t v_can = 0; v_can < D_can_height; v_can++) {
            float d_can = *(D_can + getAddressOffsetImage(u_can, v_can, D_can_width));
            if (d_can >= 0) {
                int32_t support = 0;
                for (int32_t u_can_2 = u_can - m_config.inconWindowSize; u_can_2 <= u_can + m_config.inconWindowSize; u_can_2++) {
                    for (int32_t v_can_2 = v_can - m_config.inconWindowSize; v_can_2 <= v_can + m_config.inconWindowSize; v_can_2++) {
                        if (u_can_2 >= 0 && v_can_2 >= 0 && u_can_2 < D_can_width && v_can_2 < D_can_height) {
                            float d_can_2 = *(D_can + getAddressOffsetImage(u_can_2, v_can_2, D_can_width));
                            if (d_can_2 >= 0 && abs(d_can - d_can_2) <= m_config.inconThreshold) {
                                support++;
                            }
                        }
                    }
                }

                if (support < m_config.inconMinSupport) {
                    *(D_can + getAddressOffsetImage(u_can, v_can, D_can_width)) = -1;
                }
            }
        }
    }
}

void PlaneExtractor::removeRedundantSupportPoints(float* D_can, int32_t D_can_width, int32_t D_can_height, int32_t redun_max_dist, int32_t redun_threshold, bool vertical) {
    int32_t redun_dir_u[2] = {0, 0};
    int32_t redun_dir_v[2] = {0, 0};
    if (vertical) {
        redun_dir_v[0] = -1;
        redun_dir_v[1] = +1;
    } else {
        redun_dir_u[0] = -1;
        redun_dir_u[1] = +1;
    }

    for (int32_t u_can = 0; u_can < D_can_width; u_can++) {
        for (int32_t v_can = 0; v_can < D_can_height; v_can++) {
            float d_can = *(D_can + getAddressOffsetImage(u_can, v_can, D_can_width));
            if (d_can >= 0) {
                bool redundant = true;
                for (int32_t i = 0; i < 2; i++) {
                    int32_t u_can_2 = u_can;
                    int32_t v_can_2 = v_can;
                    float d_can_2;
                    bool support = false;
                    for (int32_t j = 0; j < redun_max_dist; j++) {
                        u_can_2 += redun_dir_u[i];
                        v_can_2 += redun_dir_v[i];
                        if (u_can_2 < 0 || v_can_2 < 0 || u_can_2 >= D_can_width || v_can_2 >= D_can_height) {
                            break;
                        }
                        d_can_2 = *(D_can + getAddressOffsetImage(u_can_2, v_can_2, D_can_width));
                        if (d_can_2 >= 0 && abs(d_can - d_can_2) <= redun_threshold) {
                            support = true;
                            break;
                        }
                    }

                    if (!support) {
                        redundant = false;
                        break;
                    }
                }

                if (redundant) {
                    *(D_can + getAddressOffsetImage(u_can, v_can, D_can_width)) = -1;
                }
            }
        }
    }
}

inline uint32_t PlaneExtractor::getAddressOffsetImage(const int32_t& u, const int32_t& v, const int32_t& width) {
    return v * width + u;
}

void PlaneExtractor::applyBilateralFilter() {
    cv::Mat imDisp(m_config.height, m_config.width, CV_32FC1, cv::Scalar(-1.0f));

    for (const auto& sp : supportPoints) {
        if (sp.v >= 0 && sp.v < m_config.height && sp.u >= 0 && sp.u < m_config.width) {
            imDisp.at<float>(sp.v, sp.u) = sp.d;
        }
    }

    cv::Mat imFiltered;
    imFiltered.create(imDisp.size(), imDisp.type());
    imFiltered.setTo(-1.0f);

    float twoSigDispSq = 2.0f * m_config.sigDisp * m_config.sigDisp;
    float twoSigDistSq = 2.0f * m_config.sigDist * m_config.sigDist;

    for (int y = 0; y < m_config.height; ++y) {
        const float* srcRowPtr = imDisp.ptr<float>(y);
        float* dstRowPtr = imFiltered.ptr<float>(y);

        for (int x = 0; x < m_config.width; ++x) {
            float centerVal = srcRowPtr[x];
            if (centerVal == -1.0f) {
                continue;
            }

            float sumWeights = 0.0f;
            float sumVal = 0.0f;

            int y0 = std::max(0, y - m_config.filterRadius);
            int y1 = std::min(m_config.height - 1, y + m_config.filterRadius);
            int x0 = std::max(0, x - m_config.filterRadius);
            int x1 = std::min(m_config.width - 1, x + m_config.filterRadius);

            for (int ny = y0; ny <= y1; ++ny) {
                const float* srcNeighborRow = imDisp.ptr<float>(ny);
                for (int nx = x0; nx <= x1; ++nx) {
                    float neighborVal = srcNeighborRow[nx];
                    if (neighborVal == -1.0f) {
                        continue;
                    }

                    float dy = static_cast<float>(ny - y);
                    float dx = static_cast<float>(nx - x);
                    float distSq = dx * dx + dy * dy;

                    float dDisp = neighborVal - centerVal;
                    float dispSq = dDisp * dDisp;

                    float wDist = std::exp(-distSq / twoSigDistSq);
                    float wDisp = std::exp(-dispSq / twoSigDispSq);
                    float w = wDist * wDisp;

                    sumWeights += w;
                    sumVal += w * neighborVal;
                }
            }

            if (sumWeights > 1e-6f) {
                dstRowPtr[x] = sumVal / sumWeights;
            } else {
                dstRowPtr[x] = centerVal;
            }
        }
    }

    for (auto& sp : supportPoints) {
        if (sp.v >= 0 && sp.v < m_config.height && sp.u >= 0 && sp.u < m_config.width) {
            float val = imFiltered.at<float>(sp.v, sp.u);
            if (val != -1.0f) {
                sp.d = val;
                double z = m_config.bf / sp.d;
                double x = (sp.u - m_config.cx) / m_config.fx * z;
                double y = (sp.v - m_config.cy) / m_config.fy * z;
                sp.point3D << x, y, z;
            }
        }
    }
}

void PlaneExtractor::generateMesh() {
    struct triangulateio in, out;
    int32_t k;

    in.numberofpoints = supportPoints.size();
    in.pointlist = (float*)malloc(in.numberofpoints * 2 * sizeof(float));
    k = 0;

    for (int32_t i = 0; i < supportPoints.size(); i++) {
        in.pointlist[k++] = supportPoints[i].u;
        in.pointlist[k++] = supportPoints[i].v;
    }

    in.numberofpointattributes = 0;
    in.pointattributelist = NULL;
    in.pointmarkerlist = NULL;
    in.numberofsegments = 0;
    in.numberofholes = 0;
    in.numberofregions = 0;
    in.regionlist = NULL;

    out.pointlist = NULL;
    out.pointattributelist = NULL;
    out.pointmarkerlist = NULL;
    out.trianglelist = NULL;
    out.triangleattributelist = NULL;
    out.neighborlist = NULL;
    out.segmentlist = NULL;
    out.segmentmarkerlist = NULL;
    out.edgelist = NULL;
    out.edgemarkerlist = NULL;

    char parameters[] = "zQB";
    triangulate(parameters, &in, &out, NULL);

    k = 0;
    for (int32_t i = 0; i < out.numberoftriangles; i++) {
        Triangle t(out.trianglelist[k], out.trianglelist[k + 1], out.trianglelist[k + 2]);
        computeNormal(t);
        triangles.push_back(t);
        k += 3;
    }

    free(in.pointlist);
    free(out.pointlist);
    free(out.trianglelist);

    discardBadTriangles();

    findNeighbor(triangles);
}

void PlaneExtractor::computeNormal(Triangle& t) {
    Eigen::Vector3d point1 = supportPoints[t.c1].point3D;
    Eigen::Vector3d point2 = supportPoints[t.c2].point3D;
    Eigen::Vector3d point3 = supportPoints[t.c3].point3D;

    Eigen::Vector3d v1(point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]);
    Eigen::Vector3d v2(point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2]);

    t.normal = (v1.cross(v2)).normalized();
}

void PlaneExtractor::findNeighbor(std::vector<Triangle>& triangles) {
    std::unordered_map<Edge, std::vector<int>, EdgeHash> edge2tris;
    edge2tris.reserve(triangles.size() * 3);

    for (int i = 0; i < static_cast<int>(triangles.size()); ++i) {
        const Triangle& tri = triangles[i];

        Edge e1(tri.c1, tri.c2);
        Edge e2(tri.c1, tri.c3);
        Edge e3(tri.c2, tri.c3);

        edge2tris[e1].push_back(i);
        edge2tris[e2].push_back(i);
        edge2tris[e3].push_back(i);
    }

    for (auto& kv : edge2tris) {
        const std::vector<int>& tris = kv.second;
        if (tris.size() < 2) {
            continue;
        }

        for (int i = 0; i < static_cast<int>(tris.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(tris.size()); ++j) {
                int triA = tris[i];
                int triB = tris[j];

                triangles[triA].neighbor.push_back(triB);
                triangles[triB].neighbor.push_back(triA);
            }
        }
    }

    for (auto& tri : triangles) {
        auto& nbrs = tri.neighbor;
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
}

void PlaneExtractor::discardBadTriangles() {
    if (triangles.empty() || supportPoints.empty()) {
        std::cerr << "[Warning] No triangles or no support points, nothing to discard.\n";
        return;
    }

    std::vector<Triangle> validTriangles;
    validTriangles.reserve(triangles.size());

    std::vector<double> allEdges;
    std::vector<double> aspectRatios;
    std::vector<double> minAngles;
    allEdges.reserve(triangles.size() * 3);
    aspectRatios.reserve(triangles.size());
    minAngles.reserve(triangles.size());

    auto isCoordInvalid = [&](const Eigen::Vector3d& pt) {
        return (std::isnan(pt[0]) || std::isinf(pt[0]) ||
                std::isnan(pt[1]) || std::isinf(pt[1]) ||
                std::isnan(pt[2]) || std::isinf(pt[2]));
    };
    auto isDoubleInvalid = [&](double d) {
        return (std::isnan(d) || std::isinf(d));
    };

    for (size_t i = 0; i < triangles.size(); ++i) {
        const Triangle& tri = triangles[i];

        if (tri.c1 < 0 || tri.c1 >= static_cast<int>(supportPoints.size()) ||
            tri.c2 < 0 || tri.c2 >= static_cast<int>(supportPoints.size()) ||
            tri.c3 < 0 || tri.c3 >= static_cast<int>(supportPoints.size()))
        {
            continue;
        }

        const Eigen::Vector3d& v1 = supportPoints[tri.c1].point3D;
        const Eigen::Vector3d& v2 = supportPoints[tri.c2].point3D;
        const Eigen::Vector3d& v3 = supportPoints[tri.c3].point3D;

        if (isCoordInvalid(v1) || isCoordInvalid(v2) || isCoordInvalid(v3)) {
            continue;
        }

        double e12 = (v1 - v2).norm();
        double e23 = (v2 - v3).norm();
        double e31 = (v3 - v1).norm();

        if (e12 > m_config.maxValidEdge || e23 > m_config.maxValidEdge || e31 > m_config.maxValidEdge) {
            continue;
        }

        if (isDoubleInvalid(e12) || isDoubleInvalid(e23) || isDoubleInvalid(e31)) {
            continue;
        }

        allEdges.push_back(e12);
        allEdges.push_back(e23);
        allEdges.push_back(e31);

        double maxEdge = std::max({e12, e23, e31});
        double minEdge = std::min({e12, e23, e31});
        double aspectRatio = (minEdge > 1e-12) ? (maxEdge / minEdge) : 1e10;

        if (std::isnan(aspectRatio) || std::isinf(aspectRatio)) {
            continue;
        }
        aspectRatios.push_back(aspectRatio);

        double angle1 = computeTriAngle(v2 - v1, v3 - v1);
        double angle2 = computeTriAngle(v1 - v2, v3 - v2);
        double angle3 = computeTriAngle(v1 - v3, v2 - v3);

        if (isDoubleInvalid(angle1) || isDoubleInvalid(angle2) || isDoubleInvalid(angle3)) {
            continue;
        }
        double minAngle = std::min({angle1, angle2, angle3});
        minAngles.push_back(minAngle);

        validTriangles.push_back(tri);
    }

    if (validTriangles.empty()) {
        triangles.clear();
        return;
    }

    if (!m_config.checkOutlier) {
        triangles.swap(validTriangles);
        return;
    }

    auto computeMeanStd = [&](const std::vector<double>& arr, double& m, double& s) {
        if (arr.empty()) {
            m = 0.0;
            s = 0.0;
            return;
        }
        double count = 0.0;
        double meanLocal = 0.0;
        double M2 = 0.0;
        for (double x : arr) {
            count += 1.0;
            double delta = x - meanLocal;
            meanLocal += delta / count;
            double delta2 = x - meanLocal;
            M2 += delta * delta2;

            if (std::isnan(meanLocal) || std::isinf(meanLocal) || std::isnan(M2) || std::isinf(M2)) {
                m = 0.0;
                s = 0.0;
                return;
            }
        }
        m = meanLocal;
        double variance = (count > 1.0) ? (M2 / count) : 0.0;
        s = (variance > 0.0) ? std::sqrt(variance) : 0.0;
    };

    double meanEdge = 0.0, stdEdge = 0.0;
    double meanAspect = 0.0, stdAspect = 0.0;
    double meanMinAng = 0.0, stdMinAng = 0.0;

    computeMeanStd(allEdges, meanEdge, stdEdge);
    computeMeanStd(aspectRatios, meanAspect, stdAspect);
    computeMeanStd(minAngles, meanMinAng, stdMinAng);

    double edgeThreshold = meanEdge + m_config.badEdgeThresh * stdEdge;
    double aspectThreshold = meanAspect + m_config.badAspectThresh * stdAspect;
    double minAngleThreshold = meanMinAng - m_config.badAngleThresh * stdMinAng;

    if (minAngleThreshold < 0.0) {
        minAngleThreshold = 0.0;
    }

    std::vector<Triangle> filtered;
    filtered.reserve(validTriangles.size());

    for (auto& tri : validTriangles) {
        const Eigen::Vector3d& v1 = supportPoints[tri.c1].point3D;
        const Eigen::Vector3d& v2 = supportPoints[tri.c2].point3D;
        const Eigen::Vector3d& v3 = supportPoints[tri.c3].point3D;

        double e12 = (v1 - v2).norm();
        double e23 = (v2 - v3).norm();
        double e31 = (v3 - v1).norm();

        if (e12 > edgeThreshold || e23 > edgeThreshold || e31 > edgeThreshold) {
            continue;
        }

        double maxEdge = std::max({e12, e23, e31});
        double minEdge = std::min({e12, e23, e31});
        double aspectRatio = (minEdge > 1e-12) ? (maxEdge / minEdge) : 1e10;
        if (aspectRatio > aspectThreshold) {
            continue;
        }

        double angle1 = computeTriAngle(v2 - v1, v3 - v1);
        double angle2 = computeTriAngle(v1 - v2, v3 - v2);
        double angle3 = computeTriAngle(v1 - v3, v2 - v3);
        double minAngle = std::min({angle1, angle2, angle3});

        if (minAngle < minAngleThreshold) {
            continue;
        }

        filtered.push_back(tri);
    }

    triangles.swap(filtered);
}

inline double PlaneExtractor::computeTriAngle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    double dotVal = a.dot(b);
    double magA = a.norm();
    double magB = b.norm();
    double eps = 1e-12;
    if (magA < eps || magB < eps) {
        return 0.0;
    }
    double cosTheta = dotVal / (magA * magB);

    if (cosTheta > 1.0) cosTheta = 1.0;
    if (cosTheta < -1.0) cosTheta = -1.0;

    double rad = std::acos(cosTheta);

    return (rad * 180.0 / M_PI);
}

void PlaneExtractor::clusterTriangles() {
    clusters.reserve(triangles.size() / 10 + 1);

    for (size_t i = 0; i < triangles.size(); i++) {
        if (triangles[i].clusterID < 0) {
            int newClusterIndex = static_cast<int>(clusters.size());
            Cluster c = performRegionGrowing(static_cast<int>(i), newClusterIndex);
            if (!c.triIndices.empty()) {
                clusters.push_back(c);
            }
        }
    }
}

PlaneExtractor::Cluster PlaneExtractor::performRegionGrowing(int seedTriIdx, int clusterIdx) {
    Cluster cluster;
    cluster.triIndices.reserve(200);

    cluster.triIndices.push_back(seedTriIdx);
    triangles[seedTriIdx].clusterID = clusterIdx;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(200 * 3);

    std::unordered_set<int> addedPointIndices;
    addedPointIndices.reserve(200 * 3);

    int newPointsCountSinceLastFit = 0;

    {
        const Triangle& seedTri = triangles[seedTriIdx];
        int ids[3] = {seedTri.c1, seedTri.c2, seedTri.c3};
        int addedCount = 0;

        for (int spId : ids) {
            if (addedPointIndices.find(spId) == addedPointIndices.end()) {
                addedPointIndices.insert(spId);
                const Eigen::Vector3d& p = supportPoints[spId].point3D;
                cloud->points.emplace_back(p.x(), p.y(), p.z());
                ++addedCount;
            }
        }
        newPointsCountSinceLastFit += addedCount;

        fitPlane(cloud, cluster.planeNormal, cluster.planeDistance);
    }

    std::queue<int> triQueue;
    triQueue.push(seedTriIdx);

    while (!triQueue.empty()) {
        int currentTri = triQueue.front();
        triQueue.pop();

        const Triangle& curTri = triangles[currentTri];
        for (int nbIdx : curTri.neighbor) {
            if (triangles[nbIdx].clusterID >= 0) {
                continue;
            }

            double angle = computeNormalAngle(triangles[nbIdx].normal, cluster.planeNormal);
            if (angle <= m_config.angleThreshWide) {
                double dist = computeDistance(triangles[nbIdx], cluster.planeNormal, cluster.planeDistance);
                if (dist <= m_config.distThreshWide) {
                    cluster.triIndices.push_back(nbIdx);
                    triangles[nbIdx].clusterID = clusterIdx;
                    triQueue.push(nbIdx);

                    int addedCount = 0;
                    {
                        const Triangle& nbTri = triangles[nbIdx];
                        int ids[3] = {nbTri.c1, nbTri.c2, nbTri.c3};
                        for (int spId : ids) {
                            if (addedPointIndices.find(spId) == addedPointIndices.end()) {
                                addedPointIndices.insert(spId);
                                const Eigen::Vector3d& p = supportPoints[spId].point3D;
                                cloud->points.emplace_back(p.x(), p.y(), p.z());
                                addedCount++;
                            }
                        }
                    }

                    if (addedCount > 0) {
                        newPointsCountSinceLastFit += addedCount;

                        if (newPointsCountSinceLastFit >= m_config.reRansacBatch) {
                            fitPlane(cloud, cluster.planeNormal, cluster.planeDistance);
                            newPointsCountSinceLastFit = 0;
                        }
                    }
                }
            }
        }
    }

    {
        std::vector<int> validTri;
        validTri.reserve(cluster.triIndices.size());

        for (int tIdx : cluster.triIndices) {
            double angle2 = computeNormalAngle(triangles[tIdx].normal, cluster.planeNormal);
            double dist2 = computeDistance(triangles[tIdx], cluster.planeNormal, cluster.planeDistance);

            if (angle2 <= m_config.angleThreshStrict && dist2 <= m_config.distThreshStrict) {
                validTri.push_back(tIdx);
            } else {
                triangles[tIdx].clusterID = -1;
            }
        }
        cluster.triIndices.swap(validTri);
    }

    {
        if (!cluster.triIndices.empty()) {
            std::vector<std::vector<int>> subcomponents = extractConnectedComponents(cluster.triIndices);

            size_t maxSize = 0;
            int maxIdx = -1;
            for (size_t i = 0; i < subcomponents.size(); i++) {
                if (subcomponents[i].size() > maxSize) {
                    maxSize = subcomponents[i].size();
                    maxIdx = static_cast<int>(i);
                }
            }

            std::unordered_set<int> biggestSet;
            biggestSet.reserve(maxSize);
            for (int triIdx : subcomponents[maxIdx]) {
                biggestSet.insert(triIdx);
            }

            std::vector<int> finalValidTri;
            finalValidTri.reserve(maxSize);

            for (int tIdx : cluster.triIndices) {
                if (biggestSet.find(tIdx) != biggestSet.end()) {
                    finalValidTri.push_back(tIdx);
                } else {
                    triangles[tIdx].clusterID = -1;
                }
            }
            cluster.triIndices.swap(finalValidTri);
        }
    }

    return cluster;
}

void PlaneExtractor::fitPlane(const std::vector<Eigen::Vector3d>& pts3D, Eigen::Vector3d& normal, double& distance) {
    normal.setZero();
    distance = 0.0;

    if (pts3D.size() < 3) {
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->width = static_cast<uint32_t>(pts3D.size());
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (size_t idx = 0; idx < pts3D.size(); ++idx) {
        cloud->points[idx].x = static_cast<float>(pts3D[idx].x());
        cloud->points[idx].y = static_cast<float>(pts3D[idx].y());
        cloud->points[idx].z = static_cast<float>(pts3D[idx].z());
    }

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(m_config.RansacDistance);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (coefficients->values.size() < 4) {
        return;
    }

    double a = coefficients->values[0];
    double b = coefficients->values[1];
    double c = coefficients->values[2];
    double d = coefficients->values[3];

    double normN = std::sqrt(a*a + b*b + c*c);
    if (normN < 1e-12) {
        return;
    }

    normal = Eigen::Vector3d(a, b, c) / normN;
    distance = d / normN;
}

void PlaneExtractor::fitPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector3d& normal, double& distance) {
    if (cloud->empty()) {
        normal.setZero();
        distance = 0.0;
        return;
    }

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(m_config.RansacDistance);
    seg.setInputCloud(cloud);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    seg.segment(*inliers, *coefficients);
    if (coefficients->values.size() < 4) {
        normal.setZero();
        distance = 0.0;
        return;
    }

    double a = coefficients->values[0];
    double b = coefficients->values[1];
    double c = coefficients->values[2];
    double d = coefficients->values[3];
    double normN = std::sqrt(a * a + b * b + c * c);
    if (normN < 1e-12) {
        normal.setZero();
        distance = 0.0;
        return;
    }

    normal = Eigen::Vector3d(a, b, c) / normN;
    distance = d / normN;
}

inline double PlaneExtractor::computeNormalAngle(const Eigen::Vector3d& n1, const Eigen::Vector3d& n2) {
    double cosv = n1.dot(n2) / (n1.norm() * n2.norm());
    cosv = std::max(-1.0, std::min(1.0, cosv));

    double angle = std::acos(cosv) * (180.0 / M_PI);
    if (angle > 90.0) {
        angle = 180.0 - angle;
    }
    return angle;
}

inline double PlaneExtractor::computeDistance(const Triangle& tri, const Eigen::Vector3d& planeNormal, double planeDist) {
    auto distPoint = [&](const Eigen::Vector3d& p) {
        double val = planeNormal.dot(p) + planeDist;
        return std::fabs(val);
    };

    Eigen::Vector3d p1 = supportPoints[tri.c1].point3D;
    Eigen::Vector3d p2 = supportPoints[tri.c2].point3D;
    Eigen::Vector3d p3 = supportPoints[tri.c3].point3D;

    double d1 = distPoint(p1);
    double d2 = distPoint(p2);
    double d3 = distPoint(p3);

    return std::max({d1, d2, d3});
}

inline void PlaneExtractor::collectPoints(const Cluster& cluster, std::vector<Eigen::Vector3d>& pts) {
    pts.clear();
    pts.reserve(cluster.triIndices.size() * 3);

    std::unordered_set<int> addedIndices;
    addedIndices.reserve(cluster.triIndices.size() * 3);

    for (int triIdx : cluster.triIndices) {
        const Triangle& tri = triangles[triIdx];
        int ids[3] = {tri.c1, tri.c2, tri.c3};

        for (int spId : ids) {
            if (addedIndices.find(spId) == addedIndices.end()) {
                addedIndices.insert(spId);
                pts.push_back(supportPoints[spId].point3D);
            }
        }
    }
}

std::vector<std::vector<int>> PlaneExtractor::extractConnectedComponents(const std::vector<int>& clusterIndices) {
    std::vector<std::vector<int>> components;
    components.reserve(10);

    std::unordered_set<int> inCluster;
    inCluster.reserve(clusterIndices.size());
    for (int idx : clusterIndices) {
        inCluster.insert(idx);
    }

    std::unordered_set<int> visited;
    visited.reserve(clusterIndices.size());

    for (int startIdx : clusterIndices) {
        if (visited.find(startIdx) != visited.end()) {
            continue;
        }

        std::vector<int> subcomponent;
        std::queue<int> que;
        que.push(startIdx);
        visited.insert(startIdx);

        while (!que.empty()) {
            int cur = que.front();
            que.pop();
            subcomponent.push_back(cur);

            for (int nb : triangles[cur].neighbor) {
                if (inCluster.count(nb) && !visited.count(nb)) {
                    visited.insert(nb);
                    que.push(nb);
                }
            }
        }

        components.push_back(subcomponent);
    }

    return components;
}

void PlaneExtractor::mergeClusters() {
    updateAdjacency();

    bool merged = true;
    while (merged) {
        merged = false;

        std::vector<int> clusterIndices;
        clusterIndices.reserve(clusters.size());
        for (int i = 0; i < static_cast<int>(clusters.size()); i++) {
            if (clusters[i].valid && !clusters[i].triIndices.empty()) {
                clusterIndices.push_back(i);
            }
        }
        if (clusterIndices.empty()) {
            break;
        }

        std::sort(clusterIndices.begin(), clusterIndices.end(), [this](int a, int b) {
            return clusters[a].triIndices.size() < clusters[b].triIndices.size();
        });

        for (int ci : clusterIndices) {
            if (!clusters[ci].valid || clusters[ci].triIndices.empty()) {
                continue;
            }

            std::vector<std::pair<int, double>> potentialMerges;
            potentialMerges.reserve(clusters[ci].neighbors.size());

            for (int nb : clusters[ci].neighbors) {
                if (nb < 0 || nb >= static_cast<int>(clusters.size())) {
                    continue;
                }
                if (!clusters[nb].valid || clusters[nb].triIndices.empty()) {
                    continue;
                }

                double angle = computeNormalAngle(clusters[ci].planeNormal, clusters[nb].planeNormal);
                if (angle > m_config.angleThreshMerge) {
                    continue;
                }

                double distDiff = std::fabs(clusters[ci].planeDistance - clusters[nb].planeDistance);
                if (distDiff <= m_config.distThreshMerge) {
                    potentialMerges.emplace_back(nb, distDiff);
                }
            }

            if (potentialMerges.empty()) {
                continue;
            }

            std::sort(potentialMerges.begin(), potentialMerges.end(), [](const std::pair<int,double> &a, const std::pair<int, double> &b) {
                return a.second < b.second;
            });

            int bestNeighborIdx = potentialMerges[0].first;

            if (m_config.checkNN && potentialMerges.size() >= 2) {
                int secondBestIdx = potentialMerges[1].first;

                int lambda1 = getBoundaryTriangleCount(ci, bestNeighborIdx);
                int lambda2 = getBoundaryTriangleCount(ci, secondBestIdx);

                if (lambda1 > 0) {
                    double ratio = static_cast<double>(lambda2) / static_cast<double>(lambda1);
                    if (ratio > m_config.boundaryThresh) {
                        bestNeighborIdx = -1;
                    }
                }
            }

            if (bestNeighborIdx < 0) {
                continue;
            }

            merge(ci, bestNeighborIdx);
            merged = true;
            break;
        }

        if (merged) {
            updateAdjacency();
        }
    }
}

int PlaneExtractor::getBoundaryTriangleCount(int c1, int c2) {
    int count = 0;
    for (int triID : clusters[c1].triIndices) {
        const Triangle& t = triangles[triID];
        bool isBoundary = false;
        for (int nbTriID : t.neighbor) {
            if (nbTriID < 0 || nbTriID >= static_cast<int>(triangles.size())) {
                continue;
            }
            if (triangles[nbTriID].clusterID == c2) {
                isBoundary = true;
                break;
            }
        }
        if (isBoundary) {
            count++;
        }
    }
    return count;
}

void PlaneExtractor::merge(int srcIdx, int dstIdx) {
    auto& srcTris = clusters[srcIdx].triIndices;
    auto& dstTris = clusters[dstIdx].triIndices;

    for (int t : srcTris) {
        triangles[t].clusterID = dstIdx;
        dstTris.push_back(t);
    }
    srcTris.clear();

    clusters[srcIdx].valid = false;

    std::vector<Eigen::Vector3d> pts;
    collectPoints(clusters[dstIdx], pts);
    fitPlane(pts, clusters[dstIdx].planeNormal, clusters[dstIdx].planeDistance);
}

void PlaneExtractor::updateAdjacency() {
    for (auto& c : clusters) {
        c.neighbors.clear();
    }

    for (int i = 0; i < (int)triangles.size(); i++) {
        int ci = triangles[i].clusterID;
        if (ci < 0) {
            continue;
        }

        for (int nb : triangles[i].neighbor) {
            if (nb < 0 || nb >= (int)triangles.size()) {
                continue;
            }
            int cj = triangles[nb].clusterID;
            if (cj >= 0 && cj != ci && clusters[cj].valid) {
                clusters[ci].neighbors.push_back(cj);
                clusters[cj].neighbors.push_back(ci);
            }
        }
    }

    for (int i = 0; i < (int)clusters.size(); i++) {
        if (!clusters[i].valid) {
            continue;
        }
        auto& neigh = clusters[i].neighbors;
        std::sort(neigh.begin(), neigh.end());
        neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());
    }
}

void PlaneExtractor::computePlaneParameters() {
    mainClusters.reserve(clusters.size());
    for (auto& cluster : clusters) {
        if (!cluster.valid) {
            continue;
        }

        size_t triCount = cluster.triIndices.size();
        if (triCount > triangles.size() * m_config.minClusterRatio) {
            mainClusters.push_back(cluster);
        }
    }

    std::sort(mainClusters.begin(), mainClusters.end(), [](const Cluster& a, const Cluster& b) {
        return a.triIndices.size() > b.triIndices.size();
    });

    planes.reserve(mainClusters.size());

    for (size_t clusterIdx = 0; clusterIdx < mainClusters.size(); ++clusterIdx) {
        const auto& currentCluster = mainClusters[clusterIdx];

        std::unordered_set<int> addedIndices;
        addedIndices.reserve(currentCluster.triIndices.size() * 3);

        std::vector<cv::Point2i> pixelCoords;
        std::vector<Eigen::Vector3d> points3D;
        pixelCoords.reserve(currentCluster.triIndices.size() * 3);
        points3D.reserve(currentCluster.triIndices.size() * 3);

        for (int tIdx : currentCluster.triIndices) {
            if (tIdx < 0 || tIdx >= static_cast<int>(triangles.size())) {
                continue;
            }

            const Triangle& tri = triangles[tIdx];
            int ids[3] = {tri.c1, tri.c2, tri.c3};

            for (int spId : ids) {
                if (spId < 0 || spId >= static_cast<int>(supportPoints.size())) {
                    continue;
                }

                if (addedIndices.find(spId) == addedIndices.end()) {
                    addedIndices.insert(spId);

                    const SupportPoint& sp = supportPoints[spId];
                    int uu = static_cast<int>(std::round(sp.u));
                    int vv = static_cast<int>(std::round(sp.v));

                    pixelCoords.emplace_back(uu, vv);
                    points3D.push_back(sp.point3D);
                }
            }
        }

        if (points3D.size() < 3) {
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        cloud->width = static_cast<uint32_t>(points3D.size());
        cloud->height = 1;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);

        for (size_t idx = 0; idx < points3D.size(); ++idx) {
            cloud->points[idx].x = static_cast<float>(points3D[idx].x());
            cloud->points[idx].y = static_cast<float>(points3D[idx].y());
            cloud->points[idx].z = static_cast<float>(points3D[idx].z());
        }

        pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

        {
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(m_config.RansacDistance);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);
        }

        if (inliers->indices.empty() || coefficients->values.size() < 4) {
            continue;
        }

        double inlierRatio = static_cast<double>(inliers->indices.size()) / static_cast<double>(cloud->size());
        if (inlierRatio < m_config.inlierRatio) {
            continue;
        }

        double a = coefficients->values[0];
        double b = coefficients->values[1];
        double c = coefficients->values[2];
        double d = coefficients->values[3];

        double normN = std::sqrt(a * a + b * b + c * c);
        if (normN < 1e-12) {
            continue;
        }

        Eigen::Vector3d newNormal(a / normN, b / normN, c / normN);
        double newD = d / normN;

        bool skipThisPlane = false;
        for (auto& existingPlane : planes) {
            double cosv = existingPlane.normal.dot(newNormal);
            cosv = std::max(-1.0, std::min(1.0, cosv));
            double angleDeg = std::acos(cosv) * (180.0 / M_PI);

            if (angleDeg > 90.0) {
                angleDeg = 180.0 - angleDeg;
            }

            double distDiff = std::fabs(existingPlane.d - newD);

            if (angleDeg < m_config.angleThreshDupPlane && distDiff < m_config.distThreshDupPlane) {
                skipThisPlane = true;
                break;
            }
        }

        if (skipThisPlane) {
            continue;
        }

        Plane newPlane;
        newPlane.normal = newNormal;
        newPlane.d = newD;

        for (int idx : inliers->indices) {
            if (idx >= 0 && idx < static_cast<int>(pixelCoords.size())) {
                const auto& pix = pixelCoords[idx];
                if (pix.x >= 0 && pix.x < m_config.width && pix.y >= 0 && pix.y < m_config.height) {
                    newPlane.planarPointCoords.emplace_back(pix.x, pix.y);
                }
            }
        }

        planes.push_back(newPlane);
    }
}

void PlaneExtractor::visualize(const std::string option) {
    bool visualizePlane = false;
    bool visualizeCluster = false;

    if (option == "plane+cluster") {
        visualizePlane   = true;
        visualizeCluster = true;
    } else if (option == "plane") {
        visualizePlane = true;
    } else if (option == "cluster") {
        visualizeCluster = true;
    } else {
        std::cerr << "[Warning] Unknown visualization option: " << option << std::endl;
        return;
    }

    if (visualizePlane) {
        cv::Mat imPlanarPoint = canvas.clone();
        if (imPlanarPoint.empty()) {
            std::cerr << "[Warning] canvas is empty, cannot visualize planes.\n";
            return;
        }

        std::vector<cv::Scalar> colors = generateColors(static_cast<int>(planes.size()));

        for (size_t planeIdx = 0; planeIdx < planes.size(); ++planeIdx) {
            cv::Scalar color = colors[planeIdx % colors.size()];

            for (auto& pix : planes[planeIdx].planarPointCoords) {
                if (pix.x >= 0 && pix.x < imPlanarPoint.cols &&
                    pix.y >= 0 && pix.y < imPlanarPoint.rows)
                {
                    cv::circle(imPlanarPoint, pix, 3, color, -1);
                }
            }
        }

        cv::imshow("PlaneExtraction", imPlanarPoint);
    }

    if (visualizeCluster) {
        cv::Mat imCluster = canvas.clone();
        if (imCluster.empty()) {
            std::cerr << "[Warning] canvas is empty, cannot visualize clusters.\n";
            return;
        }

        cv::Mat overlay = imCluster.clone();

        std::vector<cv::Scalar> colors = generateColors(static_cast<int>(mainClusters.size()));

        int colorIdx = 0;

        for (size_t clusterIdx = 0; clusterIdx < clusters.size(); ++clusterIdx) {
            size_t triCount = clusters[clusterIdx].triIndices.size();
            if (triCount == 0) {
                continue;
            }

            double ratio = static_cast<double>(triCount) / static_cast<double>(triangles.size());

            cv::Scalar color;
            if (ratio > m_config.minClusterRatio) {
                color = colors[colorIdx % colors.size()];
                colorIdx++;
            } else {
                color = cv::Scalar(200, 200, 200);
            }

            for (int idxTri : clusters[clusterIdx].triIndices) {
                if (idxTri < 0 || idxTri >= static_cast<int>(triangles.size())) {
                    continue;
                }

                int32_t x1 = supportPoints[triangles[idxTri].c1].u;
                int32_t y1 = supportPoints[triangles[idxTri].c1].v;
                int32_t x2 = supportPoints[triangles[idxTri].c2].u;
                int32_t y2 = supportPoints[triangles[idxTri].c2].v;
                int32_t x3 = supportPoints[triangles[idxTri].c3].u;
                int32_t y3 = supportPoints[triangles[idxTri].c3].v;

                cv::line(overlay, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
                cv::line(overlay, cv::Point(x2, y2), cv::Point(x3, y3), color, 2);
                cv::line(overlay, cv::Point(x3, y3), cv::Point(x1, y1), color, 2);

                std::vector<cv::Point> pts;
                pts.reserve(3);
                pts.emplace_back(x1, y1);
                pts.emplace_back(x2, y2);
                pts.emplace_back(x3, y3);
                cv::fillConvexPoly(overlay, pts, color, cv::LINE_8);
            }
        }

        double alpha = 0.4;
        cv::addWeighted(overlay, alpha, imCluster, 1.0 - alpha, 0.0, imCluster);

        cv::imshow("Clusters", imCluster);
    }

    cv::waitKey(1);
}

std::vector<cv::Scalar> PlaneExtractor::generateColors(int n) {
    std::vector<cv::Scalar> colors;
    colors.reserve(n);

    for (int i = 0; i < n; i++) {
        double hue = 180.0 * i / n;
        double sat = 200.0;
        double val = 255.0;
        colors.push_back(hsv2bgr(hue, sat, val));
    }

    return colors;
}

cv::Scalar PlaneExtractor::hsv2bgr(double h, double s, double v) {
    cv::Mat3b hsvPixel(1, 1, cv::Vec3b(
            static_cast<uchar>(h),
            static_cast<uchar>(s),
            static_cast<uchar>(v)
    ));

    cv::Mat3b bgrPixel;
    cv::cvtColor(hsvPixel, bgrPixel, cv::COLOR_HSV2BGR);
    cv::Vec3b bgr = bgrPixel(0, 0);
    return cv::Scalar(bgr[0], bgr[1], bgr[2]);
}
