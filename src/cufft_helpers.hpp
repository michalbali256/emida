#pragma once

#include <cufft.h>

namespace emida
{

// This file contains cuFFT error checkers and templated wrappers around the cuFFT functions

#define FFTCH(status) emida::fft_check(status, __LINE__, __FILE__, #status)

static const char* cufft_get_error_message(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

inline void fft_check(cufftResult status, int line, const char* src_filename, const char* line_str = nullptr)
{
	if (status != CUFFT_SUCCESS)
	{
		std::stringstream ss;
		ss << "CUDA Error " << status << ":" << cufft_get_error_message(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
		std::cerr << ss.str();
		throw std::runtime_error(ss.str());
	}
}

template<typename T>
inline cufftType fft_type_R2C();
template<>
inline cufftType fft_type_R2C<float>()
{
    return cufftType::CUFFT_R2C;
}
template<>
inline cufftType fft_type_R2C<double>()
{
    return cufftType::CUFFT_D2Z;
}


template<typename T>
inline cufftType fft_type_C2R();
template<>
inline cufftType fft_type_C2R<float>()
{
    return cufftType::CUFFT_C2R;
}
template<>
inline cufftType fft_type_C2R<double>()
{
    return cufftType::CUFFT_Z2D;
}

template<typename T>
struct complex_trait
{
    using type = float;
};

template<>
struct complex_trait<float>
{
    using type = cufftComplex;
};
template<>
struct complex_trait<double>
{
    using type = cufftDoubleComplex;
};

template<typename T>
inline void fft_real_to_complex(cufftHandle plan, T* in, typename complex_trait<T>::type* out);
template<>
inline void fft_real_to_complex<float>(cufftHandle plan, float* in, complex_trait<float>::type* out)
{
    FFTCH(cufftExecR2C(plan, in, out));
}
template<>
inline void fft_real_to_complex<double>(cufftHandle plan, double* in, complex_trait<double>::type* out)
{
    FFTCH(cufftExecD2Z(plan, in, out));
}

template<typename T>
inline void fft_real_to_complex(cufftHandle plan, T* in_out);
template<>
inline void fft_real_to_complex<float>(cufftHandle plan, float* in_out)
{
    FFTCH(cufftExecR2C(plan, in_out, (cufftComplex*)in_out));
}
template<>
inline void fft_real_to_complex<double>(cufftHandle plan, double* in_out)
{
    FFTCH(cufftExecD2Z(plan, in_out, (cufftDoubleComplex*)in_out));
}

template<typename T>
inline void fft_complex_to_real(cufftHandle plan, typename complex_trait<T>::type* in, T* out);
template<>
inline void fft_complex_to_real(cufftHandle plan, typename complex_trait<float>::type* in, float* out)
{
    FFTCH(cufftExecC2R(plan, in, out));
}
template<>
inline void fft_complex_to_real(cufftHandle plan, typename complex_trait<double>::type* in, double* out)
{
    FFTCH(cufftExecZ2D(plan, in, out));
}

template<typename T>
inline void fft_complex_to_real(cufftHandle plan, T* in_out);
template<>
inline void fft_complex_to_real<float>(cufftHandle plan, float* in_out)
{
    FFTCH(cufftExecC2R(plan, (cufftComplex*)in_out, in_out));
}
template<>
inline void fft_complex_to_real<double>(cufftHandle plan, double* in_out)
{
    FFTCH(cufftExecZ2D(plan, (cufftDoubleComplex*)in_out, in_out));
}

template<typename T>
struct sums_trait
{
    using type = double;
};

template<>
struct sums_trait<uint16_t>
{
    using type = uint32_t;
};
template<>
struct sums_trait<float>
{
    using type = float;
};

template<>
struct sums_trait<double>
{
    using type = double;
};


}
