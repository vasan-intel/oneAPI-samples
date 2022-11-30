#include <math.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

#include <list>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "qrd.hpp"

/*
  COMPLEX, COLS_COMPONENT, ROWS_COMPONENT and FIXED_ITERATIONS are defined
  by the build system.
  Depending on the value of COMPLEX, the real or complex QRDecomposition is
  defined

  Function arguments:
  - a_matrix:    The input matrix. Interpreted as a transposed matrix.
  - q_matrix:    The Q matrix. The function will overwrite this matrix.
  - r_matrix     The R matrix. The function will overwrite this matrix.
                 The vector will only contain the upper triangular elements
                 of the matrix, in a row by row fashion.
  - q:           The device queue.
  - matrix_count: Number of matrices to decompose.
  - repetitions: The number of repetitions of the computation to execute.
                 (for performance evaluation)
*/
#if COMPLEX == 0
// Real single precision floating-point QR Decomposition
void QRDecomposition(std::vector<float> &a_matrix, std::vector<float> &q_matrix,
                     std::vector<float> &r_matrix, sycl::queue &q,
                     int matrix_count,
                     int repetitions) {
  constexpr bool is_complex = false;
  QRDecompositionImpl<COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS,
                       is_complex, float>(a_matrix, q_matrix, r_matrix, q,
                                          matrix_count, repetitions);
}
#else
// Complex single precision floating-point QR Decomposition
void QRDecomposition(std::vector<ac_complex<float> > &a_matrix,
                     std::vector<ac_complex<float> > &q_matrix,
                     std::vector<ac_complex<float> > &r_matrix, sycl::queue &q,
                     int matrix_count,
                     int repetitions) {
  constexpr bool is_complex = true;
  QRDecompositionImpl<COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS,
                       is_complex, float>(a_matrix, q_matrix, r_matrix, q,
                                          matrix_count, repetitions);
}
#endif

/*
  returns if both the real and complex parts of the given ac_complex
  value are finite
*/
bool IsFinite(ac_complex<float> val) {
  return std::isfinite(val.r()) && std::isfinite(val.i());
}

/*
  returns if the given value is finite
*/
bool IsFinite(float val) { return std::isfinite(val); }

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;
  constexpr size_t kRows = ROWS_COMPONENT;
  constexpr size_t kColumns = COLS_COMPONENT;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kQMatrixSize = kRows * kColumns;
  constexpr size_t kRMatrixSize = kColumns * (kColumns + 1) / 2;
  constexpr size_t kQRMatrixSize = kQMatrixSize + kRMatrixSize;
  constexpr bool kComplex = COMPLEX != 0;

  // Get the number of times we want to repeat the decomposition
  // from the command line.
#if defined(FPGA_EMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
#else
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
#endif
  if (repetitions < 1) {
    std::cout << "Number of repetitions given is lower that 1." << std::endl;
    std::cout << "The decomposition must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kMatricesToDecompose = 8;

  try {
    // SYCL boilerplate
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif

    // Enable the queue profiling to time the execution
    sycl::property_list
                    queue_properties{sycl::property::queue::enable_profiling()};
    sycl::queue q = sycl::queue(device_selector,
                                dpc_common::exception_handler,
                                queue_properties);

    sycl::device device = q.get_device();
    std::cout << "Device name: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Select a type for this compile depending on the value of COMPLEX
    using T = std::conditional_t<kComplex, ac_complex<float>, float>;

    // Create vectors to hold all the input and output matrices
    std::vector<T> a_matrix;
    std::vector<T> q_matrix;
    std::vector<T> r_matrix;

    a_matrix.resize(kAMatrixSize * kMatricesToDecompose);
    q_matrix.resize(kQMatrixSize * kMatricesToDecompose);
    r_matrix.resize(kRMatrixSize * kMatricesToDecompose);

    std::cout << "Generating " << kMatricesToDecompose << " random ";
    if constexpr (kComplex) {
      std::cout << "complex ";
    } else {
      std::cout << "real ";
    }
    std::cout << "matri" << (kMatricesToDecompose > 1 ? "ces" : "x")
              << " of size "
              << kRows << "x" << kColumns << " " << std::endl;

    // Generate the random input matrices
    srand(kRandomSeed);

    for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
                                                                matrix_index++){
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          float random_real = rand() % (kRandomMax - kRandomMin) + kRandomMin;
  #if COMPLEX == 0
          a_matrix[matrix_index * kAMatrixSize
                 + col * kRows + row] = random_real;
  #else
          float random_imag = rand() % (kRandomMax - kRandomMin) + kRandomMin;
          ac_complex<float> random_complex{random_real, random_imag};
          a_matrix[matrix_index * kAMatrixSize
                 + col * kRows + row] = random_complex;
  #endif
        }  // end of col
      }    // end of row

  #ifdef DEBUG
      std::cout << "A MATRIX " << matrix_index << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << a_matrix[matrix_index * kAMatrixSize
                              + col * kRows + row] << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row
  #endif

    } // end of matrix_index

    std::cout << "Running QR decomposition of " << kMatricesToDecompose
              << " matri" << (kMatricesToDecompose > 1 ? "ces " : "x ")
              << repetitions << " times" << std::endl;

    QRDecomposition(a_matrix, q_matrix, r_matrix, q, kMatricesToDecompose,
                                                                  repetitions);

    return 0;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;

    std::terminate();
  } catch (std::bad_alloc const &e) {
    std::cerr << "Caught a memory allocation exception on the host: "
              << e.what() << std::endl;
    std::cerr << "   You can reduce the memory requirement by reducing the "
                 "number of matrices generated. Specify a smaller number when "
                 "running the executable."
              << std::endl;
    std::cerr << "   In this run, more than "
              << ((kAMatrixSize + kQRMatrixSize) * 2 * kMatricesToDecompose
                 * sizeof(float)) / pow(2, 30)
              << " GBs of memory was requested for the decomposition of a "
              << "matrix of size " << kRows << " x " << kColumns
              << std::endl;
    std::terminate();
  }
}  // end of main
