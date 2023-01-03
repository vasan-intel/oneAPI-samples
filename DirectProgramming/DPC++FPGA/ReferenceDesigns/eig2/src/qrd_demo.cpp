#include <math.h>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>


#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>


#include <list>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#define KTHRESHOLD 1e-4
#define KDEFLIM 2
#define KETHRESHOLD 1e-4
#define RELSHIFT 0
#define SHIFT_NOISE 1e-4
#include "dpc_common.hpp"

#include "qrd.hpp"
#include "qr_MGS.hpp"
#include "hessenberg_qrd.hpp"

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

// Real double precision floating-point QR Decomposition
// void QRDecomposition(std::vector<double> &a_matrix, std::vector<double> &q_matrix,
//                      std::vector<double> &r_matrix, sycl::queue &q,
//                      int matrix_count,
//                      int repetitions) {
//   constexpr bool is_complex = false;
//   QRDecompositionImpl<COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS,
//                        is_complex, double>(a_matrix, q_matrix, r_matrix, q,
//                                           matrix_count, repetitions);

// }
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
  constexpr size_t kRandomMax = 100;
  constexpr size_t kRows = ROWS_COMPONENT;
  constexpr size_t kColumns = COLS_COMPONENT;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kQMatrixSize = kRows * kColumns;
  constexpr size_t kRMatrixSize = kRows * kColumns;
  constexpr size_t kQRMatrixSize = kQMatrixSize + kRMatrixSize;
  constexpr bool kComplex = COMPLEX != 0;

  int iter = 10000;


  // Get the number of times we want to repeat the decomposition
  // from the command line.
#if defined(FPGA_EMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;
#else
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
#endif
  if (repetitions < 1) {
    std::cout << "Number of repetitions given is lower that 1." << std::endl;
    std::cout << "The decomposition must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kMatricesToDecompose = 1;

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
    // using T = std::conditional_t<kComplex, ac_complex<double>, double>;
    using T = std::conditional_t<kComplex, ac_complex<float>, float>;

    // Create vectors to hold all the input and output matrices
    std::vector<T> a_matrix;
    std::vector<T> rq_matrix;
    std::vector<T> qq_matrix;

    a_matrix.resize(kAMatrixSize * kMatricesToDecompose);
    rq_matrix.resize(kQMatrixSize * kMatricesToDecompose);
    qq_matrix.resize(kRMatrixSize * kMatricesToDecompose);

    std::cout << "Generating " << kMatricesToDecompose << " random ";
    if constexpr (kComplex) {
      std::cout << "complex ";
    } else {
      std::cout << "real ";
    }
    std::cout << "matri" << (kMatricesToDecompose > 1 ? "ces" : "x")
              << " of size "
              << kRows << "x" << kColumns << " " << std::endl;

    // Generate the random symmetric square matrices
    srand(kRandomSeed);

    for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
                                                                matrix_index++){
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col <= row; col++) {
          float random_real = (rand() % (kRandomMax - kRandomMin) + kRandomMin); // * 1.0/kRandomMax;
  #if COMPLEX == 0
          a_matrix[matrix_index * kAMatrixSize
                 + col * kRows + row] = random_real;
          a_matrix[matrix_index * kAMatrixSize
                 + row * kRows + col] = random_real;
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

    QRDecomposition(a_matrix, rq_matrix, qq_matrix, q, kMatricesToDecompose,
                                                                  repetitions);

    // eigen value & vector computation on CPU for same data
    std::vector<T> a_matrix_cpu;
    std::vector<T> eigen_vectors_cpu;
    std::vector<T> TmpRow;

    a_matrix_cpu.resize(kAMatrixSize * kMatricesToDecompose);
    eigen_vectors_cpu.resize(kAMatrixSize * kMatricesToDecompose);
    TmpRow.resize(kRows);

    std::vector<int> sIndex(kRows);


    std::vector<T> py_w(kRows);
    std::vector<T> py_V(kRows*kRows);

    // copy A matrix to CPU data
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        a_matrix_cpu[i*kRows+j] = a_matrix[j*kRows+i];
      }
    }

    //initialize the eigen vectors to identity mtrix
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        eigen_vectors_cpu[i*kRows+j] = (i == j) ? 1 : 0;
      }
    }

    // Initialize the idexes for sorting 
    for(int i = 0; i < kRows; i++){
      sIndex[i] = i;
    }


     // Printig the diff matrix 
    std::ofstream osA("mat_A.txt");
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        osA << std::setprecision(15) << a_matrix[j*kRows+i];
        if(j != kRows-1 || i != kRows-1){
          osA << ",";
        }
      }
    }
    osA.close();

    if(system("python2 ../src/eig_IQR.py") != 0){
      std::cout << "Error occured when trying to execute the python script\n";
    }

    // reading back golden results
    std::ifstream osW("mat_W.txt");
    for(int i = 0; i < kRows; i++){
      osW >> py_w[i];
    }
    osW.close();

    // reading back golden results
    std::ifstream osV("mat_V.txt");  
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        osV >> py_V[i*kRows+j];
      }
    }
    osV.close();


    std::cout << "\n last element is: " << a_matrix_cpu[(kRows-1)*kRows + kRows-1] << "\n";

    // std::memcpy(a_matrix_cpu.data(), a_matrix.data(), kAMatrixSize * kMatricesToDecompose*sizeof(T));
    QR_Decmp<T> qrd_cpu(a_matrix_cpu.data(), kRows);
    // QR_Decmp<T> qrd_cpu(data, kRows);
    iter = 10000;
    int kP = kRows;
    T *R, *Q;
    for(int li = 0; li < iter; li++){

      T a_wilk = a_matrix_cpu[(kP-2)*kRows+kP-2];
      T b_wilk = a_matrix_cpu[(kP-1)*kRows+kP-2];
      T c_wilk = a_matrix_cpu[(kP-1)*kRows+kP-1];

      T lamda = (a_wilk - c_wilk)/2.0;
      T sign_lamda = (lamda > 0) - (lamda < 0);

      T shift = RELSHIFT ? c_wilk : c_wilk - (sign_lamda*b_wilk*b_wilk)/(fabs(lamda) + sqrt(lamda * lamda + b_wilk*b_wilk));
      shift -= SHIFT_NOISE;
      std::cout << "Shift value at iteration: " << li << " is:" << shift << " kP=" << kP << "\n";

      // T shift = a_matrix_cpu[(kP-1)*kRows+kP-1];
      // subtracting the shift from the matrix
      for(int i = 0; i < kP; i++){
        a_matrix_cpu[i*kRows+i] -= shift;
      }

      qrd_cpu.QR_decompose(kP);
      R = qrd_cpu.get_R();
      Q = qrd_cpu.get_Q();
      // RQ computation and updating A 
      for(int i = 0; i < kP; i++){
        for(int j = 0; j < kP; j++){
          a_matrix_cpu[i*kRows+j] = 0;
          for(int k = 0; k < kP; k++){
            a_matrix_cpu[i*kRows+j] += R[i*kRows+k]*Q[k*kRows+j];
          }
        }
      }

      // adding back the shift from the matrix
      for(int i = 0; i < kP; i++){
        a_matrix_cpu[i*kRows+i] += shift;
      }

      // Eigen vector accumulation 
      for(int i = 0; i < kRows; i++){
        std::fill(TmpRow.begin(), TmpRow.end(), 0);
        for(int j = 0; j < kRows; j++){
          for(int k = 0; k < kRows; k++){
            T I_val = (k==j) ? 1 : 0;
            T q_val = (j >= kP || k >= kP) ? I_val : Q[k*kRows+j];
            TmpRow[j] += eigen_vectors_cpu[i*kRows+k]*q_val;
          }
        }
        for(int k = 0; k < kRows; k++) eigen_vectors_cpu[i*kRows+k] = TmpRow[k];
      }



      // convergence test 
      bool close2zero = 1;
      // const float threshold = KTHRESHOLD;
      // check zero thereshold for lower part 

      for(int j = 0; j < kP-1; j++){
        if(std::fabs(a_matrix_cpu[(kP-1)*kRows+j]) > KTHRESHOLD){
          // std::cout << "failed at i: " << i << " j: " << j << "\n"; 
          close2zero = 0;
          break;
        }
      }


      // std::cout << "CPU Iteration: " << li << " kP: " << kP << "\n";
      // if(li == 15){
      //   break;
      // }

      if(close2zero && kP == KDEFLIM){
        std::cout << "CPU convergence achieved at iter: " << li << "\n";
        break;
      } else if(close2zero){
        // for(int j = 0; j < kP-1; j++){
        //   a_matrix_cpu[(kP-1)*kRows+j] = 0;
        //   a_matrix_cpu[j*kRows+(kP-1)] = 0;
        // }
        kP -= 1;
      }
    
    }


    // sorting the eigen values 
    std::sort(sIndex.begin(), sIndex.end(), [=](int a, int b){ return fabs(a_matrix_cpu[a*kRows+a]) > fabs(a_matrix_cpu[b*kRows+b]);});

    std::cout << "\nR matrix from cpu computation: \n";
    for(int i = 0; i < kP; i++){
      for(int j = 0; j < kP; j++){
        std::cout << R[i*kRows+j] << " ";
      }
      std::cout << "\n";
    }

    std::cout << "\nQ matrix from cpu computation: \n";
    for(int i = 0; i < kP; i++){
      for(int j = 0; j < kP; j++){
        std::cout << Q[i*kRows+j] << " ";
      }
      std::cout << "\n";
    }


    // Printig the diff matrix 
    std::cout << "\nRQ matrix from cpu computation: \n";
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        std::cout << a_matrix_cpu[i*kRows+j] << " ";
      }
      std::cout << "\n";
    }

    std::cout << "\n\nRQ matrix from SYCL kernel computation: \n";
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        std::cout << rq_matrix[j*kRows+i] << " ";
      }
      std::cout << "\n";
    }

    T diff_threshold = KETHRESHOLD;
    int rq_ecount = 0;
    for(int i = 0; i < kRows; i++){
      if(fabs(fabs(a_matrix_cpu[i*kRows+i]) - fabs(rq_matrix[i*kRows+i])) > diff_threshold 
      || isnan(rq_matrix[i*kRows+i]) || isnan(a_matrix_cpu[i*kRows+i])){
        rq_ecount++;
        std::cout << "Mis matched values are: " << a_matrix_cpu[i*kRows+i] << ", " << rq_matrix[i*kRows+i] << " at i: " << i << "\n";
      }
    }

    if(rq_ecount == 0){
      std::cout << "\n\npassed:  CPU eigen values and Kernel Eigen values are matched\n\n";
    } else {
      std::cout << "Mismatch is found between CPU RQ and Kernel RQ\n";
    }


    // Printig the diff matrix 
    std::cout << "\nEigen Vectors from cpu computation(columns): \n";
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        std::cout << eigen_vectors_cpu[j*kRows+sIndex[i]] << " ";
      }
      std::cout << "\n";
    }

    std::cout << "\n\nEigen vectors from SYCL kernel computation(columns): \n";
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        std::cout << qq_matrix[j*kRows+sIndex[i]] << " ";
      }
      std::cout << "\n";
    }

  double sq_error_cpp = 0, sq_error_SYCL = 0;
  std::cout << "\n\nEigen vectors from python numpy: \n";
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        std::cout << py_V[i*kRows+j] << " ";
        sq_error_cpp += (fabs(eigen_vectors_cpu[j*kRows+sIndex[i]]) - fabs(py_V[i*kRows+j])) * \
                          (fabs(eigen_vectors_cpu[j*kRows+sIndex[i]]) - fabs(py_V[i*kRows+j]));

        sq_error_SYCL += (fabs(qq_matrix[j*kRows+sIndex[i]]) - fabs(py_V[i*kRows+j])) * \
                  (fabs(qq_matrix[j*kRows+sIndex[i]]) - fabs(py_V[i*kRows+j]));
      }
      std::cout << "\n";
    }


    std::cout << "\n\nNumpy-CPP ABS square error sum is: " << sq_error_cpp << "\n";
    std::cout << "Numpy-SYCL ABS square error sum is: " << sq_error_SYCL << "\n\n";

    int qq_ecountCPP = 0;
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        if(fabs(fabs(py_V[i*kRows+j]) - fabs(eigen_vectors_cpu[j*kRows+sIndex[i]])) > diff_threshold 
        || isnan(eigen_vectors_cpu[j*kRows+sIndex[i]]) || isnan(py_V[i*kRows+j])){
          qq_ecountCPP++;
          // std::cout << "Mis matched values are: " << py_V[i*kRows+j] << ", " << eigen_vectors_cpu[i*kRows+sIndex[j]] << " at i,j:"
          //  << i << "," << j << "\n";
        }
      }
    }

    int qq_ecountSYCL = 0;
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        if(fabs(fabs(py_V[i*kRows+j]) - fabs(qq_matrix[j*kRows+sIndex[i]])) > diff_threshold 
        || isnan(qq_matrix[j*kRows+sIndex[i]]) || isnan(py_V[i*kRows+j])){
          qq_ecountSYCL++;
          std::cout << "Mis matched values are: " << py_V[i*kRows+j] << ", " << qq_matrix[j*kRows+sIndex[i]] << "," << eigen_vectors_cpu[j*kRows+sIndex[i]] << " at i,j:"
           << i << "," << j << "\n";
        }
      }
    }


    if(qq_ecountCPP == 0){
      std::cout << "\n\npassed:  CPU eigen vectors and numpy Eigen vectors are matched\n\n";
    } else {
      std::cout << "\n\n Error: Mismatch is found between CPU QQ and nump QQ, count: " << qq_ecountCPP << "\n\n";
    }

    if(qq_ecountSYCL == 0){
      std::cout << "\n\npassed:  SYCL and numpy Eigen vectors are matched\n\n";
    } else {
      std::cout << "\n\n Error: Mismatch is found between SYCL and numpy QQ, count: " << qq_ecountSYCL << "\n\n";
    }




  


    // // For output post-processing (op)
    // T rq_matrix_op[kRows][kColumns];
    // T qq_matrix_op[kRows][kColumns];

    // // For rectangular matrices, Q is only going to have orthogonal columns
    // // so we won't check if the rows are orthogonal
    // bool square_matrices = kRows == kColumns;

    // // Floating-point error threshold value at which we decide that the design
    // // computed an incorrect value
    // constexpr float kErrorThreshold = 1e-4;
    // // The orthogonality check is more sensible to numerical error, the
    // // threshold is then set a bit higher
    // float q_ortho_error_threshold = pow(2.0, -9);

//     // Check Q and R matrices
//     std::cout << "Verifying results...";
//     for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
//                                                                 matrix_index++){

//       // keep track of Q and R element indexes
//       size_t r_idx = 0;
//       size_t q_idx = 0;

//       // Read the R matrix from the output vector to the RMatrixOP matrix
//       for (size_t i = 0; i < kRows; i++) {
//         for (size_t j = 0; j < kColumns; j++) {
//           qq_matrix_op[i][j] = qq_matrix[matrix_index*kRMatrixSize
//                                        + i*kColumns+j];
//           }
//         }
//       }

//       // Read the Q matrix from the output vector to the QMatrixOP matrix
//       for (size_t j = 0; j < kColumns; j++) {
//         for (size_t i = 0; i < kRows; i++) {
//           rq_matrix_op[i][j] = rq_matrix[matrix_index*kQMatrixSize
//                                      + q_idx];
//           q_idx++;
//         }
//       }

//   #ifdef DEBUG
//       std::cout << "RQ MATRIX" << std::endl;
//       for (size_t i = 0; i < kRows; i++) {
//         for (size_t j = 0; j < kColumns; j++) {
//           std::cout << rq_matrix_op[i][j] << " ";
//         }
//         std::cout << std::endl;
//       }

//       std::cout << "QQ MATRIX" << std::endl;
//       for (size_t i = 0; i < kRows; i++) {
//         for (size_t j = 0; j < kColumns; j++) {
//           std::cout << qq_matrix_op[i][j] << " ";
//         }
//         std::cout << std::endl;
//       }
//   #endif

//       // Count the number of errors found for this matrix
//       size_t error_count = 0;
//       bool error = false;

//       for (size_t i = 0; i < kRows; i++) {
//         for (size_t j = 0; j < kColumns; j++) {
//           // Compute Q * R at index i,j
//           T q_r_ij{0};
//           for (size_t k = 0; k < kColumns; k++) {
//             q_r_ij += q_matrix_op[i][k] * r_matrix_op[k][j];
//           }

//           // Compute transpose(Q) * Q at index i,j
//           T qt_q_ij{0};
//           if (i < kColumns) {
//             for (size_t k = 0; k < kRows; k++) {
//   #if COMPLEX == 0
//               qt_q_ij += q_matrix_op[k][i] * q_matrix_op[k][j];
//   #else
//               qt_q_ij += q_matrix_op[k][i] * q_matrix_op[k][j].conj();
//   #endif
//             }
//           }

//           // Compute Q * transpose(Q) at index i,j
//           T q_qt_ij{0};
//           if (square_matrices) {
//             if (i < kColumns) {
//               for (size_t k = 0; k < kRows; k++) {
//   #if COMPLEX == 0
//                 q_qt_ij += q_matrix_op[i][k] * q_matrix_op[j][k];
//   #else
//                 q_qt_ij += q_matrix_op[i][k] * q_matrix_op[j][k].conj();
//   #endif
//               }
//             }
//           }

//           // Verify that all the results are OK:
//           // Q * R = A at index i,j
//           bool q_r_eq_a;
//           // transpose(Q) * Q = Id at index i,j
//           bool qt_q_eq_id;
//           // Q * transpose(Q) = Id at index i,j
//           bool q_qt_eq_id;
//           // R is upped triangular
//           bool r_is_upper_triang;
//           // R is finite at index i,j
//           bool r_is_finite;

//   #if COMPLEX == 0
//           q_r_eq_a = abs(a_matrix[matrix_index * kAMatrixSize
//                                 + j * kRows + i]
//                        - q_r_ij) < kErrorThreshold;

//           qt_q_eq_id =
//                   ((i == j) && (abs(qt_q_ij - 1) < q_ortho_error_threshold)) ||
//                   ((i != j) && (abs(qt_q_ij) < q_ortho_error_threshold));

//           q_qt_eq_id = !square_matrices ||
//                   (((i == j) && (abs(q_qt_ij - 1) < q_ortho_error_threshold)) ||
//                   ((i != j) && (abs(q_qt_ij) < q_ortho_error_threshold)));

//           r_is_upper_triang =
//               (i >= kColumns) ||
//               ((i > j) && ((abs(r_matrix_op[i][j]) < kErrorThreshold))) ||
//               ((i <= j));

//   #else
//           q_r_eq_a = (abs(a_matrix[matrix_index * kAMatrixSize
//                                  + j * kRows + i].r() -
//                        q_r_ij.r()) < kErrorThreshold) &&
//                   (abs(a_matrix[matrix_index * kAMatrixSize
//                               + j * kRows + i].i() -
//                        q_r_ij.i()) < kErrorThreshold);

//           qt_q_eq_id =
//               (((i == j) && (abs(qt_q_ij.r() - 1) < q_ortho_error_threshold)) ||
// (((i != j) || (j >= kRows)) && (abs(qt_q_ij.r()) < q_ortho_error_threshold))) &&
//               (abs(qt_q_ij.i()) < q_ortho_error_threshold);

//           q_qt_eq_id =
//               !square_matrices ||
//             ((((i == j) && (abs(q_qt_ij.r() - 1) < q_ortho_error_threshold)) ||
//                 (((i != j) || (j >= kRows)) &&
//                  (abs(q_qt_ij.r()) < q_ortho_error_threshold))) &&
//                (abs(q_qt_ij.i()) < q_ortho_error_threshold));

//           r_is_upper_triang =
//               (i >= kColumns) ||
//               ((i > j) && ((abs(r_matrix_op[i][j].r()) < kErrorThreshold) &&
//                            (abs(r_matrix_op[i][j].i()) < kErrorThreshold))) ||
//               (i <= j);

//   #endif

//           r_is_finite =
//             ((i < kColumns) && IsFinite(r_matrix_op[i][j])) || (i >= kColumns);

//           // If any of the checks failed
//           if (!q_r_eq_a || !qt_q_eq_id || !q_qt_eq_id || !r_is_upper_triang ||
//               !IsFinite(q_r_ij) || !IsFinite(qt_q_ij) || !IsFinite(q_qt_ij) ||
//               !r_is_finite) {
//             // Increase the error count for this matrix
//             error_count++;

//             // Continue counting the errors even if we now we are going to
//             // produce an error
//             if (error) {
//               continue;
//             }

//             if (!q_r_eq_a) {
//               std::cout << "Error: A[" << i << "][" << j << "] = "
//                         << a_matrix[matrix_index * kAMatrixSize
//                                   + j * kRows + i]
//                         << " but QR[" << i << "][" << j << "] = " << q_r_ij
//                         << std::endl;
//             }
//             if (!q_r_eq_a) {
//               std::cout << "The difference is greater than tolerated ("
//                         << kErrorThreshold << ")" << std::endl;
//             }
//             if (!qt_q_eq_id || !q_qt_eq_id) {
//               std::cout << "Q is not orthogonal at i " << i << " j " << j << ":"
//                         << std::endl
//                         << " transpose(Q) * Q = " << qt_q_ij << std::endl
//                         << " Q * transpose(Q) =" << q_qt_ij << std::endl;
//               std::cout << "q_ortho_error_threshold = "
//                         << q_ortho_error_threshold
//                         << std::endl;
//             }
//             if (!r_is_upper_triang) {
//               std::cout << "R is not upper triangular at i " << i << " j " << j
//                         << ":" << std::endl
//                         << " R = " << r_matrix_op[i][j] << std::endl;
//             }
//             if (!IsFinite(q_r_ij)) {
//               std::cout << "QR[" << i << "][" << j << "] = " << q_r_ij
//                         << " is not finite" << std::endl;
//             }
//             if (!IsFinite(qt_q_ij)) {
//               std::cout << "transpose(Q) * Q at i " << i << " j " << j << " = "
//                         << qt_q_ij << " is not finite" << std::endl;
//             }
//             if (!IsFinite(q_qt_ij)) {
//               std::cout << "Q * transpose(Q) at i " << i << " j " << j << " = "
//                         << q_qt_ij << " is not finite" << std::endl;
//             }
//             if (!r_is_finite) {
//               std::cout << "R[" << i << "][" << j << "] = " << r_matrix_op[i][j]
//                         << " is not finite" << std::endl;
//             }
//             error = true;
//           }
//         }  // end of j
//       }    // end of i

//       if (error_count > 0) {
//         std::cout << std::endl << "FAILED" << std::endl;
//         std::cout << std::endl
//                   << "!!!!!!!!!!!!!! " << error_count << " errors" << std::endl;
//         return 1;
//       }
//     } // end of matrix_index


    // std::cout << std::endl << "PASSED" << std::endl;
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
