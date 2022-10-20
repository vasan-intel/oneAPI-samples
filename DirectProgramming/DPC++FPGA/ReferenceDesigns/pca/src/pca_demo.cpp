#include<iostream>
#include<math.h> 
#include<cstdlib>

// #include <sycl/sycl.hpp>
// #include <sycl/ext/intel/fpga_extensions.hpp>
// #include <sycl/ext/intel/ac_types/ac_complex.hpp>

/*
this source implements the steps to 
identify principal compoents (eigen vectors) 
of a matrix and finally transform input matrix
along the directions of the principal components

Following are the main steps in order to transform a 
matrix A. Matrix A will contain n samples with p features
making it nxp order matrix

1. Calculating the mean vector u
   (F_0, F_1, ..., F_(p-1))

2. Calculating zero mean matrix 
   B = A - h^{T}*u                 
   here h is a vector with ones of size n

3. Calculate covariance matrix of size pxp
   C = (1.0/(n-1)) * B*B^{T}

4. Calculating eigen vectors and eigen values QR decomposition
   in an iterative loop

5. sort eigen vectors using eigen values in a decending order

6. form the tranformation matrix using eigen vectors
*/


template <typename T> class PCA {
 private: 
    int n, p, debug;
    T *matA, *matdA, *vecU, *matC, *matQ, *matR;
    T *eigen_vecs, *eigen_vals;
    T *matTrans;

 public: 
    PCA(int n, int p, int debug);
    ~PCA();
    void populate_A();
    void calculate_mean_vec();
    void calculate_deviation_vec();
    void calculate_covariance();
    void do_qrd_iteration(int n);
    void sort_eigen_vecs();
    T* do_pca_steps();

};


template<typename T> PCA<T>::PCA(int n,int p, int debug = 0){
    this->n = n;
    this->p = p;
    this->debug = debug;
    this->matA = new T[n*p];
    this->matdA = new T[n*p];
    this->vecU = new T[p];
    this->matC = new T[p*p];
    this->matQ = new T[p*p];
    this->matR = new T[p*p];

    this->eigen_vecs= new T[p*p];
    this->eigen_vals = new T[p];
    this->matTrans = new T[p*p]; 
}

template<typename T> PCA<T>::~PCA(){
    delete this->matA;
    delete this->matdA;
    delete this->vecU;
    delete this->matC;
    delete this->matQ;
    delete this->matR;

    delete this->eigen_vecs;
    delete this->eigen_vals;
    delete this->matTrans;
}


 // populating matrix a with random numbers
template<typename T> void PCA<T>::populate_A(){
    if(debug) std::cout << "Matrix A: \n";
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            this->matA[i*p+j] = (1.0*std::rand())/RAND_MAX;
            if(debug) std::cout << this->matA[i*p+j] << " ";
        }
        if(debug) std::cout << "\n";
    }
}

template<typename T> void PCA<T>::calculate_mean_vec(){
    // setting initial vector value to zero
    for(int i = 0; i < p; i++){
        this->vecU[i] = 0;
    }

    // getting vector sum of the samples
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            this->vecU[j] += this->matA[i*p+j];
        }
    }
    if(debug) std::cout << "\nMean vector is: \n";
    
    // calculating the average
    for(int i = 0; i < p; i++){
        this->vecU[i] /= n;
        if(debug) std::cout << this->vecU[i] << " ";
    }
    if(debug) std::cout <<"\n";


}

template<typename T> void PCA<T>::calculate_deviation_vec(){
    //subtracting mean vec from all the sample
    if(debug) std::cout << "\n Deviation matrix is: \n";
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            this->matdA[i*p+j] = this->matA[i*p+j]-this->vecU[j];
            if(debug) std::cout << this->matdA[i*p+j] << " ";
        }
        if(debug) std::cout << "\n";
    }
}

template<typename T> void PCA<T>::calculate_covariance(){
    // covariance matrix matdA^{T} * matdA
    // this corresponds to matrix order pxp
    for(int i = 0; i < p; i++){
        for(int j = 0; j < p; j++ ){
            this->matC[i*p+j] = 0;
            for(int k = 0; k < n; k++){
                this->matC[i*p+j] += this->matdA[k*p+i]*this->matA[k*p+j];
            }
            this->matC[i*p+j] = (1.0/(n-1))*this->matC[i*p+j];
        }
    }
}


template<typename T> void PCA<T>::do_qrd_iteration(int n){

}


template<typename T> T* PCA<T>::do_pca_steps(){
    this->populate_A();
    this->calculate_mean_vec();
    this->calculate_deviation_vec();
    this->calculate_covariance();

    return this->matC;
}


int main(){

    int n = 10, p = 5;
    PCA<float> pca(n, p, 1);
    float * cov = pca.do_pca_steps();

    std::cout << "\nCovariance matrix is: \n";
    for(int i = 0; i < p; i++){
        for(int j = 0; j < p; j++){
            std::cout << cov[i*p+j] << " ";
        }
        std::cout << "\n";
    }

    return 0;

}