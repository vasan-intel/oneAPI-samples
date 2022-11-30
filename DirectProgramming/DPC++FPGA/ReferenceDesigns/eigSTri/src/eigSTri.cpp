#ifndef __STREAMING_EIG_STRI__
#define __STREAMING_EIG_STRI__


#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

template <  typename T,
            int kMsize,
            typename Ain,
            typename Aout
        >

struct StreamingEigStri{

    void operator()() const {

        // Data structure for tridiagonal Matrix
        // there will be only 3 elements in each row of matrix
        // even in intermediate computation 
        // for simplicity pipe_size is set as 3 


        // b1   c1  *   *   *   *   
        // a2   b2  c2  *   *   *
        // *    a3  b3  c3  *   *
        // *    *   a4  b4  c4  *
        // *    *   *   a5  b5  c5   
        // *    *   *   *   a6  b6        


        const int kWidth = 3;
        // Type used to store the matrices in the compute loop
        using Wtuple = fpga_tools::NTuple<TT, kWidth>;

        Wtuple a_load[kMsize];


        //copying TriDiagonal matrix to local memory 
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for(int i = 0; i < kMsize; i++){
            a_load[i] = Ain::read();
        }


    }


}

#endif