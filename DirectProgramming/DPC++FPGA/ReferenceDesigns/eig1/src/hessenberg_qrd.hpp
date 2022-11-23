#include<iostream>
#include<math.h> 
#include<cstdlib>



/*

    this implements the Hessenberg QR decmposition 

*/

template<typename T> 
class Hess_QR_Decmp{
    
    public:
        // this is for square matrix
        int n;
        T *matA_ptr, *matH, *matQ;
        T *vecU, *vecV, *vecTmp; 
        T *vecC, *vecS;

    public: 
        Hess_QR_Decmp(T *matA_ptr, int n);
        ~Hess_QR_Decmp();
        void hessXform();
        void hess_qr_rq();
};


template<typename T> 
Hess_QR_Decmp<T>::Hess_QR_Decmp(T *matA_ptr, int n){
    this->matA_ptr = matA_ptr;
    this->n = n;

    this->matH = new T[n*n];
    this->matQ = new T[n*n];
    this->vecU = new T[n];
    this->vecV = new T[n];
    this->vecTmp = new T[n];

    this->vecC = new T[n];
    this->vecS = new T[n];

    // copying input matrix to H
    for(int i = 0; i < n*n; i++){
        this->matH[i] = this->matA_ptr[i];
    }

    // initialising Q matrix to identity matrix 
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matQ[i*n+j] = (i==j) ? 1.0 : 0;
        }
    }
}

template<typename T> 
Hess_QR_Decmp<T>::~Hess_QR_Decmp(){


    delete this->matH;
    delete this->matQ;
    delete this->vecU;
    delete this->vecV;
    delete this->vecTmp;

    delete this->vecC;
    delete this->vecS;
}


template<typename T> 
void Hess_QR_Decmp<T>::hessXform(){
    int n = this->n;
    for(int j = 0; j < n-2; j++){
        
        // copying j^th columns j+1:n entries 
        T sum = 0;
        for(int i = j+1; i < n; i++){
            this->vecU[i] = this->matH[i*n+j];
            sum += this->matH[i*n+j]*this->matH[i*n+j];
        }
        T norm_u = sqrt(sum);
        // update the first element and vecU
        this->vecU[j+1] +=  this->vecU[j+1]/fabs(this->vecU[j+1]) * norm_u;

        // calculation the new norm 
        sum = 0;
        for(int i = j+1; i < n; i++){
            sum += this->vecU[i]*this->vecU[i];
        }

        // compute vecV
        norm_u = sqrt(sum);
        for(int i =j+1; i < n; i++){
            this->vecV[i] = this->vecU[i]/norm_u;
        }

        // H[j+1:n,:] -=  2*v@(np.transpose(v)@H[j+1:n,:])    
        for(int k = 0; k < n; k++){
            this->vecTmp[k] = 0;
            for(int i = j+1; i < n; i++){
                this->vecTmp[k] += this->vecV[i] * this->matH[i*n+k];
            }
        }

        // updating H 
        for(int i = j+1; i < n; i++){
            for(int k = 0; k < n; k++){
                this->matH[i*n+k] -= 2 * this->vecV[i] * this->vecTmp[k];
            }
        }

        // H[:,j+1:n] -= (H[:,j+1:n] @ (2*v)) @ np.transpose(v)
        for(int k = 0; k < n; k++){
            this->vecTmp[k] = 0;
            for(int i = j+1; i < n; i++){
                this->vecTmp[k] += 2*this->matH[k*n+i]*this->vecV[i];
            }
        }

        // updating H
        for(int k = 0; k < n; k++){
            for(int i = j+1; i < n; i++){
                this->matH[k*n+i] -= this->vecV[i] * this->vecTmp[k]; 
            }
        }


    }

    // elements other than on tri-diagonal are set to zero
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matH[i*n+j] = (abs(i-j) > 1 ) ? 0 : this->matH[i*n+j];
        }
    }

}


template<typename T> 
void Hess_QR_Decmp<T>::hess_qr_rq(){

    // QR decomposition 
    for(int j = 0; j < n-1; j++){
        T u_0 = this->matH[j*n+j];
        T u_1 = this->matH[(j+1)*n+j];

        T norm = sqrt(u_0*u_0 + u_1*u_1);
        T c = u_0/norm;
        T s = u_1/norm;

        this->vecC[j] = c;
        this->vecS[j] = s;

        for(int i = 0; i < n; i++){
            T h_val = c*this->matH[j*n+i] + s*this->matH[(j+1)*n+i];
            T l_val = 0 -s*this->matH[j*n+i] + c*this->matH[(j+1)*n+i];
            this->matH[j*n+i] = h_val;
            this->matH[(j+1)*n+i] = l_val;
        }
    }

    // RQ computation 
    for(int j = 0; j < n-1; j++){

        T c = this->vecC[j];
        T s = this->vecS[j];

        for(int i = 0; i < n; i++){
            T l_val = this->matH[i*n+j]*c +this->matH[i*n+j+1]*s;
            T r_val = 0-this->matH[i*n+j]*s +this->matH[i*n+j+1]*c;
            this->matH[i*n+j] = l_val;
            this->matH[i*n+j+1] = r_val;
        }
    }

    // elements other than on tri-diagonal are set to zero
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matH[i*n+j] = (abs(i-j) > 1 ) ? 0 : this->matH[i*n+j];
        }
    }
}