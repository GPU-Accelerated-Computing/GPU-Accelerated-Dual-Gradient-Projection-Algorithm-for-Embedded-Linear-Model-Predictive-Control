#include "matrix.h"
#include <cmath>

#define N 3
#define P 4
#define NUMSAMPLES 100


int main() {


    if (N == 10){
        Matrix<double> x0(1, N);
        x0.fill({-0.1, 0.45, -0.09, 0.05, 0, -0.05, 0.3, 0.2, 0.25, -0.45});
    }else if (N == 3){
        Matrix<double> x0(1, N);
        x0.fill({-0.1, 0.05, 0, -0.05, 0.1});
    }else{
        Matrix<double> x0(1, N);
        for (int i = 0; i < N; i++){
            // randome value between 1 and 0
            x0[0][i] = (double)rand() / RAND_MAX;
            
        }
    }

    
    // Cell capacities in Ah
    Matrix<double> cellCapacities(1, N);
    cellCapacities.ones();
    cellCapacities = cellCapacities * 2.5;


    // Some random test stuff to make sure the inverse matrix calulation is correct
    Matrix<double> mat(3, 3);
    mat[0][0] = 4; mat[0][1] = 7; mat[0][2] = 2;
    mat[1][0] = 3; mat[1][1] = 6; mat[1][2] = 1;
    mat[2][0] = 2; mat[2][1] = 5; mat[2][2] = 3;

    std::cout << "Original Matrix:\n";
    mat.print();

    Matrix<double> inv(3, 3);

    inv = mat.inverse();
    std::cout << "\nInverse Matrix:\n";
    mat.print();

    mat = mat.multiply(inv);
    std::cout << "\nProduct of Matrix and its Inverse:\n";
    mat.print();

    return 0;
}
