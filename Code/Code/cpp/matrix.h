#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>

// TODO : ones

template <typename T>
class Matrix {
public:
    // Initializes and sets the data to zeros
    Matrix(int rows, int cols)
        : rows(rows), cols(cols), data(new T[rows * cols]()) {}

    // Helper class because I want to address a matrix like [][]
    class MatrixRow {
    public:
        MatrixRow(T* rowData, int cols) : rowData(rowData), cols(cols) {}

        T& operator[](int col) {
            return rowData[col];
        }

        const T& operator[](int col) const {
            return rowData[col];
        }

    private:
        T* rowData;
        int cols;
    };

    MatrixRow operator[](int row) {
        return MatrixRow(&data[row * cols], cols); 
    }

    const MatrixRow operator[](int row) const {
        return MatrixRow(&data[row * cols], cols); 
    }

    // Destructor
    ~Matrix() {
        delete[] data;
    }

    // Copy Constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(new T[other.rows * other.cols]) {
        std::copy(other.data, other.data + rows * cols, data);
    }

    // Copy Assignment Operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = new T[rows * cols];
            std::copy(other.data, other.data + rows * cols, data);
        }
        return *this;
    }

    void fill(std::initializer_list<T> values) {
        if (values.size() != rows * cols) {
            throw std::invalid_argument("Initializer list size does not match matrix dimensions.");
        }

        int index = 0;
        for (const auto& val : values) {
            data[index++] = val;
        }
    }

    void ones() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = 1;
            }
        }
    }

    // Identity matrix
    void identity() {
        if (rows != cols) {
            throw std::invalid_argument("Identity matrix must be square");
        }
        for (int i = 0; i < rows; ++i) {
            data[i * cols + i] = 1; 
        }
    }

    Matrix<T> returnIdentity() const {
        if (rows != cols) {
            throw std::invalid_argument("Identity matrix must be square");
        }
        Matrix<T> result(rows, cols);
        result.identity();
        return result;
    }

    // Eye matrix
    void eye() {
        for (int i = 0; i < std::min(rows, cols); ++i) {
            data[i * cols + i] = 1; 
        }
    }

    Matrix<T> multiply(const Matrix<T>& other) const {
        // Check if the matrices can be multiplied (columns of this matrix must equal rows of other matrix)
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible matrices for multiplication");
        }

        // Create a new matrix to store the result (rows of this * columns of other)
        Matrix<T> result(rows, other.cols);

        // Perform the multiplication: result[i][j] = dot product of row i of this and column j of other
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                T sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += (*this)[i][k] * other[k][j];  // Dot product of row i and column j
                }
                result[i][j] = sum;  // Set the computed value in the result matrix
            }
        }

        return result;
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible matrices for multiplication");
        }

        Matrix<T> result(rows, other.cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                T sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += (*this)[i][k] * other[k][j];  
                }
                result[i][j] = sum;
            }
        }

        return result;
    }

    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[i][j] = (*this)[i][j] * scalar;
            }
        }
        return result;
    }
        


    // Matrix<T> result(rows, other.cols);
    //     result.identity();

    //     for (int i = 0; i < rows; ++i) {
    //         for (int j = 0; j < other.cols; ++j) {
    //             T sum = 0;
    //             for (int k = 0; k < cols; ++k) {
    //                 sum += (*this)[i][k] * other[k][j];
    //             }
    //             result[i][j] = sum;
    //         }
    //     }
    //     return result;
    // }

    // void operator + (){
    //     for (int i = 0; i < rows; i++) {
    //         for (int j = 0; j < cols; j++) {
    //             data[i * cols + j] = data[i * cols + j] + data[i * cols + j];
    //         }
    //     }
    // }

    Matrix<T> augment(const Matrix<T>& other) const {
        if (this->rows != other.rows) {
            throw std::invalid_argument("Matrices must have the same number of rows to augment.");
        }

        Matrix<T> result(this->rows, this->cols + other.cols);

        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                result[i][j] = (*this)[i][j];
            }
        }

        for (int i = 0; i < other.rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                result[i][j + this->cols] = other[i][j];
            }
        }

        return result;
    }


   // Inverse Matrix
    Matrix<T> inverse() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square");
        }

        // A|I augmented matrix
        Matrix<T> aug = this->augment(this->returnIdentity());
        Matrix<T> result(rows, rows);

        // Gaussian Elimination with partial pivoting
        for (int i = 0; i < rows; ++i) {
            if (aug[i][i] == 0) {
                bool swapped = false;
                for (int j = i + 1; j < rows; ++j) {
                    if (aug[j][i] != 0) {
                        for (int k = 0; k < 2 * rows; ++k) {
                            std::swap(aug[i][k], aug[j][k]);
                        }
                        swapped = true;
                        break;
                    }
                }
                if (!swapped) {
                    throw std::runtime_error("Matrix is singular and cannot be inverted.");
                }
            }

            // Scale
            T pivot = aug[i][i];
            for (int j = 0; j < 2 * rows; ++j) {
                aug[i][j] /= pivot;
            }

            for (int j = 0; j < rows; ++j) {
                if (j != i) {
                    T factor = aug[j][i];
                    for (int k = 0; k < 2 * rows; ++k) {
                        aug[j][k] -= factor * aug[i][k];
                    }
                }
            }
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                result[i][j] = aug[i][j + rows];
            }
        }

    return result;
}


    // Print
    void print() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << (*this)[i][j] << " ";
            }
            std::cout << std::endl; 
        }
    }

private:
    int rows;
    int cols;
    T* data;
};

#endif
