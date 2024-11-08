#include <iostream>
#include <stdexcept>
#include <algorithm>

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

        // Overload operator[] to access individual elements in a row
        T& operator[](int col) {
            return rowData[col]; // Access element in the row
        }

        const T& operator[](int col) const {
            return rowData[col]; // Access element in the row (const version)
        }

    private:
        T* rowData;
        int cols;
    };

    // Overload operator[] to return a MatrixRow for a given row
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

    // identity
    void identity() {
        if (rows != cols) {
            throw std::invalid_argument("Identity matrix must be square (rows == cols).");
        }
        for (int i = 0; i < rows; ++i) {
            data[i * cols + i] = 1; 
        }
    }

    // eye
    void eye() {
        for (int i = 0; i < std::min(rows, cols); ++i) {
            data[i * cols + i] = 1; 
        }
    }

    // print
    void print() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << std::endl; 
        }
    }

    //TODO
    //Inverse Matrix Operations
    //

private:
    int rows;
    int cols;
    T* data; // Pointer to the matrix data
};
