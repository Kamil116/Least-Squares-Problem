#include <iostream>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <random>

using namespace std;

/**
 * This class is storing a matrix
 */
class Matrix {
public:
    int n;
    int m;
    // Creating two-dimensional dynamic array to store matrix
    vector<vector<double>> matrix;

    friend istream &operator>>(istream &input, Matrix &obj);

    friend ostream &operator<<(ostream &output, Matrix &obj);

    /**
     * Constructor for assigning matrix and it's sizes
     * @param matrix
     * @param n
     * @param m
     */
    Matrix(vector<vector<double>> mtrx, int n, int m) {
        this->matrix = std::move(mtrx);
        this->n = n;
        this->m = m;
    }

    Matrix() {
        this->n = 0;
        this->m = 0;
    }

    Matrix(int n, int m) {
        for (int i = 0; i < n; i++) {
            matrix[i].resize(m);
        }
    }


    /**
     * "+" operator will sum up two matrices
     * @param second
     * @return
     */
    Matrix *operator+(Matrix &second) {
        if (this->n != second.n || this->m != second.m) {
            return nullptr;
        }

        vector<vector<double>> ans;
        for (int i = 0; i < this->n; i++) {
            ans.emplace_back(this->m);
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < this->m; j++) {
                ans[i][j] = this->matrix[i][j] + second.matrix[i][j];
            }
        }

        return new Matrix(ans, this->n, this->m);
    }

    /** "=" operator will assign one matrix to another
     *
     * @param second
     */
    void operator=(Matrix &second) {
        this->matrix = second.matrix;
    }

    /**
     * "-" operator will subtract two matrices
     * @param second
     * @return
     */
    Matrix *operator-(Matrix second) {
        if (this->n != second.n || this->m != second.m) {
            return nullptr;
        }

        vector<vector<double>> ans;
        for (int i = 0; i < this->n; i++) {
            ans.emplace_back(this->m);
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < this->m; j++) {
                ans[i][j] = this->matrix[i][j] - second.matrix[i][j];
            }
        }

        return new Matrix(ans, this->n, this->m);
    }

    /**
     * "*" operator will multiply two matrices
     * @param second
     * @return
     */
    Matrix *operator*(Matrix second) {
        if (this->m != second.n) {
            return nullptr;
        }

        vector<vector<double>> ans;
        for (int i = 0; i < this->n; i++) {
            ans.emplace_back(second.m);
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < second.m; j++) {
                double counter = 0;
                for (int k = 0; k < this->m; k++) {
                    counter += this->matrix[i][k] * second.matrix[k][j];
                }
                ans[i][j] = counter;
            }
        }

        return new Matrix(ans, this->n, second.m);
    }

    /**
     * Transposing matrix
     * @return
     */
    Matrix *transposed() {
        vector<vector<double>> ans;
        for (int i = 0; i < this->m; i++) {
            ans.emplace_back(this->n);
        }

        for (int i = 0; i < this->m; i++) {
            for (int j = 0; j < this->n; j++) {
                ans[i][j] = this->matrix[j][i];
            }
        }

        return new Matrix(ans, this->m, this->n);
    }

    int getN() const {
        return this->n;
    }

    int getM() const {
        return this->m;
    }
};

class SquareMatrix : public Matrix {
public:
    SquareMatrix(int n, vector<vector<double>> &matrix) : Matrix(matrix, n, n) {};

    SquareMatrix() {};

    SquareMatrix(int size) {
        this->n = size;
        this->m = size;
        matrix.resize(size);
        for (int i = 0; i < size; i++) {
            matrix[i].resize(size);
        }
    }

    SquareMatrix *operator+(Matrix &second) {
        if (this->n != second.n) {
            return nullptr;
        }

        vector<vector<double>> ans;
        for (int i = 0; i < this->n; i++) {
            ans.emplace_back(this->m);
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < this->m; j++) {
                ans[i][j] = this->matrix[i][j] + second.matrix[i][j];
            }
        }

        auto *ret = new SquareMatrix(this->n, ans);
        return ret;
    }

    SquareMatrix *operator-(Matrix second) {
        if (this->n != second.n || this->m != second.m) {
            return nullptr;
        }

        vector<vector<double>> ans;
        for (int i = 0; i < this->n; i++) {
            ans.emplace_back(this->m);
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < this->m; j++) {
                ans[i][j] = this->matrix[i][j] - second.matrix[i][j];
            }
        }

        return new SquareMatrix(this->n, ans);
    }

    /**
     * "*" operator will multiply two matrices
     * @param second
     * @return
     */
    SquareMatrix *operator*(Matrix &second) {
        if (this->m != second.n) {
            return nullptr;
        }

        vector<vector<double>> ans;
        for (int i = 0; i < this->n; i++) {
            ans.emplace_back(second.m);
        }

        for (int i = 0; i < this->n; i++) {
            for (int j = 0; j < second.m; j++) {
                double counter = 0;
                for (int k = 0; k < this->m; k++) {
                    counter += this->matrix[i][k] * second.matrix[k][j];
                }
                ans[i][j] = counter;
            }
        }

        return new SquareMatrix(this->n, ans);
    }

    /**
     * Transposing matrix
     * @return
     */
    SquareMatrix *transposed() {
        vector<vector<double>> ans;
        for (int i = 0; i < this->m; i++) {
            ans.emplace_back(this->n);
        }

        for (int i = 0; i < this->m; i++) {
            for (int j = 0; j < this->n; j++) {
                ans[i][j] = this->matrix[j][i];
            }
        }

        return new SquareMatrix(this->n, ans);
    }
};

class IdentityMatrix : public SquareMatrix {
public:
    IdentityMatrix(int n) : SquareMatrix(n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    matrix[i][j] = 1;
                } else {
                    matrix[i][j] = 0;
                }
            }
        }
    }

};

class EliminationMatrix : public IdentityMatrix {
public:

    double k;
    int row = 0;

    EliminationMatrix(Matrix &matrix1) : IdentityMatrix(matrix1.n) {
    }

    EliminationMatrix(int a, int b, Matrix &matrix1, double q) : IdentityMatrix(matrix1.n) {
        k = q;
        this->matrix[a][b] = k;
    }

    void directWay(int a, int b, Matrix &matrix1) {
        for (int i = a - 1; i >= 0; i--) {
            if (matrix1.matrix[i][b] != 0) {
                k = -(matrix1.matrix[a][b] / matrix1.matrix[i][b]);
                this->matrix[a][b] = k;
                row = i;
                return;
            }
        }
    }

    void wayBack(int a, int b, Matrix &matrix1) {
        for (int i = a + 1; i < n; i++) {
            if (matrix1.matrix[i][b] != 0) {
                k = -(matrix1.matrix[a][b] / matrix1.matrix[i][b]);
                this->matrix[a][b] = k;
                row = i;
                return;
            }
        }
    }

};

class PermutationMatrix : public IdentityMatrix {
    int i, j;

public:

    PermutationMatrix(int a, int b, Matrix &matrix1) : IdentityMatrix(matrix1.n) {
        i = a;
        j = b;
        this->matrix[i][i] = 0;
        this->matrix[i][j] = 1;

        this->matrix[j][j] = 0;
        this->matrix[j][i] = 1;
    }
};

class InversedMatrix {
    Matrix *A;
    int n;
    Matrix *right;

public:
    InversedMatrix(Matrix &B) {
        A = &B;
        this->n = A->n;
        right = new IdentityMatrix(n);
    }

    Matrix *inversed() {

        for (int i = 0; i < n; i++) {
            checkPermutation(i);
            for (int j = 0; j < n; j++) {
                directElimination(i);
            }
        }

        for (int j = n - 1; j >= 0; j--) {
            for (int k = 0; k < n; k++) {
                backWardElimination(j);
            }
        }


        normalization();
        return right;
    }

    void directElimination(int column) {
        for (int j = column + 1; j < n; j++) {
            if (A->matrix[j][column] != 0) {

                EliminationMatrix eliminationMatrix(*A);
                eliminationMatrix.directWay(j, column, *A);

                A = eliminationMatrix * *A;
                A->matrix[j][column] = 0;

                EliminationMatrix eliminationMatrix2(j, column, *right, eliminationMatrix.k);
                right = eliminationMatrix2 * *right;

                return;
            }
        }
    }

    void backWardElimination(int column) {
        for (int j = column - 1; j >= 0; j--) {
            if (A->matrix[j][column] != 0) {

                EliminationMatrix eliminationMatrix(*A);
                eliminationMatrix.wayBack(j, column, *A);
                A = eliminationMatrix * *A;
                A->matrix[j][column] = 0;

                EliminationMatrix eliminationMatrix2(j, column, *right, eliminationMatrix.k);
                right = eliminationMatrix2 * *right;

                return;
            }
        }
    }

    void checkPermutation(int row) {

        // i = number of the row
        double pivot;
        pivot = abs(A->matrix[row][row]);
        double maxim_pivot = 1e-100;
        int ind;
        for (int i = row + 1; i < n; i++) {
            if (abs(A->matrix[i][row]) > maxim_pivot) {
                maxim_pivot = abs(A->matrix[i][row]);
                ind = i;
            }
        }
        if (maxim_pivot > pivot) {
            permute(row, ind);
        }
    }

    void permute(int row, int i) {
        // need to permute rows i and j
        // A = P_ij * A

        PermutationMatrix perm(row, i, *A);
        A = perm * *A;

        PermutationMatrix perm2(row, i, *right);
        right = perm2 * *right;
    }

    void normalization() {
        for (int i = 0; i < n; i++) {
            double divider = A->matrix[i][i];
            if (divider == 0.0001) {
                continue;
            }
            A->matrix[i][i] /= divider;
            for (int j = 0; j < n; j++) {
                right->matrix[i][j] /= divider;
            }
        }
    }
};


/**
 * To read input and not copy-paste the same code
 */
istream &operator>>(istream &input, Matrix &obj) {
    input >> obj.n;
    obj.m = obj.n;

    for (int i = 0; i < obj.n; i++) {
        obj.matrix.emplace_back(obj.m);
    }

    for (int i = 0; i < obj.n; i++) {
        for (int j = 0; j < obj.m; j++) {
            input >> obj.matrix[i][j];
        }
    }

    return input;
}

/**
     * Writing matrix to the output
     */
ostream &operator<<(ostream &output, Matrix &obj) {
    if (obj.matrix.empty()) {
        output << "Error: the dimensional problem occurred" << endl;
        return output;
    }

    for (int i = 0; i < obj.n; i++) {
        for (int j = 0; j < obj.m; j++) {
            if (j == obj.m - 1) {
                if (abs(obj.matrix[i][j]) < 0.0001) {
                    output << 0.00;
                } else {
                    output << fixed << setprecision(4) << obj.matrix[i][j];
                }
            } else {
                if (abs(obj.matrix[i][j]) < 0.0001) {
                    output << 0.00 << " ";
                } else {
                    output << fixed << setprecision(4) << obj.matrix[i][j] << " ";
                }
            }
        }
        cout << endl;
    }
    return output;
}

void fillMatrix(Matrix &mtrx, vector<vector<double>> &original_coeff) {
    for (int i = 0; i < mtrx.n; i++) {
        for (int j = 0; j < mtrx.m; j++) {
            if (j == 0) {
                mtrx.matrix[i][j] = 1;
            } else {
                mtrx.matrix[i][j] = pow(original_coeff[i][0], j);
            }
        }
    }
}

#define GNUPLOT_NAME "C:\\gnuplot\\bin\\gnuplot -persist"

int main() {
    FILE *pipe = _popen(GNUPLOT_NAME, "w");
    int dimension = 10, polyn_degree = 3;

    default_random_engine _random{std::random_device{}()};
    uniform_real_distribution<double> interval(-10, 10);

    vector<vector<double>> plt;

    plt.resize(dimension);
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < polyn_degree + 1; j++) {
            plt[i].push_back(interval(_random));
        }
    }

    Matrix A(plt, dimension, polyn_degree + 1);
    fillMatrix(A, plt);


    cout << "A:" << endl;
    cout << A;

    Matrix *b = *A.transposed() * A;
    cout << "A_T*A:" << endl;
    cout << *b;

    InversedMatrix c(*b);
    Matrix *d = c.inversed();
    cout << "(A_T*A)^-1:" << endl;
    cout << d;

    vector<vector<double>> b_matrix;
    b_matrix.resize(dimension);

    for (int i = 0; i < dimension; i++) {
        double y = plt[i][1];
        b_matrix[i].resize(1);
        b_matrix[i][0] = y;
    }

    cout << "A_T*b:" << endl;
    Matrix b_vector(b_matrix, dimension, 1);
    Matrix *e = *A.transposed() * b_vector;
    cout << *e;

    cout << "x~:" << endl;
    Matrix ans = *(*d * *e);
    cout << ans;

    fprintf(pipe, "plot [-10 : 20] [-10 : 100] %lf*x**3 + %lf*x**2 + %lf*x**1 + %lf*x**0 ,"
                  " '-' using 1:2 with points\n", ans.matrix[3][0], ans.matrix[2][0], ans.matrix[1][0],
            ans.matrix[0][0]);


    for (int i = 0; i < dimension - 1; i++) {
        double x = plt[i][0];
        double y = plt[i][1];
        fprintf(pipe, "%f\t%f\n", x, y);
    }

    fprintf(pipe, "e\n");
    fflush(pipe);


    _pclose(pipe);
    return 0;
}
