#include <iostream>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>

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

    EliminationMatrix(int a, int b, Matrix &matrix1) : IdentityMatrix(matrix1.n) {
        k = -(matrix1.matrix[a][b] / matrix1.matrix[b][b]);
        this->matrix[a][b] = k;
    }

    EliminationMatrix(int a, int b, Matrix &matrix1, double q) : IdentityMatrix(matrix1.n) {
        k = q;
        this->matrix[a][b] = k;
    }
};

class PermutationMatrix : public IdentityMatrix {
    int i, j;

public:

    PermutationMatrix(int a, int b, Matrix &matrix1) : IdentityMatrix(matrix1.n) {
        i = a;
        j = b;
/*        for (int q = 0; q < matrix1.n; q++) {
            for (int w = 0; w < matrix1.n; w++) {

            }
        }*/
        this->matrix[i][i] = 0;
        this->matrix[i][j] = 1;

        this->matrix[j][j] = 0;
        this->matrix[j][i] = 1;
    }
};

class InversedMatrix {
    Matrix *A;
    int n;
public:

    Matrix *inversed;

    InversedMatrix(Matrix &B) {
        A = &B;
        this->n = A->n;
        inversed = new IdentityMatrix(n);
        inversing();
    }

    void inversing() {
        Matrix z = connectMatrices();

        while (!isUpperTriangular()) {
            for (int i = 0; i < n; i++) {
                checkPermutation();
            }

            for (int i = 0; i < n; i++) {
                directElimination();
                for (int j = 0; j < n; j++) {
                    checkPermutation();
                }
            }
        }

        while (!almostIdentity()) {
            backWardElimination();
        }

        normalization();

        Matrix q = connectMatrices();
    }

    void directElimination() {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (A->matrix[j][i] != 0) {

                    EliminationMatrix eliminationMatrix(j, i, *A);
                    A = eliminationMatrix * *A;

                    EliminationMatrix eliminationMatrix2(j, i, *inversed, eliminationMatrix.k);
                    inversed = eliminationMatrix2 * *inversed;

                    Matrix q = connectMatrices();
                    return;
                }
            }
        }
    }

    void backWardElimination() {
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                if (A->matrix[j][i] != 0) {

                    EliminationMatrix eliminationMatrix(j, i, *A);
                    A = eliminationMatrix * *A;

                    EliminationMatrix eliminationMatrix2(j, i, *inversed, eliminationMatrix.k);
                    inversed = eliminationMatrix2 * *inversed;

                    Matrix q = connectMatrices();

                    return;
                }
            }
        }
    }

    void checkPermutation() {
        // i = number of the row
        for (int i = 0; i < n; i++) {
            double pivot;
            if (havesPivot(A->matrix[i], i)) {
                pivot = abs(findPivot(A->matrix[i]));
            } else {
                continue;
            }
            // j = number of the column
            for (int j = i + 1; j < n; j++) {
                if (abs(A->matrix[j][i]) > abs(pivot) and havesPivot(A->matrix[j], i)) {
                    // need to permute rows i and j
                    // A = P_ij * A

                    PermutationMatrix perm(i, j, *A);
                    A = perm * *A;

                    PermutationMatrix perm2(i, j, *inversed);
                    inversed = perm2 * *inversed;

                    Matrix q = connectMatrices();

                    return;
                }
            }
        }
    }

    void normalization() {
        for (int i = 0; i < n; i++) {
            double divider = A->matrix[i][i];
            if (divider == 0.0001) {
                continue;
            }
            A->matrix[i][i] /= divider;
            for (int j = 0; j < n; j++) {
                inversed->matrix[i][j] /= divider;
            }
        }
    }

    bool havesPivot(vector<double> &m, int row) {
        for (int i = 0; i < row; i++) {
            if (m[i] != 0) {
                return false;
            }
        }
        return true;
    }

    int findPivot(vector<double> &m) {
        for (int elem: m) {
            if (elem != 0) {
                return elem;
            }
        }
        return 0;
    }

    bool isUpperTriangular() {
        for (int i = 0; i < n; i++) {
            if (!havesPivot(A->matrix[i], i)) {
                return false;
            }
        }
        return true;
    }

    bool almostIdentity() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (A->matrix[i][j] != 0 && i != j) {
                    return false;
                }
            }
        }
        return true;
    }

    Matrix connectMatrices() {
        Matrix ans;
        ans.n = A->n;
        ans.m = A->n * 2;
        ans.matrix.resize(A->n);
        for (int i = 0; i < A->n; i++) {
            ans.matrix[i].resize(A->n * 2);
            for (int j = 0; j < A->n * 2; j++) {
                if (i > n - 1 || j > n - 1) {
                    ans.matrix[i][j] = inversed->matrix[i][j - n];
                } else {
                    ans.matrix[i][j] = A->matrix[i][j];
                }
            }
        }
        return ans;
    }
};

class ColumnVector : Matrix {
public:
    vector<double> coords;
    int dimension;
};

class LeastSquareProcess {
    Matrix *a;
    ColumnVector *b;
    int dimension;

    LeastSquareProcess(vector<double> &t, vector<double> &x, int dim) {
        b->coords = x;
        dimension = dim;

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

void fillMatrix(int polynomial_degree, Matrix &mtrx, vector<vector<double>> &original_coeff) {
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

int main() {
    int dimension, polyn_degree;
    cin >> dimension;
    vector<vector<double>> a_matrix;
    a_matrix.resize(dimension);

    vector<vector<double>> b_matrix;
    b_matrix.resize(dimension);

    for (int i = 0; i < dimension; i++) {
        double x;
        cin >> x;
        a_matrix[i].resize(1);
        a_matrix[i][0] = x;

        double y;
        cin >> y;
        b_matrix[i].resize(1);
        b_matrix[i][0] = y;
    }

    cin >> polyn_degree;

    for (int i = 0; i < dimension; i++) {
        a_matrix[i].resize(polyn_degree + 1);
        a_matrix[i][polyn_degree + 1 - 2] = a_matrix[i][0];
    }

    Matrix A(a_matrix, dimension, polyn_degree + 1);
    fillMatrix(polyn_degree, A, a_matrix);

    cout << "A:" << endl;
    cout << A;

    Matrix *b = *A.transposed() * A;
    cout << "A_T*A:" << endl;
    cout << *b;

    InversedMatrix c(*b);
    Matrix d = *c.inversed;
    cout << "(A_T*A)^-1:" << endl;
    cout << d;

    cout << "A_T*b:" << endl;
    Matrix b_vector(b_matrix, dimension, 1);
    Matrix *e = *A.transposed() * b_vector;
    cout << *e;

    cout << "x~:" << endl;
    cout << *(d * *e);
    return 0;
}
