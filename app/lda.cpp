#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <map>
#include <tuple>
#include <vector>

namespace py = pybind11;
using namespace torch::indexing;

std::map<std::string, torch::Tensor> lda_fit_transform(torch::Tensor X, torch::Tensor y) {
    // Convertir tensores de entrada a matrices de Eigen
    Eigen::Map<Eigen::MatrixXd> X_eigen(X.data_ptr<double>(), X.size(0), X.size(1));
    Eigen::Map<Eigen::VectorXd> y_eigen(y.data_ptr<double>(), y.size(0));

    // Obtener dimensiones
    int n_samples = X_eigen.rows();
    int n_features = X_eigen.cols();
    
    auto unique_result = at::_unique(y, /*sorted=*/true, /*return_inverse=*/false);
    at::Tensor unique_labels = std::get<0>(unique_result);
    int n_classes = unique_labels.size(0);

    // Calcular la media global
    Eigen::VectorXd mean_global = X_eigen.colwise().mean();

    // Inicializar matrices de dispersión
    Eigen::MatrixXd S_w = Eigen::MatrixXd::Zero(n_features, n_features);
    Eigen::MatrixXd S_b = Eigen::MatrixXd::Zero(n_features, n_features);

    // Calcular matrices de dispersión intra-clase y entre-clases
    for (int i = 0; i < n_classes; ++i) {
        auto class_mask = (y == unique_labels[i]);
    
        using namespace torch::indexing;
        std::vector<TensorIndex> indices = {class_mask};
        auto X_class = X.index(indices);
        Eigen::Map<Eigen::MatrixXd> X_class_eigen(X_class.data_ptr<double>(), X_class.size(0), X_class.size(1));
        int n_class_samples = X_class_eigen.rows();

        // Media de la clase
        Eigen::VectorXd mean_class = X_class_eigen.colwise().mean();

        // Dispersión intra-clase
        Eigen::MatrixXd X_centered = X_class_eigen.rowwise() - mean_class.transpose();
        S_w += X_centered.transpose() * X_centered;

        // Dispersión entre-clases
        Eigen::VectorXd mean_diff = mean_class - mean_global;
        S_b += n_class_samples * (mean_diff * mean_diff.transpose());
    }

    // Resolver el problema de valores propios para S_w^(-1) * S_b
    Eigen::MatrixXd S_w_inv = S_w.inverse();
    Eigen::MatrixXd mat = S_w_inv * S_b;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);

    // Obtener los autovectores (componentes lineales)
    Eigen::MatrixXd components = solver.eigenvectors().rightCols(n_classes - 1);

    // Transformar X a las nuevas componentes
    Eigen::MatrixXd X_new = X_eigen * components;

    // Convertir resultados a tensores de PyTorch
    torch::Tensor components_tensor = torch::from_blob(components.data(), {components.rows(), components.cols()}, torch::kDouble).clone();
    torch::Tensor X_new_tensor = torch::from_blob(X_new.data(), {X_new.rows(), X_new.cols()}, torch::kDouble).clone();

    return {{"components", components_tensor}, {"transformed", X_new_tensor}};
}

PYBIND11_MODULE(lda_module, m) {
    m.def("lda_fit_transform", &lda_fit_transform, "Realiza LDA y transforma los datos",
          py::arg("X"), py::arg("y"));
}
