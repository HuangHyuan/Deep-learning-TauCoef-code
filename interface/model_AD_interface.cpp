#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <mutex>
#include <torch/autograd.h>

static torch::jit::script::Module model;
static bool model_loaded = false;
static std::mutex model_mutex;

extern "C" {    
    void run_model_adjoint(
        float* input_data, 
        float* output_adjoint_data,
        float* input_adjoint_data,
        int n_profiles, 
        int n_layers, 
        int n_channels,
        int n_predictors
    ) {
        std::lock_guard<std::mutex> lock(model_mutex);

        const int B = n_profiles;
        const int T = n_layers;
        const int D = n_channels;
        const int I = n_predictors;

        // First call to load the model
        if (!model_loaded) {
            try {
                // load TorchScript model
                model = torch::jit::load("/home/yuan/ARMS_v1.3/coefficients/ResAttention_TotOD_FY3F.pt");
                model.eval();
                model.to(torch::kFloat32);
                std::cout << "AD Model loaded successfully" << std::endl;
                model_loaded = true;

            } catch (const c10::Error& e) {
                std::cerr << "Error loading model or bounds: " << e.msg() << std::endl;
                return;
            } 
        }

        try {
            // Create input tensor 
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor input_tensor = torch::from_blob(input_data, {B, T, I}, options).clone();
            input_tensor = input_tensor.requires_grad_(true);

            // Forward propagation
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            torch::Tensor output = model.forward(inputs).toTensor().contiguous();

            // Initialize the Jacobian J: [B, T, D, I]
            torch::Tensor jacobian = torch::zeros({B, T, D, I}, options);

            // Calculate the Jacobian
            for (int d = 0; d < D; d++) {
                torch::Tensor grad_outputs = torch::zeros_like(output);
                grad_outputs.index({torch::indexing::Slice(), torch::indexing::Slice(), d}) = 1.0;

                auto gradients = torch::autograd::grad(
                    {output}, {input_tensor}, {grad_outputs},
                    /*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/false
                );

                jacobian.index_put_(
                    {torch::indexing::Slice(), torch::indexing::Slice(), d, torch::indexing::Slice()},
                    gradients[0]
                );
            }

            // Hard-coded cropping
            jacobian.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1, 1}, 0.0);

            // Calculate the adjoint：input_adjoint = J^T @ output_adjoint
            torch::Tensor output_adjoint_tensor = torch::from_blob(
                output_adjoint_data, {B, T, D}, options
            ).clone();

            // Use einsum: input_ad[b,t,i] = Σ_d jacobian[b,t,d,i] * output_ad[b,t,d]
            torch::Tensor input_adjoint_tensor = torch::einsum("btdi,btd->bti", {jacobian, output_adjoint_tensor});

            // output

            input_adjoint_tensor = input_adjoint_tensor.contiguous();
            std::memcpy(
                input_adjoint_data,
                input_adjoint_tensor.data_ptr<float>(),
                input_adjoint_tensor.numel() * sizeof(float)
            );

        } catch (const c10::Error& e) {
            std::cerr << "Torch error: " << e.msg() << std::endl;
            std::memset(input_adjoint_data, 0, B * T * I * sizeof(float));
        } catch (const std::exception& e) {
            std::cerr << "Standard exception: " << e.what() << std::endl;
            std::memset(input_adjoint_data, 0, B * T * I * sizeof(float));
        }
    }
}