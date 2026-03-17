#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cstring>

#include <torch/autograd.h>

static torch::jit::script::Module model;
static bool model_loaded = false;

extern "C" {
    void run_model_tangent_linear(
        float* input_data, 
        float* input_tl_data,
        float* output_tl_data,
        int n_profiles, 
        int n_layers, 
        int n_channels,
        int n_predictors
    ) {
        const int B = n_profiles;
        const int T = n_layers;  // H
        const int D = n_channels; // W
        const int I = n_predictors;

        // First call to load the model
        if (!model_loaded) {
            try {
                // load TorchScript model
                model = torch::jit::load("/home/yuan/ARMS_v1.3/coefficients/ResAttention_TotOD_FY3F.pt");
                model.eval();
                model.to(torch::kFloat32);
                std::cout << "TL Model loaded successfully" << std::endl;
                model_loaded = true;

            } catch (const c10::Error& e) {
                std::cerr << "Error loading model or bounds: " << e.msg() << std::endl;
                return;
            } 
        }

        try {
            // Create input tensor [batch_size, seq_len, input_size]
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor input_tensor = torch::from_blob(input_data, {B, T, I}, options).clone();
            input_tensor = input_tensor.requires_grad_(true);

            // Forward propagation
            std::vector<torch::jit::IValue> inputs_fw;
            inputs_fw.push_back(input_tensor);
            torch::Tensor output = model.forward(inputs_fw).toTensor().contiguous();

            // Jacobian J: [B, T, D, I]
            torch::Tensor jacobian = torch::zeros({B, T, D, I}, options);

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
            jacobian.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1, 1}, 0.0); // 示例

            // caculate output_tl = J @ input_tl
            torch::Tensor input_tl_tensor = torch::from_blob(input_tl_data, {B, T, I}, options).clone();
            torch::Tensor output_tl = torch::einsum("btdi,bti->btd", {jacobian, input_tl_tensor});

            // output
            output_tl = output_tl.contiguous();
            std::memcpy(
                output_tl_data,
                output_tl.data_ptr<float>(),
                output_tl.numel() * sizeof(float)
            );

        } catch (const c10::Error& e) {
            std::cerr << "Torch error: " << e.msg() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard exception: " << e.what() << std::endl;
        }
    }
}