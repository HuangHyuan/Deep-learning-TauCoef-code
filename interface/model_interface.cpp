#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cstring>

static torch::jit::script::Module model;
static bool model_loaded = false;

extern "C" {
    void run_model(float* input_data, float* output_data, int n_profiles, int n_layers, int n_channels, int n_predictors) {
        // First call to load the model
        if (!model_loaded) {
            try {
                // load TorchScript model
                model = torch::jit::load("/home/yuan/ARMS_v1.3/coefficients/ResAttention_TotOD_FY3F.pt");
                model.eval();
                model_loaded = true;
                std::cout << "ForWard Model loaded successfully" << std::endl;
            } catch (const c10::Error& e) {
                std::cerr << "Error loading model: " << e.what() << std::endl;
                return;
            }
        }
        
        // Create input tensor [batch_size, seq_len, input_size]
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(
            input_data, 
            {n_profiles, n_layers, n_predictors},  // N_Predictor=3
            options
        );
        
        // Carry out reasoning
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        try {
            auto output = model.forward(inputs);
            
            // Check the output type
            if (output.isTensor()) {
                torch::Tensor output_tensor = output.toTensor();
                
                // Ensure that the output tensor is continuous
                output_tensor = output_tensor.contiguous();
                
                // Check the output dimension
                if (output_tensor.sizes().size() != 3 || 
                    output_tensor.size(0) != n_profiles ||
                    output_tensor.size(1) != n_layers ||
                    output_tensor.size(2) != n_channels) {
                    std::cerr << "Output tensor has unexpected shape: ";
                    for (auto s : output_tensor.sizes()) 
                        std::cerr << s << " ";
                    std::cerr << std::endl;
                    return;
                }
                
                // output 
                auto output_flat = output_tensor.view({-1});
                std::memcpy(output_data, output_flat.data_ptr<float>(), 
                           output_flat.numel() * sizeof(float));
            } else {
                std::cerr << "Model output is not a tensor" << std::endl;
            }
        } catch (const c10::Error& e) {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
        }
    }
}