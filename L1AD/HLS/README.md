# HLS4ML Neural Network FPGA Synthesis Workflow

This repository contains the workflow for converting a Keras neural network model to FPGA synthesis using hls4ml.

## Dependencies

- TensorFlow
- hls4ml
- QKeras
- Vitis HLS 

## Workflow Steps

### 1. Data Loading (optional)
- Load preprocessed data from H5 file
- Data includes various topology features and weights

### 2. Model Loading
- Load pre-trained Keras model with QKeras custom objects
- Model architecture includes QDense layers and QBatchNormalization

### 3. HLS Configuration
- Generate initial HLS config from Keras model
- Configure precision for each layer:
  - Input: 15,6 fixed-point
  - Dense layers: 5,1 fixed-point weights/biases
  - Batch normalization: Various precision settings
  - Output (z_mean): 6,2 fixed-point weights/biases

### 4. Synthesis Configuration
- Backend: Vitis
- IO Type: Parallel
- Clock Period: 25ns
- Target FPGA: xcvu9p-flga2104-2-e
- Project Name: Topo2A_AD_proj

### 5. HLS4ML Conversion
- Convert Keras model to HLS
- Compile HLS model
- Build project (without C simulation)

## Important Settings

```python
hls_config['Model']['ReuseFactor'] = 1
hls_config['LayerName'][layer]['Strategy'] = 'Latency'
```

All layers are configured for latency-optimized implementation.

## Usage

1. Ensure environment variables are set for Vitis HLS backend
2. Run the script to perform the conversion
3. The synthesized design will be generated in 'FiorDiLatte_Topo2A_trigger_VAE' directory

## Notes

- The model uses fixed-point precision to optimize FPGA resource usage
- Layer-specific precision settings are crucial for balancing accuracy and resource utilization
- Synthesis is configured for a Xilinx UltraScale+ VU9P FPGA
