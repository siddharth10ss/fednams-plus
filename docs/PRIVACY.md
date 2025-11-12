# Privacy and Security Documentation

## Overview

FedNAMs+ implements federated learning to enable privacy-preserving collaborative training across multiple clients (simulated hospitals) without sharing raw medical imaging data.

## Privacy Guarantees

### Data Isolation

**Guarantee**: Raw chest X-ray images never leave client boundaries.

**Implementation**:
- Each client trains locally on its private data partition
- Only model parameters (weights) are transmitted to the server
- Server performs aggregation without accessing raw data
- No image data is stored on the server

**Verification**:
- Review `training/local_trainer.py` - only parameters are extracted via `get_model_parameters()`
- Review `training/orchestrator.py` - only parameters are aggregated and distributed

### Federated Averaging (FedAvg)

**Guarantee**: Server aggregates model updates without seeing individual client data.

**Implementation**:
- Clients send model parameters after local training
- Server computes weighted average of parameters
- Global model distributed back to all clients
- Process repeats for multiple rounds

**Privacy Level**: Honest-but-curious server
- Server follows protocol correctly
- Server does not attempt to infer training data from parameters
- For stronger guarantees, see Differential Privacy section below

## Privacy Risks and Mitigations

### Model Inversion Attacks

**Risk**: Adversary might attempt to reconstruct training data from model parameters.

**Mitigation**:
- Use large, diverse training sets per client (reduces memorization)
- Apply regularization (dropout, weight decay)
- Limit number of local epochs (reduces overfitting)

**Current Status**: Standard federated learning protections in place

### Membership Inference Attacks

**Risk**: Adversary might determine if a specific sample was in training data.

**Mitigation**:
- Differential privacy (optional, see below)
- Model ensembling
- Confidence thresholding

**Current Status**: Optional differential privacy support via Opacus

### Explanation Privacy

**Risk**: SHAP values might reveal information about data distribution.

**SHAP Privacy Properties** (based on literature):
- SHAP values computed locally on client data
- Only aggregated statistics shared (mean, variance)
- Individual SHAP values not transmitted
- Feature importances may reveal data distribution characteristics

**Mitigation**:
- Add noise to aggregated SHAP statistics (optional)
- Limit granularity of shared explanations
- Document potential information leakage

**References**:
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- Shokri et al. (2017): "Membership Inference Attacks Against Machine Learning Models"

## Optional: Differential Privacy

FedNAMs+ supports optional differential privacy using the Opacus library.

### Enabling Differential Privacy

```python
from opacus import PrivacyEngine

# Wrap model and optimizer
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,  # Noise level
    max_grad_norm=1.0,     # Gradient clipping
)

# Track privacy budget
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Privacy budget: ε={epsilon:.2f}")
```

### Privacy-Utility Trade-off

- **Higher noise** → Better privacy, lower accuracy
- **Lower noise** → Worse privacy, higher accuracy

**Recommended Settings**:
- `noise_multiplier`: 0.5 - 1.5
- `max_grad_norm`: 1.0
- `delta`: 1e-5
- Target `epsilon`: < 10 for reasonable privacy

## Audit Logging

All data access and model transmission events are logged for audit purposes.

### Logged Events

1. **Data Access**:
   - Client data loading
   - Dataset partitioning
   - Sample counts per client

2. **Model Transmission**:
   - Parameter uploads from clients
   - Parameter downloads to clients
   - Aggregation operations

3. **Training Events**:
   - Training round start/end
   - Client selection
   - Convergence detection

### Log Location

Logs are saved to:
- `outputs/{experiment_name}/logs/experiment.log`
- `outputs/{experiment_name}/logs/training.log`

### Log Format

```
2024-11-12 15:30:45 - training.orchestrator - INFO - Round 1/50
2024-11-12 15:30:45 - training.orchestrator - INFO - Selected 5 clients: [0, 1, 2, 3, 4]
2024-11-12 15:31:20 - training.orchestrator - INFO - Aggregating parameters from 5 clients
2024-11-12 15:31:22 - training.orchestrator - INFO - Distributing global model to all clients
```

## Compliance Considerations

### GDPR Compliance

**Relevant Aspects**:
- ✅ Data minimization: Only model parameters transmitted
- ✅ Purpose limitation: Data used only for model training
- ✅ Storage limitation: No long-term storage of raw data on server
- ✅ Transparency: Audit logs track all operations
- ⚠️ Right to erasure: Challenging in federated setting (model retains learned patterns)

### HIPAA Compliance

**Relevant Aspects**:
- ✅ Access controls: Client data isolated
- ✅ Audit controls: Comprehensive logging
- ✅ Transmission security: Parameters only (no PHI)
- ⚠️ De-identification: Model parameters may encode patient information

**Note**: This system uses publicly available, de-identified NIH dataset. For real clinical deployment, additional safeguards required.

## Security Best Practices

### Secure Communication

**Current**: Parameters transmitted in plain PyTorch tensors

**Recommended for Production**:
- Use TLS/SSL for all client-server communication
- Implement secure aggregation protocols
- Add authentication and authorization

### Secure Aggregation (Future Enhancement)

**Concept**: Server aggregates without seeing individual client updates

**Implementation Options**:
- Secure multi-party computation (MPC)
- Homomorphic encryption
- Secret sharing schemes

**Libraries**:
- PySyft
- TenSEAL
- CrypTen

## Privacy Impact Assessment

### Data Flow

```
Client 1 Data → Local Model → Parameters → Server
Client 2 Data → Local Model → Parameters → Server
Client 3 Data → Local Model → Parameters → Server
                                    ↓
                            Aggregated Model
                                    ↓
                    Distributed to All Clients
```

### Information Leakage Analysis

**Low Risk**:
- Raw images never transmitted
- Only aggregated parameters shared
- Multiple clients contribute to each update

**Medium Risk**:
- Model parameters encode training data patterns
- Feature importances reveal data characteristics
- Repeated updates may leak information

**Mitigation**:
- Use differential privacy for stronger guarantees
- Limit number of training rounds
- Add noise to aggregated statistics

## Recommendations for Clinical Deployment

1. **Enable Differential Privacy**: Add DP-SGD for formal privacy guarantees
2. **Secure Communication**: Use TLS/SSL for all transmissions
3. **Access Controls**: Implement authentication and authorization
4. **Audit Trail**: Maintain comprehensive logs
5. **Regular Security Audits**: Review for vulnerabilities
6. **Compliance Review**: Consult with legal/compliance teams
7. **Informed Consent**: Ensure patients consent to federated learning

## References

1. McMahan et al. (2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Kairouz et al. (2021): "Advances and Open Problems in Federated Learning"
3. Abadi et al. (2016): "Deep Learning with Differential Privacy"
4. Shokri & Shmatikov (2015): "Privacy-Preserving Deep Learning"

## Contact

For privacy concerns or questions, please open an issue on GitHub or contact the maintainers.
