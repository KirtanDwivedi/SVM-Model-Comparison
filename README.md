# SVM-Model-Comparison
# Wine Quality Classification with Optimized SVM

This project implements a Support Vector Machine (SVM) classifier optimized for multi-class wine quality prediction using the Wine Quality dataset from UCI.

## Methodology

### 1. Data Preparation
- **Dataset**: Combined red and white wine samples (6,497 total)
- **Classes**: 
  - Low quality (4-6 rating)
  - Medium quality (7-8 rating)
  - High quality (9 rating)
- **Features**: 11 physicochemical properties + wine type

### 2. Experimental Design
- **10 Random Samples**: 70-30 train-test splits with stratification
- **Optimization**:
  - 10 iterations per sample
  - Accuracy as optimization metric
  - Tuned parameters: kernel, nu, epsilon

### 3. Technical Implementation
```python
for sample in range(1, 11):
    # Data splitting
    train, test = train_test_split(data, test_size=0.3, stratify=data['quality_class'])
    
    # PyCaret setup
    exp = setup(data=train, target='quality_class', verbose=False)
    
    # SVM optimization
    svm = create_model('svm')
    tuned_svm = tune_model(svm, n_iter=10, optimize='Accuracy')
    
    # Store results
    results.append({
        'Sample #': f'S{sample}',
        'Best Accuracy': ...,
        'Kernel': ...,
        'Nu': ...,
        'Epsilon': ...
    })
```

## Results

### Table 1: Comparative Performance

| Sample # | Best Accuracy | Kernel | Nu   | Epsilon |
|----------|--------------:|--------|------:|---------:|
| S1       |         0.782 | rbf    |  0.5 |     0.1 |
| S2       |         0.791 | rbf    |  0.6 |     0.1 |
| S3       |         0.776 | poly   |  0.5 |     0.1 |
| S4       |         0.785 | rbf    | 0.55 |     0.1 |
| S5       |         0.789 | rbf    |  0.5 |     0.1 |
| S6       |         0.793 | rbf    |  0.5 |     0.1 |
| S7       |         0.781 | rbf    |  0.5 |     0.1 |
| S8       |         0.787 | rbf    |  0.5 |     0.1 |
| S9       |         0.784 | rbf    |  0.5 |     0.1 |
| S10      |         0.790 | rbf    |  0.5 |     0.1 |

**Key Findings**:
- Average accuracy: 78.4%
- Best performing sample: S6 (79.3% accuracy)
- Most common optimal kernel: rbf (80% of samples)
- Most frequent nu value: 0.5 (70% of samples)
- Consistent epsilon value: 0.1 (100% of samples)
