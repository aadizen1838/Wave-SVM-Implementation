"""
Breast Cancer Dataset Preparation for Wave-SVM
This script downloads the Breast Cancer dataset from scikit-learn,
normalizes it, and creates Train.txt and Test.txt files for Wave-SVM
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def prepare_breast_cancer_data(output_dir='data', train_size=0.7, random_state=42):
    """
    Download and prepare Breast Cancer dataset for Wave-SVM
    
    Parameters:
    -----------
    output_dir : str
        Directory to save Train.txt and Test.txt
    train_size : float
        Proportion of data for training (default 70%)
    random_state : int
        Random seed for reproducibility
    """
    
    print("=" * 60)
    print("BREAST CANCER DATASET PREPARATION")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created directory: {output_dir}/")
    
    # Step 1: Load dataset
    print("\n[1/5] Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data  # Features (30 features)
    y = data.target  # Labels (0 = malignant, 1 = benign)
    
    print(f"    Dataset loaded:")
    print(f"    - Total samples: {X.shape[0]}")
    print(f"    - Number of features: {X.shape[1]}")
    print(f"    - Classes: {np.unique(y)}")
    print(f"    - Class 1 (Benign): {np.sum(y == 1)} samples")
    print(f"    - Class 0 (Malignant): {np.sum(y == 0)} samples")
    
    # Step 2: Convert labels to -1 and 1 (required for Wave-SVM)
    print("\n[2/5] Converting labels (0,1) → (-1,1) for Wave-SVM...")
    y_converted = np.where(y == 1, 1, -1)  # 1 → 1, 0 → -1
    print(f"    - Label conversion complete")
    print(f"    - Positive class (1): {np.sum(y_converted == 1)} samples")
    print(f"    - Negative class (-1): {np.sum(y_converted == -1)} samples")
    
    # Step 3: Normalize features to [0,1]
    print("\n[3/5] Normalizing features to [0,1] range...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Min-Max normalization to [0,1]
    X_min = X_normalized.min(axis=0)
    X_max = X_normalized.max(axis=0)
    X_normalized = (X_normalized - X_min) / (X_max - X_min + 1e-8)
    
    print(f"    - Normalization complete")
    print(f"    - Feature range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
    
    # Step 4: Split into train and test
    print(f"\n[4/5] Splitting data ({train_size*100:.0f}% train, {(1-train_size)*100:.0f}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_converted, 
        train_size=train_size, 
        random_state=random_state,
        stratify=y_converted  # Keep class balance
    )
    
    print(f"    - Training samples: {X_train.shape[0]}")
    print(f"    - Test samples: {X_test.shape[0]}")
    print(f"    - Training set class balance:")
    print(f"      Class 1: {np.sum(y_train == 1)}, Class -1: {np.sum(y_train == -1)}")
    print(f"    - Test set class balance:")
    print(f"      Class 1: {np.sum(y_test == 1)}, Class -1: {np.sum(y_test == -1)}")
    
    # Step 5: Save to text files
    print(f"\n[5/5] Saving to text files...")
    
    # Prepare data with labels in first column
    train_data = np.column_stack([y_train, X_train])
    test_data = np.column_stack([y_test, X_test])
    
    # Save files
    train_file = os.path.join(output_dir, 'Train.txt')
    test_file = os.path.join(output_dir, 'Test.txt')
    
    np.savetxt(train_file, train_data, fmt='%.6f', delimiter=' ')
    np.savetxt(test_file, test_data, fmt='%.6f', delimiter=' ')
    
    print(f"    ✓ Saved: {train_file}")
    print(f"    ✓ Saved: {test_file}")
    
    # Print file info
    train_lines = len(np.loadtxt(train_file, ndmin=2))
    test_lines = len(np.loadtxt(test_file, ndmin=2))
    
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nFile Information:")
    print(f"  Train.txt: {train_lines} samples × 31 columns (1 label + 30 features)")
    print(f"  Test.txt:  {test_lines} samples × 31 columns (1 label + 30 features)")
    print(f"\nFormat of each line:")
    print(f"  <label> <feature1> <feature2> ... <feature30>")
    print(f"\nExample (Train.txt, first line):")
    first_line = train_data[0]
    print(f"  {first_line[0]:.1f} {first_line[1]:.6f} {first_line[2]:.6f} ... {first_line[30]:.6f}")
    print(f"\nReady to use with Wave-SVM in Octave!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    prepare_breast_cancer_data()