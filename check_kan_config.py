"""Check KAN configuration (no PyTorch required)

Run: python check_kan_config.py
"""

import os
import re
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("Checking KAN Configuration")
print("=" * 60)

# Check 1: nets.py exists
print("\n[1/5] Checking nets.py file...")
if os.path.exists("nets.py"):
    print("✓ nets.py exists")
    
    with open("nets.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "class EfficientKAN" in content:
        print("✓ Found EfficientKAN class definition")
    else:
        print("✗ EfficientKAN class not found")
else:
    print("✗ nets.py not found")

# Check 2: train_burgers_1d.py imports EfficientKAN
print("\n[2/5] Checking imports in train_burgers_1d.py...")
if os.path.exists("train_burgers_1d.py"):
    with open("train_burgers_1d.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "from nets import EfficientKAN" in content:
        print("✓ EfficientKAN is imported")
    else:
        print("✗ EfficientKAN not imported")
        print("  Need to add: from nets import EfficientKAN")
else:
    print("✗ train_burgers_1d.py not found")

# Check 3: FCNetKAN class exists
print("\n[3/5] Checking FCNetKAN class definition...")
if os.path.exists("train_burgers_1d.py"):
    with open("train_burgers_1d.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "class FCNetKAN" in content:
        print("✓ Found FCNetKAN class definition")
        
        # Check for KAN usage indicator
        if "KAN is being used!" in content:
            print("✓ Found KAN usage indicator")
        else:
            print("⚠ KAN usage indicator not found (doesn't affect functionality)")
    else:
        print("✗ FCNetKAN class not found")

# Check 4: main function uses FCNetKAN
print("\n[4/5] Checking model instantiation in main()...")
if os.path.exists("train_burgers_1d.py"):
    with open("train_burgers_1d.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for model creation in main
    if "model_phys = FCNetKAN(" in content:
        print("✓ main() uses FCNetKAN")
    elif "model_phys = FCNet1D(" in content:
        print("⚠ main() uses FCNet1D (MLP version, not KAN)")
    else:
        print("✗ Could not find model instantiation")

# Check 5: List all network classes
print("\n[5/5] Summary of all network classes...")
if os.path.exists("train_burgers_1d.py"):
    with open("train_burgers_1d.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all class definitions
    class_pattern = r"class\s+(\w+)\s*\("
    classes = re.findall(class_pattern, content)
    
    network_classes = [c for c in classes if 'Net' in c or 'MLP' in c]
    print(f"Found {len(network_classes)} network classes:")
    for cls in network_classes:
        print(f"  - {cls}")

# Final summary
print("\n" + "=" * 60)
print("Configuration Check Complete")
print("=" * 60)

print("\nHow to confirm you're training with KAN:")
print("\n1. Run training command:")
print("   python train_burgers_1d.py --model fcnet ...")
print("\n2. You should see this output:")
print("   ============================================================")
print("   ✓ Using KAN version of FCNet")
print("   ============================================================")
print("   Training PHYS ...")
print("   ✓ KAN is being used!")
print("\n3. If you see these messages, you're using KAN!")
print("\nSee '如何确认使用KAN训练.md' for details")
