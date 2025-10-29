# MultiFeatureExplorer Refactoring: Unified Single and Multi-Node Support

**Date**: 2025-10-28  
**Author**: Trading Research Team

## Overview

Refactored `MultiFeatureExplorer` to support both single and multiple bias nodes in a **DRY (Don't Repeat Yourself)** and **SIMPLE** architecture. The previous `MultiNodeExplorer` class was eliminated as it duplicated too much code.

## Key Changes

### 1. Unified Class Structure

**Before**: Two separate classes
- `MultiFeatureExplorer` - for single bias node
- `MultiNodeExplorer` - for multiple bias nodes (duplicated code)

**After**: One unified class
- `MultiFeatureExplorer` - handles both single and multiple bias nodes

### 2. Dual-Mode Architecture

The class now operates in two modes, detected automatically:

#### Single-Node Mode
```python
explorer = MultiFeatureExplorer.from_bias_node(
    module_name='rsi',
    ticker=Ticker.SPY,
    params={'lookback': 14}
)
# explorer.is_multi_node = False
# explorer.explorers = {'rsi': FeatureExplorer}
```

#### Multi-Node Mode
```python
explorer = MultiFeatureExplorer.from_bias_nodes(
    nodes=[
        {'module_name': 'rsi', 'params': {'lookback': 14}},
        {'module_name': 'cmma', 'params': {'lookback': 20}}
    ],
    ticker=Ticker.SPY
)
# explorer.is_multi_node = True
# explorer.nodes = {'rsi_lookback14': {...}, 'cmma_lookback20': {...}}
# explorer.explorers = {'rsi_lookback14.rsi': FeatureExplorer, ...}
```

### 3. Internal Data Structure

**Single-Node Mode**:
```python
explorers = {
    'feature1': FeatureExplorer,
    'feature2': FeatureExplorer
}
```

**Multi-Node Mode**:
```python
nodes = {
    'node1': {
        'feature1': FeatureExplorer,
        'feature2': FeatureExplorer
    },
    'node2': {
        'feature3': FeatureExplorer
    }
}

# Flattened for easy access:
explorers = {
    'node1.feature1': FeatureExplorer,
    'node1.feature2': FeatureExplorer,
    'node2.feature3': FeatureExplorer
}
```

### 4. New Attributes

- `is_multi_node`: bool - True if managing multiple nodes
- `nodes`: dict or None - Node-organized explorers (multi-node only)
- `node_names`: list or None - List of node names (multi-node only)
- `n_nodes`: int - Number of nodes (1 for single-node)

### 5. Enhanced Access Patterns

#### Single-Node Mode
```python
# Direct feature access
feature_exp = explorer['rsi']

# Iteration over features
for feature_name, feature_explorer in explorer:
    print(feature_name)
```

#### Multi-Node Mode
```python
# Access by node name (returns dict of FeatureExplorers)
node_features = explorer['cmma_lookback20']

# Access specific feature with dot notation
feature_exp = explorer['cmma_lookback20.cmma']

# Access specific feature with nested dict
feature_exp = explorer['cmma_lookback20']['cmma']

# Iteration over nodes
for node_name, node_explorers_dict in explorer:
    print(node_name, len(node_explorers_dict))
```

### 6. Method Delegation

The `_create_multi_method()` now handles both modes:

**Single-Node Mode**: Returns flat dict
```python
results = explorer.binning_permutation_test(n_bins=3, nreps=100)
# Returns: {'feature1': result, 'feature2': result}
```

**Multi-Node Mode**: Returns nested dict
```python
results = explorer.binning_permutation_test(n_bins=3, nreps=100)
# Returns: {'node1': {'feature1': result}, 'node2': {'feature2': result}}
```

### 7. New Helper Methods

```python
# Get flat dict of all features (works in both modes)
all_features = explorer.get_all_features()

# Get summary DataFrame
summary = explorer.get_feature_summary()
# Single-node: columns = [feature_name, n_samples, mean, std, min, max]
# Multi-node: columns = [node_name, feature_name, n_samples, mean, std, min, max]
```

### 8. String Representations

**Single-Node Mode**:
```python
print(explorer)
# MultiFeatureExplorer(mode='single-node', n_features=3, features=['rsi', 'rsi_ma', 'rsi_std'])
```

**Multi-Node Mode**:
```python
print(explorer)
# MultiFeatureExplorer (multi-node mode) with 2 nodes:
#   - rsi_lookback14: 3 features
#   - cmma_lookback20: 1 features
# Total: 4 features
```

## Benefits

### 1. DRY Principle
- **Eliminated ~300 lines of duplicate code** from `MultiNodeExplorer`
- Single source of truth for feature extraction and management
- All logic centralized in one class

### 2. Simplicity
- One class to learn instead of two
- Same API for both single and multiple nodes
- Automatic mode detection - no manual configuration

### 3. Backward Compatibility
- All existing single-node code continues to work
- `from_bias_node()` method unchanged
- Same method delegation pattern

### 4. Flexibility
- Easy to switch between single and multi-node
- Supports custom node naming
- Feature filtering per node
- Multiple access patterns (hierarchical, flat, dot notation)

### 5. Maintainability
- Single class to maintain and test
- Consistent behavior across modes
- Clear separation of concerns

## Migration Guide

### Old Code (MultiNodeExplorer)
```python
from feature_selection.multi_node_explorer import MultiNodeExplorer

explorer = MultiNodeExplorer.from_bias_nodes(
    nodes=[...],
    ticker=Ticker.SPY
)
```

### New Code (MultiFeatureExplorer)
```python
from feature_selection.multi_feature_explorer import MultiFeatureExplorer

explorer = MultiFeatureExplorer.from_bias_nodes(
    nodes=[...],
    ticker=Ticker.SPY
)
```

**That's it!** The API is identical, just change the import and class name.

## Examples

See `examples/multi_feature_explorer_demo.py` for comprehensive examples of:
1. Basic multi-node extraction
2. Custom node names
3. Feature filtering
4. Multiple tickers
5. Method delegation
6. Hierarchical access
7. Permutation testing
8. Single-node mode (backward compatibility)

## Technical Details

### Mode Detection
```python
def __init__(self, explorers):
    # Detect multi-node if values are dicts
    if explorers and isinstance(next(iter(explorers.values())), dict):
        self.is_multi_node = True
        # ... multi-node setup
    else:
        self.is_multi_node = False
        # ... single-node setup
```

### Flattening Strategy
Multi-node explorers are stored in two ways:
1. **Hierarchical** (`self.nodes`): Preserves node structure
2. **Flat** (`self.explorers`): Node-prefixed keys for easy iteration

This allows both hierarchical access and flat iteration without code duplication.

## Performance

No performance impact:
- Same underlying FeatureExplorer instances
- Minimal overhead for mode detection (one-time at initialization)
- Same method delegation mechanism

## Testing

All existing tests pass. The refactoring is purely structural with no behavioral changes for single-node mode.

## Future Enhancements

Potential additions:
1. Node grouping/tagging for organizing large feature sets
2. Cross-node feature comparison methods
3. Node-level filtering and selection
4. Parallel processing across nodes

## Summary

This refactoring achieves the goal of **DRY and SIMPLE** code by:
- ✅ Eliminating duplicate code (~300 lines removed)
- ✅ Unifying single and multi-node under one class
- ✅ Maintaining backward compatibility
- ✅ Providing flexible access patterns
- ✅ Keeping the same intuitive API

The result is a more maintainable, easier to understand, and more powerful feature exploration system.
