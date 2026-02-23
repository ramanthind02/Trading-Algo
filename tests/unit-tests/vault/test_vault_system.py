"""
Tests for Vault System

Unit tests for the vault system including:
- Direction enum
- BaseModel (feature extraction + binning)
- Vault manager functions

Author: Trading Research Team
Date: 2025-01-07
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from utils.core.enums import Direction, TimeFrame, Ticker
from feature_selection.base_models.feature_base_model import BaseModel
from feature_selection.base_models import ContinuousBinningModel, DecisionTreeBinningModel
from ensemble.vault_manager import (
    generate_model_id,
    create_ensemble_directory,
    get_ensemble_path,
    list_ensembles,
    add_feature_to_ensemble,
    load_feature_base_models,
    update_base_model_fitted_params,
    remove_base_model_variant,
    list_features,
    get_bias_node_specs,
    get_all_base_model_names,
    initialize_vault,
    validate_ensemble_directory
)


class TestDirectionEnum:
    """Test suite for Direction enum."""
    
    def test_direction_values(self):
        """Test Direction enum values."""
        assert Direction.LONG.value == 'long'
        assert Direction.SHORT.value == 'short'
        assert Direction.LONG_SHORT.value == 'long_short'

    def test_direction_string_conversion(self):
        """Test Direction string conversion."""
        assert str(Direction.LONG) == 'long'
        assert str(Direction.SHORT) == 'short'
        assert str(Direction.LONG_SHORT) == 'long_short'

    def test_from_string_valid(self):
        """Test from_string with valid inputs."""
        assert Direction.from_string('long') == Direction.LONG
        assert Direction.from_string('LONG') == Direction.LONG
        assert Direction.from_string('Long') == Direction.LONG
        assert Direction.from_string('short') == Direction.SHORT
        assert Direction.from_string('SHORT') == Direction.SHORT
        assert Direction.from_string('long_short') == Direction.LONG_SHORT
        assert Direction.from_string('both') == Direction.LONG_SHORT

    def test_from_string_invalid(self):
        """Test from_string with invalid inputs."""
        with pytest.raises(ValueError, match="Invalid direction"):
            Direction.from_string('invalid')
        with pytest.raises(ValueError, match="Invalid direction"):
            Direction.from_string('neutral')

    def test_sorting(self):
        """Test Direction sorting (by definition order: LONG, SHORT, LONG_SHORT)."""
        directions = [Direction.SHORT, Direction.LONG_SHORT, Direction.LONG]
        assert sorted(directions) == [Direction.LONG, Direction.SHORT, Direction.LONG_SHORT]


class TestModelIDGeneration:
    """Test suite for model ID generation."""
    
    def test_continuous_binning_model_id(self):
        """Test model ID generation for ContinuousBinningModel."""
        params = {'n_bins': 3, 'selection_metric': 'sortino', 'strategy': 'long'}
        model_id = generate_model_id('continuous_binning', params)
        assert model_id == 'continuous_binning_3'
    
    def test_continuous_binning_model_id_different_bins(self):
        """Test model ID generation with different bin counts."""
        params_3 = {'n_bins': 3, 'selection_metric': 'sortino'}
        params_5 = {'n_bins': 5, 'selection_metric': 'sortino'}
        
        id_3 = generate_model_id('continuous_binning', params_3)
        id_5 = generate_model_id('continuous_binning', params_5)
        
        assert id_3 == 'continuous_binning_3'
        assert id_5 == 'continuous_binning_5'
        assert id_3 != id_5
    
    def test_decision_tree_binning_model_id(self):
        """Test model ID generation for DecisionTreeBinningModel."""
        params = {'n_bins': 5, 'min_samples_leaf_pct': 0.05, 'selection_metric': 'sortino'}
        model_id = generate_model_id('decision_tree_binning', params)
        assert model_id == 'decision_tree_binning_5'


@pytest.fixture
def bias_node_spec():
    """Create bias node spec for RSI."""
    return {
        'module_name': 'rsi',
        'timeframes': [TimeFrame.D],
        'params': {'lookback': 14}
    }


class TestBaseModel:
    """Test suite for BaseModel (feature extraction + binning)."""
    
    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        candles = []
        
        for i, date in enumerate(dates):
            from utils.core.models import Candle
            candle = Candle(
                datetime=date,
                open=100.0 + i * 0.1,
                high=101.0 + i * 0.1,
                low=99.0 + i * 0.1,
                close=100.5 + i * 0.1,
                volume=1000000.0,
                ticker=Ticker.ES,
                tf=TimeFrame.D
            )
            candles.append(candle)
        
        return candles
    
    @pytest.fixture
    def candles_df(self, sample_candles):
        """Create candles DataFrame."""
        data = []
        for candle in sample_candles:
            data.append({
                'datetime': candle.datetime,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'ticker': candle.ticker,
                'timeframe': candle.tf
            })
        return pd.DataFrame(data)
    
    @pytest.fixture
    def target_data(self, candles_df):
        """Create target data (returns)."""
        # Simple returns based on price changes
        returns = candles_df['close'].pct_change().fillna(0)
        return pd.Series(returns.values, index=candles_df['datetime'])
    
    def test_base_model_initialization(self, bias_node_spec):
        """Test BaseModel initialization."""
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        assert base_model.bias_node_spec == bias_node_spec
        assert base_model.binning_model == binning_model
        assert base_model.ticker == Ticker.ES
        assert len(base_model.bias_nodes) == 1
        assert TimeFrame.D in base_model.bias_nodes
    
    def test_add_candle(self, bias_node_spec, sample_candles):
        """Test adding candles to BaseModel."""
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        # Add a candle
        candle = sample_candles[0]
        base_model.add_candle(candle, TimeFrame.D)
        
        # Check that feature value was tracked
        assert candle.datetime in base_model._feature_values
    
    def test_get_feature(self, bias_node_spec, sample_candles):
        """Test feature extraction from bias nodes."""
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        # Add several candles
        for candle in sample_candles[:50]:
            base_model.add_candle(candle, TimeFrame.D)
        
        # Extract feature
        feature = base_model.get_feature()
        
        assert isinstance(feature, pd.Series)
        assert len(feature) == 50
        assert base_model.feature_column is not None
        assert 'rsi' in base_model.feature_column.lower()
    
    def test_fit(self, bias_node_spec, candles_df, target_data):
        """Test BaseModel fitting."""
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        # Fit model
        base_model.fit(candles_df, target_data)
        
        # Check that binning model is fitted
        assert base_model.binning_model.is_fitted_
        assert base_model.feature_column is not None
    
    def test_predict(self, bias_node_spec, candles_df, target_data):
        """Test BaseModel prediction."""
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        # Fit first
        base_model.fit(candles_df[:80], target_data[:80])
        
        # Predict on remaining data
        predictions = base_model.predict(candles_df[80:], strategy='long')
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(candles_df[80:])
        assert predictions.dtype in [np.int64, np.float64]


class TestVaultManager:
    """Test suite for vault manager functions."""
    
    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault directory for testing."""
        temp_dir = tempfile.mkdtemp()
        vault_root = Path(temp_dir) / 'vault'
        vault_root.mkdir()
        yield str(vault_root)
        shutil.rmtree(temp_dir)
    
    def test_initialize_vault(self, temp_vault):
        """Test vault initialization."""
        initialize_vault(temp_vault)
        
        vault_path = Path(temp_vault)
        assert vault_path.exists()
        assert (vault_path / 'README.md').exists()
    
    def test_create_ensemble_directory(self, temp_vault):
        """Test ensemble directory creation."""
        initialize_vault(temp_vault)
        
        ensemble_dir = create_ensemble_directory(
            vault_root=temp_vault,
            timeframe=TimeFrame.D,
            ensemble_name='test_ensemble',
            direction=Direction.LONG
        )
        
        ensemble_path = Path(ensemble_dir)
        assert ensemble_path.exists()
        assert (ensemble_path / 'features').exists()
        assert ensemble_path.name == 'test_ensemble_long'
    
    def test_create_ensemble_directory_duplicate(self, temp_vault):
        """Test that creating duplicate ensemble raises error."""
        initialize_vault(temp_vault)
        
        create_ensemble_directory(
            vault_root=temp_vault,
            timeframe=TimeFrame.D,
            ensemble_name='test_ensemble',
            direction=Direction.LONG
        )
        
        # Try to create again
        with pytest.raises(ValueError, match="already exists"):
            create_ensemble_directory(
                vault_root=temp_vault,
                timeframe=TimeFrame.D,
                ensemble_name='test_ensemble',
                direction=Direction.LONG
            )
    
    def test_get_ensemble_path(self, temp_vault):
        """Test getting ensemble path."""
        initialize_vault(temp_vault)
        
        path = get_ensemble_path(
            vault_root=temp_vault,
            timeframe=TimeFrame.D,
            ensemble_name='test_ensemble',
            direction=Direction.LONG
        )
        
        expected = Path(temp_vault) / 'D' / 'test_ensemble_long'
        assert path == str(expected)
    
    def test_list_ensembles(self, temp_vault):
        """Test listing ensembles."""
        initialize_vault(temp_vault)
        
        # Create multiple ensembles
        create_ensemble_directory(temp_vault, TimeFrame.D, 'ensemble1', Direction.LONG)
        create_ensemble_directory(temp_vault, TimeFrame.D, 'ensemble2', Direction.SHORT)
        create_ensemble_directory(temp_vault, TimeFrame.W, 'ensemble3', Direction.LONG)
        
        df = list_ensembles(temp_vault)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'ensemble_name' in df.columns
        assert 'timeframe' in df.columns
        assert 'direction' in df.columns
    
    def test_add_feature_to_ensemble(self, temp_vault, bias_node_spec):
        """Test adding feature to ensemble."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Create and fit a base model
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        # Create minimal candles and targets for fitting
        np.random.seed(42)
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        candles_df = pd.DataFrame({
            'datetime': dates,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.0 + np.arange(50) * 0.1,
            'volume': 1000000.0,
            'ticker': Ticker.ES,
            'timeframe': TimeFrame.D
        })
        target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
        
        base_model.fit(candles_df, target_data)
        
        # Add to vault
        model_id = add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column=base_model.feature_column,
            bias_node_spec=bias_node_spec,
            base_model=base_model
        )
        
        assert model_id == 'continuous_binning_3'
        
        # Check file was created
        feature_file = Path(ensemble_dir) / 'features' / f"{base_model.feature_column}.json"
        assert feature_file.exists()
        
        # Check file content
        with open(feature_file, 'r') as f:
            config = json.load(f)
        
        assert config['feature_column'] == base_model.feature_column
        assert len(config['base_models']) == 1
        assert config['base_models'][0]['model_id'] == model_id
    
    def test_add_feature_strategy_mismatch(self, temp_vault, bias_node_spec):
        """Test that strategy mismatch raises error."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Create model with wrong strategy
        binning_model = ContinuousBinningModel(n_bins=3, strategy='short')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        # Try to add - should fail
        with pytest.raises(ValueError, match="does not match"):
            add_feature_to_ensemble(
                ensemble_dir=ensemble_dir,
                feature_column='test_feature',
                bias_node_spec=bias_node_spec,
                base_model=base_model
            )
    
    def test_load_feature_base_models(self, temp_vault, bias_node_spec):
        """Test loading base models from vault."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Create and save a model
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        candles_df = pd.DataFrame({
            'datetime': dates,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.0 + np.arange(50) * 0.1,
            'volume': 1000000.0,
            'ticker': Ticker.ES,
            'timeframe': TimeFrame.D
        })
        target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
        
        base_model.fit(candles_df, target_data)
        
        model_id = add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column=base_model.feature_column,
            bias_node_spec=bias_node_spec,
            base_model=base_model
        )
        
        # Load models
        models = load_feature_base_models(
            ensemble_dir=ensemble_dir,
            feature_column=base_model.feature_column
        )
        
        # Models are now keyed by (ticker, model_id) tuple
        key = (Ticker.ES, model_id)
        assert key in models
        loaded_model = models[key]
        assert isinstance(loaded_model, BaseModel)
        assert loaded_model.binning_model.is_fitted_
        assert loaded_model.ticker == Ticker.ES
    
    def test_update_base_model_fitted_params(self, temp_vault, bias_node_spec):
        """Test updating fitted parameters."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Create and save unfitted model
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        candles_df = pd.DataFrame({
            'datetime': dates,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.0 + np.arange(50) * 0.1,
            'volume': 1000000.0,
            'ticker': Ticker.ES,
            'timeframe': TimeFrame.D
        })
        target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
        
        model_id = add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column='test_feature',
            bias_node_spec=bias_node_spec,
            base_model=base_model
        )
        
        # Fit the model
        base_model.fit(candles_df, target_data)
        
        # Update fitted params
        update_base_model_fitted_params(
            ensemble_dir=ensemble_dir,
            feature_column='test_feature',
            model_id=model_id,
            fitted_params={
                'thresholds': base_model.binning_model.thresholds_.tolist(),
                'best_long_bin': base_model.binning_model.best_long_bin_,
                'best_short_bin': base_model.binning_model.best_short_bin_,
                'bin_stats': base_model.binning_model.bin_stats_
            },
            train_start='2020-01-01',
            train_end='2020-02-20'
        )
        
        # Verify update
        models = load_feature_base_models(ensemble_dir, 'test_feature', fitted_only=True)
        key = (Ticker.ES, model_id)
        assert key in models
        assert models[key].binning_model.is_fitted_
    
    def test_remove_base_model_variant(self, temp_vault, bias_node_spec):
        """Test removing base model variant."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Add a model
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        model_id = add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column='test_feature',
            bias_node_spec=bias_node_spec,
            base_model=base_model
        )
        
        # Remove it
        remove_base_model_variant(
            ensemble_dir=ensemble_dir,
            feature_column='test_feature',
            model_id=model_id
        )
        
        # Verify removal
        models = load_feature_base_models(ensemble_dir, 'test_feature')
        key = (Ticker.ES, model_id)
        assert key not in models
    
    def test_list_features(self, temp_vault, bias_node_spec):
        """Test listing features in ensemble."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Add multiple features
        for i in range(3):
            binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
            base_model = BaseModel(
                feature_config={'bias_node_spec': bias_node_spec},
                binning_model=binning_model,
                tickers=[Ticker.ES]
            )
            
            dates = pd.date_range('2020-01-01', periods=50, freq='D')
            candles_df = pd.DataFrame({
                'datetime': dates,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0 + np.arange(50) * 0.1,
                'volume': 1000000.0,
                'ticker': Ticker.ES,
                'timeframe': TimeFrame.D
            })
            target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
            
            base_model.fit(candles_df, target_data)
            
            add_feature_to_ensemble(
                ensemble_dir=ensemble_dir,
                feature_column=f'test_feature_{i}',
                bias_node_spec=bias_node_spec,
                base_model=base_model
            )
        
        # List features
        df = list_features(ensemble_dir)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'feature_column' in df.columns
        assert 'n_base_models' in df.columns
    
    def test_get_bias_node_specs(self, temp_vault, bias_node_spec):
        """Test getting bias node specs."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Add a feature
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        candles_df = pd.DataFrame({
            'datetime': dates,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.0 + np.arange(50) * 0.1,
            'volume': 1000000.0,
            'ticker': Ticker.ES,
            'timeframe': TimeFrame.D
        })
        target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
        
        base_model.fit(candles_df, target_data)
        
        add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column=base_model.feature_column,
            bias_node_spec=bias_node_spec,
            base_model=base_model
        )
        
        # Get bias node specs
        specs = get_bias_node_specs(ensemble_dir)
        
        assert len(specs) == 1
        assert specs[0]['module_name'] == 'rsi'
    
    def test_get_all_base_model_names(self, temp_vault, bias_node_spec):
        """Test getting all base model names."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Add multiple models to same feature
        for n_bins in [3, 5]:
            binning_model = ContinuousBinningModel(n_bins=n_bins, strategy='long')
            base_model = BaseModel(
                feature_config={'bias_node_spec': bias_node_spec},
                binning_model=binning_model,
                tickers=[Ticker.ES]
            )
            
            dates = pd.date_range('2020-01-01', periods=50, freq='D')
            candles_df = pd.DataFrame({
                'datetime': dates,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0 + np.arange(50) * 0.1,
                'volume': 1000000.0,
                'ticker': Ticker.ES,
                'timeframe': TimeFrame.D
            })
            target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
            
            base_model.fit(candles_df, target_data)
            
            add_feature_to_ensemble(
                ensemble_dir=ensemble_dir,
                feature_column=base_model.feature_column,
                bias_node_spec=bias_node_spec,
                base_model=base_model
            )
        
        # Get all model names
        model_names = get_all_base_model_names(ensemble_dir)
        
        assert len(model_names) == 2
        assert all('::' in name for name in model_names)
        assert any('continuous_binning_3' in name for name in model_names)
        assert any('continuous_binning_5' in name for name in model_names)
    
    def test_validate_ensemble_directory(self, temp_vault, bias_node_spec):
        """Test ensemble directory validation."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Add a valid feature
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        candles_df = pd.DataFrame({
            'datetime': dates,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.0 + np.arange(50) * 0.1,
            'volume': 1000000.0,
            'ticker': Ticker.ES,
            'timeframe': TimeFrame.D
        })
        target_data = pd.Series(np.random.randn(50) * 0.01, index=dates)
        
        base_model.fit(candles_df, target_data)
        
        add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column=base_model.feature_column,
            bias_node_spec=bias_node_spec,
            base_model=base_model
        )
        
        # Validate - should pass
        validate_ensemble_directory(ensemble_dir)
        
        # Test invalid directory
        with pytest.raises(ValueError):
            validate_ensemble_directory('/nonexistent/path')
    
    def test_multiple_tickers_in_feature_spec(self, temp_vault, bias_node_spec):
        """Test storing and loading multiple tickers in feature spec."""
        initialize_vault(temp_vault)
        ensemble_dir = create_ensemble_directory(
            temp_vault, TimeFrame.D, 'test_ensemble', Direction.LONG
        )
        
        # Create base model for ES
        binning_model = ContinuousBinningModel(n_bins=3, strategy='long')
        base_model_es = BaseModel(
            feature_config={'bias_node_spec': bias_node_spec},
            binning_model=binning_model,
            tickers=[Ticker.ES]
        )
        
        # Add feature with multiple tickers
        model_id = add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column='test_feature',
            bias_node_spec=bias_node_spec,
            base_model=base_model_es,
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM]
        )
        
        # Verify tickers are stored
        feature_file = Path(ensemble_dir) / 'features' / 'test_feature.json'
        with open(feature_file, 'r') as f:
            config = json.load(f)
        
        assert 'tickers' in config
        assert set(config['tickers']) == {'ES', 'NQ', 'YM'}
        
        # Load models - should create instances for all tickers
        models = load_feature_base_models(ensemble_dir, 'test_feature')
        
        # Should have 3 models (one per ticker)
        assert len(models) == 3
        
        # Verify each ticker has a model
        assert (Ticker.ES, model_id) in models
        assert (Ticker.NQ, model_id) in models
        assert (Ticker.YM, model_id) in models
        
        # Verify each model has correct ticker
        assert models[(Ticker.ES, model_id)].ticker == Ticker.ES
        assert models[(Ticker.NQ, model_id)].ticker == Ticker.NQ
        assert models[(Ticker.YM, model_id)].ticker == Ticker.YM
        
        # Verify all models share the same binning model config
        es_model = models[(Ticker.ES, model_id)]
        nq_model = models[(Ticker.NQ, model_id)]
        assert es_model.binning_model.n_bins == nq_model.binning_model.n_bins
        assert es_model.binning_model.strategy == nq_model.binning_model.strategy
