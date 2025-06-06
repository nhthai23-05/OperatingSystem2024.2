# demo/policy_loader.py
"""
Policy loader for CPU scheduling algorithms.
Supports loading PPO (Reinforcement Learning) and XGBoost models.
"""
import onnxruntime as ort
import numpy as np
import sys
import os
import cloudpickle


def fix_numpy_compatibility():
    """Fix NumPy compatibility issue for PPO model loading"""
    import numpy
    
    # Patch missing numpy._core module for older model compatibility
    if not hasattr(numpy, '_core'):
        numpy._core = numpy.core
    if 'numpy._core.numeric' not in sys.modules:
        sys.modules['numpy._core.numeric'] = getattr(numpy.core, 'numeric', numpy)
        
    # Patch cloudpickle's import mechanism
    original_loads = cloudpickle.loads
    
    def patched_loads(data):
        if 'numpy._core.numeric' not in sys.modules:
            sys.modules['numpy._core.numeric'] = getattr(numpy.core, 'numeric', numpy)
        return original_loads(data)
    
    cloudpickle.loads = patched_loads


def get_model_path(step_folder, model_name):
    """Get absolute path to model file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, step_folder, model_name)


def load_ppo_policy():
    """Load PPO reinforcement learning model"""
    from stable_baselines3 import PPO
    
    # Fix NumPy compatibility before loading
    fix_numpy_compatibility()
    
    model_path = get_model_path("Step4", os.path.join("best_model", "best_model.zip"))
    print(f"Loading PPO model from: {model_path}")
    return PPO.load(model_path)


def load_xgb_policy():
    """Load XGBoost machine learning model"""
    xgb_path = get_model_path("Step2", "xgb_os_sched.onnx")
    sess = ort.InferenceSession(xgb_path)
    input_name = sess.get_inputs()[0].name

    def predict(obs):
        """Predict next process to schedule using XGBoost"""
        obs_matrix = obs.reshape(-1, 7)  # Reshape to (5 processes, 7 features)
        
        # Select process with minimum remaining time as representative
        idx = np.argmin(obs_matrix[:, 6])  # Index 6: remaining_time
        feature_vector = obs_matrix[idx]
        
        # Pad or truncate features to match model requirements (12 features)
        if len(feature_vector) < 12:
            padded = np.pad(feature_vector, (0, 12 - len(feature_vector)), mode='constant')
        else:
            padded = feature_vector[:12]

        # Run inference
        result = sess.run(None, {input_name: padded.reshape(1, -1)})
        return result[0][0].argmax()

    # Return mock policy object with predict method
    return type("XGBPolicy", (), {
        "predict": lambda self, obs, deterministic=True: [predict(obs)]
    })()


def load_policy(policy_name):
    """
    Load scheduling policy by name.
    
    Args:
        policy_name (str): Policy name ('ppo', 'xgb')
        
    Returns:
        Policy object with predict method
    """
    if policy_name.lower() == "ppo":
        return load_ppo_policy()
    elif policy_name.lower() == "xgb":
        return load_xgb_policy()
    else:
        raise ValueError(f"Unknown policy: {policy_name}. Supported: 'ppo', 'xgb'")
