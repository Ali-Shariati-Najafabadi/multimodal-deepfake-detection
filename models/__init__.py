

def get_model(phase, config):
    if phase == 'phase1':
        from .phase1_model_updated import create_phase1_model
        return create_phase1_model(config)
    elif phase == 'phase2':
        from .phase2_model_updated import create_phase2_model
        return create_phase2_model(config)
    elif phase == 'phase3':
        from .phase3_model import create_phase3_model
        return create_phase3_model(config)
    else:
        raise ValueError(f"Unknown phase: {phase}")
