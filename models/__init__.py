

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
    elif phase == 'phase4':
        from .phase4_model import create_phase4_model
        return create_phase4_model(config)
    elif phase == 'phase5':
        from .phase5_model import create_phase5_model
        return create_phase5_model(config)
    elif phase == 'phase6':
        from .phase6_model import create_phase6_model
        return create_phase6_model(config)
    else:
        raise ValueError(f"Unknown phase: {phase}")